"""Track 1 acoustic five-level pilot; waveform and master segment index are immutable.

conda run --no-capture-output -n qwen3-asr python scripts/analyze_track1_tones.py extract
"""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, math, re
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d, label

ROOT=Path(__file__).resolve().parents[1]
RID='recording_11_20260909'
BASE=ROOT/'generated'/RID
GEN=BASE/'tone_pilot'
OUT=ROOT/'data/annotations/draft'/RID/'tone_pilot'
META=ROOT/'data/metadata/recording_11_track1_tone_normalization.json'
INDEX=ROOT/'data/metadata/recording_11_segments.csv'
SOURCE=BASE/'track_1_16k.wav'
DT=.01
SAMPLES=[8,78,240,351,579]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def five_level(f0,bounds):
    """Left-closed bins using stored Hz edges, avoiding log roundoff at T integers."""
    return np.searchsorted(np.asarray(bounds[1:-1]),f0,side='right')+1

def db_rms_at(y,sr,t):
    # RMS from 30 ms centered windows, no sample-by-sample Python loop.
    z=np.sqrt(np.maximum(uniform_filter1d(y*y,size=round(.03*sr),mode='nearest'),0))
    return 20*np.log10(np.maximum(z[np.minimum((t*sr).astype(int),len(z)-1)],1e-10))

def extract():
    import parselmouth as pm
    GEN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    y,sr=sf.read(SOURCE,dtype='float32')
    print('Extract entire track',len(y)/sr,'seconds; Praat',pm.PRAAT_VERSION,flush=True)
    pitch=pm.Sound(y,sampling_frequency=sr).to_pitch_ac(time_step=DT,pitch_floor=50,
        max_number_of_candidates=15,very_accurate=True,silence_threshold=.03,
        voicing_threshold=.45,octave_cost=.01,octave_jump_cost=.35,
        voiced_unvoiced_cost=.14,pitch_ceiling=600)
    pitch.save_as_binary_file(str(GEN/'track1_raw_ac.Pitch'))
    t=pitch.xs(); f=pitch.selected_array['frequency'];strength=pitch.selected_array['strength']
    rms=db_rms_at(y,sr,t)
    other,other_sr=sf.read(BASE/'track_2_16k.wav',dtype='float32')
    assert other_sr==sr and len(other)==len(y),'Track timelines differ'
    rms2=db_rms_at(other,sr,t)
    voiced=f>0
    jump=np.zeros(len(f),bool)
    pair=voiced[1:]&voiced[:-1]
    jumps=np.where(pair & (np.abs(12*np.log2(np.maximum(f[1:],1)/np.maximum(f[:-1],1)))>7))[0]
    for k in jumps:jump[max(0,k-1):min(len(f),k+3)]=True
    prelim=voiced&(strength>=.6)&(rms>=-45)&~jump
    ids,n=label(prelim); counts=np.bincount(ids)
    valid=prelim&(counts[ids]>=5)
    assert valid.any(),'No frames passed automatic screening'
    logs=np.log(f[valid]);lo,hi=np.exp(np.quantile(logs,[.05,.95]))
    assert hi>lo,'Degenerate normalization range'
    bounds=np.exp(np.linspace(np.log(lo),np.log(hi),6))
    T=np.full(len(f),np.nan);q=np.zeros(len(f),np.uint8)
    T[valid]=5*(np.log(f[valid])-np.log(lo))/(np.log(hi)-np.log(lo))
    q[valid]=five_level(f[valid],bounds).astype(np.uint8)
    dominant=valid&(rms-rms2>=6)
    assert dominant.any(),'No track1-dominant frames for sensitivity check'
    dlogs=np.log(f[dominant]);dlo,dhi=np.exp(np.quantile(dlogs,[.05,.95]))
    minlo,maxhi=float(np.min(f[valid])),float(np.max(f[valid]))
    summary=dict(recording_id=RID,created='2026-09-10',scope='entire_track1; provisional, not speaker-isolated',
        source=str(SOURCE.relative_to(ROOT)),source_sha256=sha(SOURCE),duration_s=len(y)/sr,
        protected_index_sha256=sha(INDEX),parselmouth_version=pm.__version__,praat_version=pm.PRAAT_VERSION,
        extraction=dict(method='raw_autocorrelation',time_step_s=DT,pitch_floor_hz=50,pitch_ceiling_hz=600,
            very_accurate=True,max_number_of_candidates=15,silence_threshold=.03,voicing_threshold=.45,
            octave_cost=.01,octave_jump_cost=.35,voiced_unvoiced_cost=.14),
        quality_filter=dict(strength_min=.6,rms_dbfs_min=-45,rms_window_s=.03,
            adjacent_semitone_jump_max=7,jump_mask='previous/current/next two frames',min_valid_run_frames=5),
        total_frames=len(f),raw_voiced_frames=int(voiced.sum()),reliable_frames=int(valid.sum()),
        reliable_time_s=float(valid.sum()*DT),excluded_jump_frames=int(jump.sum()),
        f0_quantiles_hz={str(p):float(v) for p,v in zip([0,1,5,25,50,75,95,99,100],np.exp(np.quantile(logs,[0,.01,.05,.25,.5,.75,.95,.99,1])))},
        normalization=dict(method='robust_log_range_T; 5th/95th quantiles of log F0',low_hz=float(lo),high_hz=float(hi),
            T_formula='5*(ln(f0)-ln(low))/(ln(high)-ln(low))',five_level='max(1,min(5,floor(clip(T,0,5))+1))',
            boundaries_hz=bounds.tolist(),intervals='[L,b1), [b1,b2), [b2,b3), [b3,b4), [b4,H]; outliers saturated to 1/5',
            below_low=int(np.sum(valid&(f<lo))),above_high=int(np.sum(valid&(f>hi)))),
        log_zscore=dict(mean_log_hz=float(logs.mean()),std_log_hz_population=float(logs.std()),
            purpose='continuous statistical diagnostic, not direct five-level labels'),
        minmax_comparison=dict(low_hz=minlo,high_hz=maxhi,boundaries_hz=np.exp(np.linspace(np.log(minlo),np.log(maxhi),6)).tolist()),
        channel_dominance_diagnostic=dict(threshold_db=6,reference_frames=int(dominant.sum()),
            fraction_of_reliable=float(dominant.sum()/valid.sum()),low_hz=float(dlo),high_hz=float(dhi),
            boundaries_hz=np.exp(np.linspace(np.log(dlo),np.log(dhi),6)).tolist(),
            warning='channel loudness is not speaker diarization; all-track parameters remain primary'),
        levels=[dict(level=i+1,lower_hz=float(bounds[i]),upper_hz=float(bounds[i+1]),
            reliable_frames=int(np.sum(q==i+1))) for i in range(5)])
    save(META,summary)
    np.savez_compressed(GEN/'track1_f0.npz',time_s=t,f0_raw_hz=f,strength=strength,rms_dbfs=rms,
        track2_rms_dbfs=rms2,jump_flag=jump,reliable=valid,T=T,level=q)
    with gzip.open(GEN/'track1_f0_frames.csv.gz','wt',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['time_s','f0_raw_hz','periodicity_strength','rms_dbfs','track2_rms_dbfs','jump_flag','reliable','T_unclipped','five_level'])
        for i in range(len(t)):
            w.writerow([round(t[i],4),round(f[i],3) if voiced[i] else '',round(strength[i],4),round(rms[i],2),round(rms2[i],2),int(jump[i]),int(valid[i]),round(T[i],4) if valid[i] else '',int(q[i]) if valid[i] else ''])
    candidates=[]
    with INDEX.open(encoding='utf-8-sig',newline='') as stream:
        for r in csv.DictReader(stream):
            a,b=float(r['start_sec']),float(r['end_sec']);m=(t>=a)&(t<b)
            chinese=re.findall(r'[\u3400-\u9fff]',r['text'])
            if len(chinese)>=8 and len(chinese)<=30 and 1.8<b-a<8:
                candidates.append(dict(utterance_id=r['utterance_id'],start=a,end=b,text=r['text'],
                    voiced_fraction=float(valid[m].mean()),rms_dbfs=float(np.median(rms[m])),
                    dominance_db=float(np.median((rms-rms2)[m])),median_f0=float(np.median(f[m&valid])) if np.any(m&valid) else None))
    save(GEN/'sample_candidates.json',candidates)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def pilots():
    import os, torch
    from qwen_asr import Qwen3ForcedAligner
    from dataclasses import asdict
    os.environ['HF_HUB_OFFLINE']='1'
    torch.set_num_threads(4)
    y,sr=sf.read(SOURCE,dtype='float32')
    with INDEX.open(encoding='utf-8-sig',newline='') as stream:
        rows={int(r['utterance_id'].rsplit('_',1)[1]):r for r in csv.DictReader(stream)}
    model=Qwen3ForcedAligner.from_pretrained('Qwen/Qwen3-ForcedAligner-0.6B',dtype=torch.bfloat16,
        device_map='cuda:0',local_files_only=True)
    selected=[]
    for num in SAMPLES:
        row=rows[num];a,b=float(row['start_sec']),float(row['end_sec'])
        ia=max(0,round((a-.25)*sr));ib=min(len(y),round((b+.25)*sr))
        clip=y[ia:ib];uid=row['utterance_id']
        sf.write(OUT/f'{uid}_track1.wav',clip,sr,subtype='PCM_16')
        result=model.align(audio=(clip,sr),text=row['text'],language='Chinese')[0]
        tokens=[asdict(w) for w in result.items]
        chars=''.join(re.findall(r'[\u3400-\u9fff]',row['text']))
        assert ''.join(w['text'] for w in tokens)==chars,'Aligner modified transcript!'
        sample=dict(utterance_id=uid,text=row['text'],source_cue_ids=row['source_cue_ids'],
            source_label_start_s=a,source_label_end_s=b,clip_start_global_s=ia/sr,clip_end_global_s=ib/sr,
            clip_file=f'{uid}_track1.wav',alignment_model='Qwen/Qwen3-ForcedAligner-0.6B',
            alignment_language='Chinese; Shanghai dialect not specifically calibrated',
            alignment_status='new_forced_alignment_machine_draft',tokens=tokens)
        save(OUT/f'{uid}_alignment.json',sample);selected.append(sample)
        print('ALIGNED',uid,row['text'],len(tokens),flush=True)
    save(OUT/'pilot_samples.json',selected)

def configure_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    return plt

def compare_filtered(praat_executable):
    import parselmouth as pm
    import subprocess
    path=GEN/'track1_filtered_ac.Pitch'
    if not path.exists():
        result=subprocess.run([praat_executable,'--utf8','--no-pref-files','--run',
            str(ROOT/'scripts/track1_filtered_ac.praat'),str(SOURCE),str(path)],
            check=True,capture_output=True,encoding='utf-8')
        print(result.stdout)
    p=pm.read(str(path));time=p.xs();freq=p.selected_array['frequency'];strength=p.selected_array['strength']
    voiced=freq>0
    d=np.load(GEN/'track1_f0.npz');t=d['time_s'];raw=d['f0_raw_hz'];good=d['reliable']
    nearest=np.clip(np.rint((t-time[0])/p.dx).astype(int),0,len(time)-1)
    match=np.abs(time[nearest]-t)<=.0051
    paired=match&good&(freq[nearest]>0)
    cents=np.zeros(len(t));cents[paired]=np.abs(1200*np.log2(freq[nearest[paired]]/raw[paired]))
    def describe(a,b,name):
        m=(t>=a)&(t<b);both=m&paired
        return dict(label=name,start_s=a,end_s=b,matched_frames=int(np.sum(m&match)),
            raw_voiced_fraction=float(np.mean(raw[m]>0)),filtered_voiced_fraction=float(np.mean(freq[nearest[m]]>0)),
            raw_reliable_fraction=float(np.mean(good[m])),both_filtered_voiced_and_raw_reliable_frames=int(both.sum()),
            median_absolute_cents_difference=float(np.median(cents[both])),p95_absolute_cents_difference=float(np.quantile(cents[both],.95)),
            fraction_difference_above_100_cents=float(np.mean(cents[both]>100)),fraction_difference_above_600_cents=float(np.mean(cents[both]>600)))
    q=np.quantile(freq[voiced],[.01,.05,.5,.95,.99])
    version=subprocess.run([praat_executable,'--utf8','--version'],capture_output=True).stdout.decode('utf-8',errors='replace').strip('\x00\n\r ')
    summary=dict(extractor='standalone Praat filtered autocorrelation',praat_version=version,input=str(SOURCE.relative_to(ROOT)),
        time_step_s=.01,pitch_floor_hz=50,pitch_top_hz=600,max_number_of_candidates=15,very_accurate=False,
        attenuation_at_top=.03,silence_threshold=.09,voicing_threshold=.5,octave_cost=.055,octave_jump_cost=.35,voiced_unvoiced_cost=.14,
        xmin=p.xmin,xmax=p.xmax,first_frame_s=float(time[0]),last_frame_s=float(time[-1]),frames=len(time),
        voiced_frames=int(voiced.sum()),voiced_fraction=float(voiced.mean()),
        f0_quantile_hz={key:float(value) for key,value in zip(['q01','q05','q50','q95','q99'],q)},
        postfilter='None; sensitivity comparison only, do not mix algorithms in the primary reference',
        comparison_time_matching='nearest frame center <=5.1ms, only raw-screened and filtered-voiced common frames',
        full_track_comparison=describe(0,p.xmax,'full_track'),pilot_comparison=[])
    if (OUT/'pilot_samples.json').exists():
        for sample in load(OUT/'pilot_samples.json'):
            st=describe(sample['source_label_start_s'],sample['source_label_end_s'],sample['utterance_id']);st['text']=sample['text'];summary['pilot_comparison'].append(st)
    save(GEN/'filtered_comparison_summary.json',summary)
    with (GEN/'filtered_comparison.tsv').open('w',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream,delimiter='\t');w.writerow(['time_s','f0_hz','strength','voiced'])
        for i in range(len(time)):w.writerow([time[i],freq[i] if voiced[i] else '',strength[i],int(voiced[i])])
    print('FILTERED COMPARISON',summary['full_track_comparison'])

def textgrid(sample,characters):
    duration=sample['clip_end_global_s']-sample['clip_start_global_s']
    offset=sample['clip_start_global_s']
    def intervals(which):
        entries=[];cursor=0.
        if which in ['word','AP']:return [(0,duration,'')]
        for c in characters:
            a,b=c['start_global_s']-offset,c['end_global_s']-offset
            if b<=a:continue
            assert a>=cursor-1e-6 and 0<=a<b<=duration+1e-6
            if a>cursor:entries.append((cursor,a,''))
            content=c['char'] if which=='syllable' else c['tone5_points'] if which=='tone5_auto' else 'machine_draft;'+c['qc_flags']
            entries.append((a,b,content));cursor=b
        if cursor<duration:entries.append((cursor,duration,''))
        return entries
    lines=['File type = "ooTextFile"','Object class = "TextGrid"','', 'xmin = 0',f'xmax = {duration:.9f}',
        'tiers? <exists>','size = 6','item []:']
    for k,tier in enumerate(['syllable','word','AP','misc','tone5_auto'],1):
        ii=intervals(tier)
        lines.extend([f'    item [{k}]:','        class = "IntervalTier"',f'        name = "{tier}"',
            '        xmin = 0',f'        xmax = {duration:.9f}',f'        intervals: size = {len(ii)}'])
        for j,(a,b,s) in enumerate(ii,1):
            s=s.replace('"','""')
            lines.extend([f'        intervals [{j}]:',f'            xmin = {a:.9f}',f'            xmax = {b:.9f}',f'            text = "{s}"'])
    collapsed={}
    for c in characters:
        if c['end_global_s']<=c['start_global_s']:
            pos=c['start_global_s']-offset
            collapsed.setdefault(pos,[]).append(f"{c['char_index']}:{c['char']} zero_duration")
    lines.extend(['    item [6]:','        class = "TextTier"','        name = "alignment_issues"',
        '        xmin = 0',f'        xmax = {duration:.9f}',f'        points: size = {len(collapsed)}'])
    for j,(pos,marks) in enumerate(sorted(collapsed.items()),1):
        lines.extend([f'        points [{j}]:',f'            number = {pos:.9f}',f'            mark = "{"; ".join(marks)}"'])
    (OUT/f"{sample['utterance_id']}_track1.TextGrid").write_text('\n'.join(lines)+'\n',encoding='utf-8')

def export_pilots():
    plt=configure_plot()
    meta=load(META);lo=meta['normalization']['low_hz'];hi=meta['normalization']['high_hz']
    bounds=meta['normalization']['boundaries_hz']
    d=np.load(GEN/'track1_f0.npz');t=d['time_s'];f=d['f0_raw_hz'];good=d['reliable']
    records=[];summary=[]
    for sample in load(OUT/'pilot_samples.json'):
        uid=sample['utterance_id'];offset=sample['clip_start_global_s'];chars=[]
        for i,w in enumerate(sample['tokens'],1):
            a=offset+w['start_time'];b=offset+w['end_time'];mask=(t>=a)&(t<b)
            flags=[];valid=mask&good
            if w['end_time']<=w['start_time']:flags.append('zero_duration_alignment')
            if w['end_time']-w['start_time']<.08-1e-7:flags.append('short_alignment')
            zero_nearby=any(v['end_time']<=v['start_time'] and abs(v['start_time']-w['start_time'])<=.16+1e-7 for v in sample['tokens'])
            if zero_nearby:flags.append('alignment_near_collapsed_token')
            ids,n=label(valid);counts=np.bincount(ids);counts[0]=0
            run=(ids==np.argmax(counts)) if n else np.zeros(len(t),bool)
            inds=np.flatnonzero(run);ratio=float(valid.sum()/max(mask.sum(),1))
            if ratio<.35:flags.append('low_voiced_fraction')
            if len(inds)<5:flags.append('insufficient_contiguous_f0')
            if valid.sum() and len(inds)/valid.sum()<.8:flags.append('fragmented_voicing')
            if np.any(d['jump_flag'][mask]):flags.append('f0_jump_nearby')
            hz=[];levels=[];centers=[]
            if len(inds)>=5 and 'zero_duration_alignment' not in flags and 'short_alignment' not in flags and ratio>=.35:
                left,right=t[inds[0]],t[inds[-1]]
                for frac in [.2,.5,.8]:
                    center=left+frac*(right-left);window=max(.015,.1*(right-left))
                    ff=f[run&(t>=center-window)&(t<=center+window)]
                    if len(ff)<2:hz.append(None);levels.append(None);centers.append(float(center));continue
                    value=float(np.median(ff));hz.append(value);centers.append(float(center))
                    T=5*(np.log(value)-np.log(lo))/(np.log(hi)-np.log(lo))
                    levels.append(int(five_level(value,bounds)))
            else:hz=[None]*3;levels=[None]*3;centers=[None]*3
            if any(v is None for v in levels):flags.append('tone_unavailable')
            if any(v is not None and (v<lo or v>hi) for v in hz):flags.append('saturated_level')
            points=''.join(str(v) if v is not None else '?' for v in levels)
            # Three fixed-position numbers, not a guessed lexical/citation tone.
            record=dict(utterance_id=uid,char_index=i,char=w['text'],
                start_global_s=round(a,6),end_global_s=round(b,6),
                start_clip_s=w['start_time'],end_clip_s=w['end_time'],
                voiced_fraction=round(ratio,4),reliable_frames=int(valid.sum()),
                main_voiced_run_start_s=float(t[inds[0]]) if len(inds) else None,
                main_voiced_run_end_s=float(t[inds[-1]]) if len(inds) else None,
                f0_20_hz=hz[0],f0_50_hz=hz[1],f0_80_hz=hz[2],
                sample_20_global_s=centers[0],sample_50_global_s=centers[1],sample_80_global_s=centers[2],
                tone5_points=points,alignment_status='machine_draft',tone_status='machine_acoustic_estimate' if '?' not in points else 'unavailable',
                qc_flags=';'.join(flags),normalization_file=str(META.relative_to(ROOT)).replace('\\','/'))
            chars.append(record);records.append(record)
        assert all(c['end_global_s']<=n['start_global_s']+1e-6 for c,n in zip(chars,chars[1:]))
        textgrid(sample,chars)
        y,sr=sf.read(OUT/sample['clip_file']);x=np.arange(len(y))/sr+offset
        fig,(axw,ax)=plt.subplots(2,1,figsize=(15,6),sharex=True,gridspec_kw={'height_ratios':[1,3]},layout='constrained')
        axw.plot(x[::10],y[::10],lw=.45,color='#4b5563');axw.set_ylabel('波形')
        axw.set_title(f"{uid}  音轨1\n{sample['text']}",fontsize=12)
        mask=(t>=offset)&(t<sample['clip_end_global_s'])
        ax.scatter(t[mask&(f>0)],f[mask&(f>0)],s=7,color='#aeb6c4',label='Praat原始F0')
        ax.scatter(t[mask&good],f[mask&good],s=8,color='#216b9a',label='自动筛选保留')
        for k in range(5):
            ax.axhspan(bounds[k],bounds[k+1],color=['#e0edff','#edf5ff'][k%2],alpha=.6)
            ax.text(offset+.01,(bounds[k]+bounds[k+1])/2,str(k+1),fontsize=10,color='#64748b')
        for boundary in bounds:ax.axhline(boundary,color='#9eabc0',lw=.6,ls='--')
        ymax=max(hi*1.15,float(np.quantile(f[mask&good],.99))*1.1) if np.any(mask&good) else hi*1.15
        label_groups=[]
        for c in chars:
            a,b=c['start_global_s'],c['end_global_s'];mid=(a+b)/2
            ax.axvline(a,color='#a6abb4',lw=.5,alpha=.6)
            if label_groups and abs(label_groups[-1][0]-mid)<1e-6:
                label_groups[-1][1].append(c)
            else:label_groups.append([mid,[c]])
            for key in ['20','50','80']:
                if c[f'f0_{key}_hz'] is not None:ax.plot(c[f'sample_{key}_global_s'],c[f'f0_{key}_hz'],'o',ms=3,color='#c45c2a')
        previous=-np.inf;lane=0
        for mid,group in label_groups:
            if mid-previous<(sample['clip_end_global_s']-offset)*.037:lane=1-lane
            else:lane=0
            label_text='/'.join(c['char'] for c in group)+'\n'+'/'.join(c['tone5_points'] for c in group)
            ypos=1.07+.17*lane
            ax.annotate(label_text,xy=(mid,ymax),xytext=(mid,ypos),textcoords=('data','axes fraction'),
                ha='center',va='bottom',fontsize=8,color='#963c25' if any(c['qc_flags'] for c in group) else '#155d40',
                arrowprops=dict(arrowstyle='-',color='#c4c8cd',lw=.5))
            previous=mid
        ax.set_ylim(max(35,lo*.65),ymax);ax.set_xlim(offset,sample['clip_end_global_s']);ax.set_ylabel('F0（Hz）');ax.set_xlabel('Audition工程绝对时间（秒）')
        ax.legend(loc='lower right',fontsize=8)
        fig.savefig(GEN/f'{uid}_alignment.png',dpi=160);plt.close(fig)
        summary.append(dict(utterance_id=uid,text=sample['text'],characters=len(chars),
            with_tone=sum('?' not in c['tone5_points'] for c in chars),
            display=' '.join(c['char']+'['+c['tone5_points']+']' for c in chars),
            flagged=sum(bool(c['qc_flags']) for c in chars)))
    save(OUT/'character_tones.json',records);save(OUT/'pilot_summary.json',summary)
    with (OUT/'character_tones.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    assert sha(INDEX)==meta['protected_index_sha256'],'Master index was modified!'
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def report():
    import parselmouth as pm
    import importlib.metadata as versions
    plt=configure_plot()
    meta=load(META); d=dict(np.load(GEN/'track1_f0.npz'));good=d['reliable'];f=d['f0_raw_hz'];t=d['time_s']
    bounds=meta['normalization']['boundaries_hz'];lo,hi=bounds[0],bounds[-1]
    meta['normalization']['five_level']='max(1,min(5,floor(clip(T,0,5))+1))'
    meta['normalization']['intervals']='(-infinity,b1), [b1,b2), [b2,b3), [b3,b4), [b4,infinity); accepted voiced frames only'
    meta['normalization']['boundary_convention']='left-closed T bins [0,1),[1,2),[2,3),[3,4),[4,5]; saturation at both tails'
    meta['normalization']['implementation']='numpy.searchsorted(unrounded_Hz_boundaries[1:-1],f0,side=right)+1; avoids log-roundoff at exact bin edges'
    meta['normalization']['reference']='Zhang 2018 PACLIC32 Eq.7 p826, robust log-P5/P95 endpoints are this project adaptation'
    meta['normalization']['quantile_estimator']='numpy.quantile(log_f0,[.05,.95],method=linear), frame-weighted'
    meta['reliability_status']='automatic_screen_only; not manually corrected'
    q=np.zeros(len(f),np.uint8);q[good]=five_level(f[good],bounds).astype(np.uint8);d['level']=q
    meta['levels']=[dict(level=k+1,calibration_lower_hz=bounds[k],calibration_upper_hz=bounds[k+1],
        accepted_lower_inclusive_hz=None if k==0 else bounds[k],accepted_upper_exclusive_hz=None if k==4 else bounds[k+1],
        passed_frames=int(np.sum(q==k+1))) for k in range(5)]
    meta['automatic_qc_extremes']=dict(below_70_hz=int(np.sum(good&(f<70))),above_250_hz=int(np.sum(good&(f>250))),
        note='possible creak/octave error/background sources; retained with caveats, robust normalization limits leverage')
    meta['ten_minute_profiles']=[]
    for start in range(0,3600,600):
        mask=good&(t>=start)&(t<start+600)
        meta['ten_minute_profiles'].append(dict(start_s=start,end_s=min(start+600,meta['duration_s']),
            passed_frames=int(mask.sum()),p05_hz=float(np.exp(np.quantile(np.log(f[mask]),.05))),
            p50_hz=float(np.median(f[mask])),p95_hz=float(np.exp(np.quantile(np.log(f[mask]),.95)))))
    compare=load(GEN/'filtered_comparison_summary.json')
    meta['filtered_ac_comparison']=compare
    save(META,meta);np.savez_compressed(GEN/'track1_f0.npz',**d)
    with gzip.open(GEN/'track1_f0_frames.csv.gz','wt',encoding='utf-8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['time_s','f0_raw_hz','periodicity_strength','rms_dbfs','track2_rms_dbfs','jump_flag','passed_auto_screen','T_unclipped','five_level','semitones_relative_median','log_f0_zscore','saturated'])
        median=meta['f0_quantiles_hz']['50'];mean=meta['log_zscore']['mean_log_hz'];std=meta['log_zscore']['std_log_hz_population']
        for i in range(len(t)):
            v=bool(good[i]);w.writerow([round(t[i],6),round(f[i],3) if f[i]>0 else '',round(d['strength'][i],4),round(d['rms_dbfs'][i],2),round(d['track2_rms_dbfs'][i],2),int(d['jump_flag'][i]),int(v),
                round(d['T'][i],6) if v else '',int(q[i]) if v else '',round(12*np.log2(f[i]/median),6) if v else '',round((np.log(f[i])-mean)/std,6) if v else '',int(f[i]<lo or f[i]>hi) if v else ''])
    # Compact binary Pitch retains candidates and is readable by standalone Praat.
    pitch=pm.read(str(GEN/'track1_raw_ac.Pitch'));pitch.save_as_binary_file(str(GEN/'track1_raw_ac.Pitch'))
    references=[]
    for k in range(5):
        freq=math.sqrt(bounds[k]*bounds[k+1]);sr=16000;x=.2*np.sin(2*np.pi*freq*np.arange(sr)/sr)
        pp=pm.Sound(x,sampling_frequency=sr).to_pitch_ac(time_step=.01,pitch_floor=50,pitch_ceiling=600,very_accurate=True)
        hz=pp.selected_array['frequency'];hz=float(np.median(hz[hz>0]))
        value=int(five_level(hz,bounds))
        assert value==k+1 and abs(hz-freq)/freq<.01
        references.append(dict(target_hz=freq,measured_hz=hz,expected_level=k+1,actual_level=value))
    silent=pm.Sound(np.zeros(16000),sampling_frequency=16000).to_pitch_ac(pitch_floor=50,pitch_ceiling=600)
    assert np.all(silent.selected_array['frequency']==0)
    for k,boundary in enumerate(bounds[1:-1],1):
        assert five_level(boundary,bounds)==k+1
        assert five_level(np.nextafter(boundary,-np.inf),bounds)==k
    for p in OUT.glob('*.TextGrid'):assert pm.read(str(p)).class_name=='TextGrid'
    chars=load(OUT/'character_tones.json');summary=load(OUT/'pilot_summary.json')
    assert sha(INDEX)==meta['protected_index_sha256']
    assert sha(SOURCE)==meta['source_sha256']
    original=load(ROOT/'data/metadata'/f'{RID}.json')
    raw_ok=all(sha(ROOT/item['path'])==item['sha256'] for item in original['files'])
    assert raw_ok
    validation=dict(pure_tone_tests=references,silence_has_no_f0=True,exact_bin_edge_checks_passed=True,textgrid_files_readable=5,
        source_and_index_hashes_unchanged=True,raw_archive_and_recording_hashes_verified=raw_ok,
        sample_count=len(summary),characters=len(chars),characters_with_three_points=sum('?' not in c['tone5_points'] for c in chars),
        collapsed_alignment_tokens=sum(c['end_clip_s']<=c['start_clip_s'] for c in chars),
        available_does_not_mean_verified=True,
        versions={name:versions.version(name) for name in ['numpy','scipy','soundfile','praat-parselmouth','qwen-asr','torch','matplotlib']})
    save(GEN/'validation.json',validation)
    # Overview: frame histogram + ten-minute quantiles with fixed global bounds.
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    ax1.hist(f[good],bins=np.geomspace(45,650,110),color='#4384af',alpha=.9)
    ax1.set_xscale('log');ax1.set_xticks([50,75,100,150,200,300,500]);ax1.set_xticklabels(['50','75','100','150','200','300','500'])
    from matplotlib.ticker import NullFormatter
    ax1.xaxis.set_minor_formatter(NullFormatter())
    for b in bounds:ax1.axvline(b,color='#b34922',lw=.8,ls='--')
    ax1.set_title('音轨1通过自动筛选的F0分布');ax1.set_xlabel('F0（Hz，对数轴）');ax1.set_ylabel('帧数')
    profiles=meta['ten_minute_profiles'];xx=[(x['start_s']+x['end_s'])/120 for x in profiles]
    ax2.fill_between(xx,[x['p05_hz'] for x in profiles],[x['p95_hz'] for x in profiles],color='#bcd7eb',label='每10分钟P5–P95')
    ax2.plot(xx,[x['p50_hz'] for x in profiles],'o-',color='#28668c',label='每10分钟中位数')
    for bound in [lo,hi]:ax2.axhline(bound,ls='--',color='#b34922',lw=.9)
    ax2.set_title('固定全轨阈值与分段调域');ax2.set_xlabel('全轨位置（分钟）');ax2.set_ylabel('F0（Hz）');ax2.legend(fontsize=8)
    fig.savefig(GEN/'track1_normalization_overview.png',dpi=170);plt.close(fig)
    lines=['# 音轨1五度标记与汉字匹配试验报告','',
        '日期：2026-09-10。已完成完整音轨1的F0提取和归一化，并试做5段、72个汉字的时间匹配。原始音频、用户修订稿及593行重要索引均未改动。',
        '本报告的数字是连续口语的**声学五度候选**，不是已经听辨确认的本调、音系调类或AP边界。',
        '## 1. 全轨范围与五度阈值','',
        f"分析源是完整 `track_1_16k.wav`，时长{meta['duration_s']:.3f}秒，由原工程音轨1等时间重建；音量未归一化、未变调、未删静音。Praat raw AC每10 ms取一帧，共{len(t):,}帧，其中{int(good.sum()):,}帧通过自动筛选，累计约{good.sum()*.01/60:.2f}分钟。清音、静音与筛除帧保留为缺测，不作为0 Hz参加对数运算。",'',
        f"采用对数F0的P5/P95：L={lo:.6f} Hz，H={hi:.6f} Hz。T=5·ln(F0/L)/ln(H/L)，d=min(5,max(1,floor(clip(T,0,5))+1))。五等宽区间在对数轴上定义，端点采用左闭约定。参考方法与其他方案见[方法笔记](10_五度标记与F0归一化方法笔记.md)。",'',
        '| 五度 | 本次通过筛选的F0实际分类区间（Hz） | 标定范围内的区间（Hz） |','|---|---|---|']
    for i in range(5):
        actual=f'F0 < {bounds[1]:.3f}' if i==0 else f'F0 ≥ {bounds[4]:.3f}' if i==4 else f'{bounds[i]:.3f} ≤ F0 < {bounds[i+1]:.3f}'
        lines.append(f'| {i+1} | {actual} | {bounds[i]:.3f}–{bounds[i+1]:.3f} |')
    lines += ['',f"低于{lo:.3f}或高于{hi:.3f}的通过筛选帧分别保留在1/5并标记饱和；它们不是被删除的无声帧。P5/P95各排除约5%尾部只是估计声域的工程选择，并不是所有材料通用的五度标准。",'',
        '![全轨F0与归一化范围](../generated/recording_11_20260909/tone_pilot/track1_normalization_overview.png)',
        '## 2. 质量与方法敏感性','',
        f"最低/最高通过筛选值为{meta['minmax_comparison']['low_hz']:.2f}/{meta['minmax_comparison']['high_hz']:.2f} Hz，明显比85–156 Hz的主体声域宽。少量极端值可能来自嘎裂、倍/半频、串音或其他声源；本轮不手工宣称已校正。直接用全局最小/最大会让五度边界变成：{', '.join(f'{b:.2f}' for b in meta['minmax_comparison']['boundaries_hz'])} Hz。完整对照保存在参数JSON。",'',
        f"在通过筛选帧中，轨1能量比轨2至少高6 dB的帧占{meta['channel_dominance_diagnostic']['fraction_of_reliable']:.2%}；该子集P5/P95为{meta['channel_dominance_diagnostic']['low_hz']:.3f}/{meta['channel_dominance_diagnostic']['high_hz']:.3f} Hz，接近全轨结果。声道优势只能帮助排查串音，不能证明整轨只有同一说话人。",'',
        f"另用本机Praat 6.6.30 filtered AC对全轨作对照，未施加主流程后筛的P5/P95约{compare['f0_quantile_hz']['q05']:.2f}/{compare['f0_quantile_hz']['q95']:.2f} Hz。两法共同有声且主流程通过筛选的{compare['full_track_comparison']['both_filtered_voiced_and_raw_reliable_frames']:,}帧，F0差绝对值中位{compare['full_track_comparison']['median_absolute_cents_difference']:.2f} cents，P95={compare['full_track_comparison']['p95_absolute_cents_difference']:.2f} cents；100 cents=1半音。两法清浊覆盖不同，不能将算法一致性当作人工准确率。",'',
        '自动筛选要求周期性强度≥0.6、30 ms窗RMS≥−45 dBFS、连续保留至少5帧，并屏蔽相邻帧超过7半音的跳变附近。持续倍频/半频不一定被此规则发现。raw AC very_accurate模式在50 Hz下用较长分析窗，跨字平滑与相邻字污染仍可能发生。',
        '## 3. 五段逐字匹配','',
        f"从现有索引选取五个不同位置、轨1占优且语句相对完整的片段，测试并非随机抽样，不可外推准确率。每段前后各留250 ms；用修订文字和音轨1重新调用Qwen3-ForcedAligner。共{len(chars)}字，{validation['characters_with_three_points']}字取得三个采样数字，{len(chars)-validation['characters_with_three_points']}字保留???；其中{validation['collapsed_alignment_tokens']}字为零时长对齐。所有字的边界仍是机器候选。",'',
        '全局时间 = 本次切片实际起点 + 对齐模型的局部时间。F0使用Praat实际帧中心，不把帧号乘10 ms当全局时间。模型给出的时间多以80 ms为步长；文本中显示毫秒仅为坐标格式。',
        '每个字区间内寻找最长连续的通过筛选有声段，在该段20%、50%、80%位置附近取局部F0中位数，再转换为三个五度数字。三个位置对应有声段，并非整个汉字区间的等长三分。不足5帧、字长小于80 ms或有声覆盖不足35%等情形不强行补值。',
        '例如444表示三个采样位置均落第4度，不额外宣称该字本调是44。字体图中棕色表示存在自动质检标记，绿色仅表示未触发这些规则；都未经过人工听校。','']
    for sample in summary:
        uid=sample['utterance_id'];cc=[c for c in chars if c['utterance_id']==uid]
        lines += [f"### {uid.rsplit('_',1)[-1]}：{sample['text']}",'',sample['display'],'',
            f"[音轨1片段](../data/annotations/draft/{RID}/tone_pilot/{uid}_track1.wav) · [Praat标注](../data/annotations/draft/{RID}/tone_pilot/{uid}_track1.TextGrid)",'',
            f'![F0与汉字对齐](../generated/{RID}/tone_pilot/{uid}_alignment.png)','',
            '| 字 | 全轨起止秒 | F0 20/50/80%（Hz） | 三点五度 | 待核标记 |','|---|---|---|---|---|']
        for c in cc:
            ff='/'.join('—' if c[f'f0_{k}_hz'] is None else f"{c[f'f0_{k}_hz']:.1f}" for k in ['20','50','80'])
            lines.append(f"| {c['char_index']}.{c['char']} | {c['start_global_s']:.3f}–{c['end_global_s']:.3f} | {ff} | {c['tone5_points']} | {c['qc_flags'] or '未触发规则；仍待听校'} |")
        lines.append('')
    lines += ['## 4. 如何继续复核','',
        '1. 在Praat同时打开同名WAV与TextGrid，先听字音并修正 `syllable` 的机器边界；不要先把数字当作答案。',
        '2. `word`、`AP`层保持空白；`tone5_auto`只是测量候选。零时长字放在 `alignment_issues` 点层，并在CSV/JSON完整保留，未虚构时长。',
        '3. 对比波形/语谱图与Pitch候选，处理持续倍频、弱声、嘎裂、串音和重叠。修改字边界后重新计算对应有声段的三点读数。',
        '4. 若确认音轨含多说话人，应先按说话人重建参考分布，再确定正式五度阈值。不要在每个字或每句话内单独拉满1–5。','',
        '详见[Praat与Python操作笔记](11_Praat与Python音频处理操作笔记.md)。','',
        '## 5. 文件与重跑','',
        f'- [全轨归一化参数](../data/metadata/recording_11_track1_tone_normalization.json)：公式、Hz边界、筛选参数、版本与对照。',
        f'- [逐字CSV](../data/annotations/draft/{RID}/tone_pilot/character_tones.csv) / [逐字JSON](../data/annotations/draft/{RID}/tone_pilot/character_tones.json)：保留全部72字、起止坐标、缺测、原F0与状态。',
        f'- `generated/{RID}/tone_pilot/track1_f0_frames.csv.gz`：全轨每帧原F0、筛选状态、T、五度、半音和log-z；静音不赋五度。',
        f'- `generated/{RID}/tone_pilot/track1_raw_ac.Pitch`、`track1_filtered_ac.Pitch`：Praat可读取的候选对象。',
        f'- `generated/{RID}/tone_pilot/validation.json`：五个已知基频纯音、静音测试、TextGrid读取与索引/原件哈希检查。',
        '- 程序入口 `scripts/analyze_track1_tones.py`，按 `extract` → `pilots` → `filtered` → `export` → `report` 执行；`filtered`阶段读取或调用本机Praat生成对照对象和统计。缓存与主索引分开，AP和L0–L4状态没有改为已完成。','',
        '本轮结论：可以批量测量全轨F0并产生五度候选，也可以按新字级时间尝试匹配汉字；目前字级强制对齐是主要人工复核点，尚不足以把全部逐字数字直接用作正式标注。','']
    (ROOT/'docs/12_音轨1五度标记试验报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','pilots','filtered','export','report'])
    p.add_argument('--praat',default='Praat.exe');a=p.parse_args()
    if a.stage=='filtered':compare_filtered(a.praat)
    else:{'extract':extract,'pilots':pilots,'export':export_pilots,'report':report}[a.stage]()
