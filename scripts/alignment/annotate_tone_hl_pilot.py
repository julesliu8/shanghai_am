"""Acoustic five-level values and relative H/L candidates on five aligned clips.

Roberts (2020) informs target interpretation, not numerical detection thresholds.
No automatic H*, L*, La, AP, or gold labels are inferred from local extrema alone.
"""
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter, find_peaks
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/annotations/draft/aligner_v1/character_pilot_v2'
OUT=ROOT/'data/annotations/draft/aligner_v1/tone_hl_pilot_v1'
GEN=ROOT/'generated/aligner_v1/tone_hl_pilot_v1'
F0ROOT=ROOT/'generated/recording_11_20260909/tone_pilot'
META=ROOT/'data/metadata/recording_11_track1_tone_normalization.json'
PARAMS=dict(frame_s=.01,smoothing_median_frames=5,smoothing_savgol_frames=9,
            interpolation_max_gap_s=.03,min_voiced_span_s=.05,
            target_min_prominence_st=.8,target_min_distance_s=.12,
            endpoint_min_change_st=1.2,endpoint_context_s=.25,
            plateau_min_duration_s=.10,plateau_max_range_st=.30,
            algorithm_disagreement_max_st=1.,tone_sampling_fractions=[.2,.5,.8])


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def runs(mask):
    edge=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)))


def smooth_track(t,f,valid):
    st=np.full(len(t),np.nan);st[valid]=12*np.log2(f[valid]/100.)
    for a,b in runs(~valid):
        if a>0 and b<len(t) and b-a<=3 and valid[a-1] and valid[b]:
            st[a:b]=np.interp(t[a:b],[t[a-1],t[b]],[st[a-1],st[b]])
    out=np.full_like(st,np.nan)
    for a,b in runs(np.isfinite(st)):
        if b-a<5:continue
        values=median_filter(st[a:b],size=5,mode='nearest')
        window=min(9,len(values) if len(values)%2 else len(values)-1)
        out[a:b]=savgol_filter(values,window,2,mode='interp') if window>=5 else values
    return out


def targets(t,smoothed,valid):
    candidates=[]
    for a,b in runs(np.isfinite(smoothed)):
        if b-a<10:continue
        y=smoothed[a:b]
        for sign,label in [(1,'H'),(-1,'L')]:
            peaks,props=find_peaks(sign*y,prominence=PARAMS['target_min_prominence_st'],distance=12)
            for peak,prom in zip(peaks,props['prominences']):
                index=a+int(peak)
                if not valid[index] or peak<3 or peak>=len(y)-3:continue
                candidates.append(dict(index=index,label=label,prominence_st=float(prom),kind='two_sided_extremum',edge_context=False))
        # Endpoints are provisional one-sided observations, not phrase-boundary tones.
        width=min(25,len(y)//2)
        for index,near,far in [(a+2,float(np.median(y[:5])),float(np.median(y[width-3:width+2]))),
                               (b-3,float(np.median(y[-5:])),float(np.median(y[-width-2:-width+3])) )]:
            delta=near-far
            if abs(delta)>=PARAMS['endpoint_min_change_st'] and valid[index]:
                candidates.append(dict(index=index,label='H' if delta>0 else 'L',prominence_st=abs(delta),kind='one_sided_edge_candidate',edge_context=True))
    chosen=[]
    for event in sorted(candidates,key=lambda e:e['prominence_st'],reverse=True):
        if any(e['label']==event['label'] and abs(t[e['index']]-t[event['index']])<.12 for e in chosen):continue
        index=int(event['index']);event['index']=index;a=index;b=index+1
        while a>0 and np.isfinite(smoothed[a-1]) and abs(smoothed[a-1]-smoothed[index])<=.15:a-=1
        while b<len(t) and np.isfinite(smoothed[b]) and abs(smoothed[b]-smoothed[index])<=.15:b+=1
        plateau=t[b-1]-t[a]>=PARAMS['plateau_min_duration_s']
        event.update(time_s=float(t[index]),f0_smoothed_hz=float(100*2**(smoothed[index]/12)),
                     plateau=bool(plateau),plateau_span_s=[float(t[a]),float(t[b-1])] if plateau else None,
                     label_status='acoustic_target_candidate_not_Sh_ToBI_category')
        chosen.append(event)
    return sorted(chosen,key=lambda e:e['time_s'])


def measure_character(c,t,f,valid,normalization,events,origin):
    c=dict(c)
    inside=(t>=c['start_s'])&(t<c['end_s'])
    indices=np.flatnonzero(inside&valid)
    spans=[(a,b) for a,b in runs(inside&valid) if b-a>=5]
    c.update(valid_f0_frames=int(len(indices)),voiced_coverage=float(len(indices)/max(1,inside.sum())),
             tone5_candidate=None,tone5_label='???',f0_three_points_hz=None,sample_times_s=[],
             sampling_span_s=None,tone_status='insufficient_reliable_f0')
    if spans:
        a,b=max(spans,key=lambda ab:ab[1]-ab[0])
        times=t[a]+np.array([.2,.5,.8])*(t[b-1]-t[a])
        values=[float(np.median(f[np.flatnonzero((t>=q-.02)&(t<=q+.02)&inside&valid)])) for q in times]
        degrees=np.searchsorted(normalization['boundaries_hz'][1:-1],values,side='right')+1
        candidate=''.join(map(str,degrees))
        weak=c['annotation_status']=='manual_boundary_needed'
        c.update(tone5_candidate=candidate,tone5_label=candidate+'?' if weak else candidate,
                 f0_three_points_hz=values,sample_times_s=[float(x) for x in times],
                 sample_times_global_s=[float(x+origin) for x in times],
                 sampling_span_s=[float(t[a]),float(t[b-1])],
                 tone_status='tentative_window_pending_character_boundary' if weak else 'machine_acoustic_tone_candidate',
                 outside_global_reference=any(v<normalization['low_hz'] or v>normalization['high_hz'] for v in values))
    own=[e for e in events if e.get('char_index')==c['index']]
    c['hl_events']=[dict(label=e['label'],time_s=e['time_s'],kind=e['kind'],plateau=e['plateau']) for e in own]
    c['hl_label']=' '.join(e['label']+('?' if e['edge_context'] or c['annotation_status']=='manual_boundary_needed' else '') for e in own) or '·'
    c['hl_status']='candidate_targets' if own else ('insufficient_f0' if not spans else 'no_distinct_target_or_transition')
    return c


def write_grid(result):
    path=OUT/(result['utterance_id']+'.TextGrid')
    g=parselmouth.read(str(SOURCE/(result['utterance_id']+'.TextGrid')))
    n=int(call(g,'Get number of tiers'))
    for name in ['tone5_auto','tone5_tentative']:
        n+=1;call(g,'Insert interval tier',n,name)
        selected=[c for c in result['characters'] if c['tone5_candidate'] and
                  (c['annotation_status']!='manual_boundary_needed' if name=='tone5_auto' else c['annotation_status']=='manual_boundary_needed')]
        for cut in sorted({x for c in selected for x in (c['start_s'],c['end_s']) if 0<x<result['duration_s']}):call(g,'Insert boundary',n,cut)
        for c in selected:call(g,'Set interval text',n,call(g,'Get interval at time',n,(c['start_s']+c['end_s'])/2),c['tone5_label'])
    for name in ['HL_auto','HL_uncertain']:
        n+=1;call(g,'Insert point tier',n,name)
        for e in result['hl_events']:
            uncertain=e['edge_context'] or e['mapping_status']!='automatic_character_candidate'
            if uncertain!=(name=='HL_uncertain'):continue
            label=e['label']+('?' if uncertain else '')
            call(g,'Insert point',n,e['time_s'],label)
    call(g,'Save as text file',str(path));parselmouth.read(str(path))


def plot(result,t,f,valid,smoothed):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,1,figsize=(17,6),sharex=True,layout='constrained',height_ratios=[4,1])
    ax=axs[0];ax.set_yscale('log',base=2);ax.set_ylim(55,230)
    edges=[55]+result['normalization']['boundaries_hz'][1:-1]+[230]
    for i,(a,b) in enumerate(zip(edges,edges[1:]),1):
        ax.axhspan(a,b,color=['#dbeafe','#e0f2fe','#f0fdf4','#fef3c7','#fee2e2'][i-1],alpha=.65)
        ax.text(result['duration_s']+.015,np.sqrt(a*b),str(i),va='center',fontsize=11)
    ax.scatter(t[valid],f[valid],s=7,color='#64748b',alpha=.5,label='可靠F0')
    ax.plot(t,100*2**(smoothed/12),lw=1.5,color='#183f4b',label='平滑F0')
    for e in result['hl_events']:
        color='#b54537' if e['label']=='H' else '#286bad'
        uncertain=e['edge_context'] or e['mapping_status']!='automatic_character_candidate'
        ax.scatter(e['time_s'],e['f0_smoothed_hz'],marker='^' if e['label']=='H' else 'v',s=50,color=color,zorder=4)
        ax.annotate(e['label']+('?' if uncertain else ''),(e['time_s'],e['f0_smoothed_hz']),xytext=(0,12 if e['label']=='H' else -18),textcoords='offset points',ha='center',color=color,fontsize=10)
    for c in result['characters']:
        mid=(c['start_s']+c['end_s'])/2;weak=c['annotation_status']=='manual_boundary_needed'
        ax.axvline(c['start_s'],color='#8a9ba6',alpha=.3,ls='--' if weak else '-',lw=.7)
        ax.text(mid,1.015,c['char']+('?' if weak else ''),transform=ax.get_xaxis_transform(),ha='center',fontsize=11)
        axs[1].text(mid,.75,c['tone5_label'],ha='center',fontsize=10,color='#9b612b' if weak else '#234b56')
        axs[1].text(mid,.24,c['hl_label'],ha='center',fontsize=10,color='#374f69')
    ax.set_yticks([60,80,100,120,150,180,220],[60,80,100,120,150,180,220]);ax.minorticks_off();ax.set_ylabel('F0 / Hz（对数轴）')
    ax.set_title(result['text']+'\n五度使用全轨固定归一化；H/L来自局部目标，不由五度数字直接换算',pad=35,fontsize=13)
    axs[1].set_ylim(0,1);axs[1].set_yticks([.75,.24],['五度(20/50/80%)','H/L候选']);axs[1].set_xlim(0,result['duration_s']);axs[1].set_xlabel('片段内时间 / 秒；? 为弱字/单侧目标待核，· 为未检出独立目标')
    for ax0 in axs:ax0.spines[['top','right']].set_visible(False)
    p=GEN/(result['utterance_id']+'.png');fig.savefig(p,dpi=180);plt.close(fig)


def page(results):
    sections=[]
    for i,r in enumerate(results):
        audio=base64.b64encode((OUT/r['clip_file']).read_bytes()).decode();pic=base64.b64encode((GEN/(r['utterance_id']+'.png')).read_bytes()).decode()
        buttons=''.join(f'<button data-start="{c["start_s"]:.6f}" data-end="{c["end_s"]:.6f}" onclick="playChar({i},this)"><b>{c["char"]}</b><small>{c["tone5_label"]}</small><small>{c["hl_label"]}</small></button>' for c in r['characters'])
        table=''.join(f'<tr><td>{c["index"]}</td><td>{c["char"]}</td><td>{c["start_s"]:.3f}–{c["end_s"]:.3f}</td><td>{c["tone5_label"]}</td><td>{c["hl_label"]}</td><td>{c["voiced_coverage"]:.0%}</td><td>{"待人工补界" if c["annotation_status"]=="manual_boundary_needed" else "候选"}</td></tr>' for c in r['characters'])
        sections.append(f'<section><h2>{r["utterance_id"].split("_")[-1]} · {html.escape(r["text"])}</h2><audio id="a{i}" controls src="data:audio/wav;base64,{audio}"></audio><select onchange="document.getElementById(\'a{i}\').playbackRate=+this.value"><option value="1">1倍速</option><option value="0.8">0.8倍速</option><option value="0.65">0.65倍速</option></select><div id="c{i}" class="chars">{buttons}</div><img src="data:image/png;base64,{pic}"><details><summary>逐字测量表</summary><table><tr><th>字序</th><th>汉字</th><th>候选时间/秒</th><th>声学五度</th><th>H/L候选</th><th>可靠F0覆盖</th><th>字界状态</th></tr>{table}</table></details></section>')
    header='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐字五度与H/L候选</title><style>body{font:16px system-ui,"Microsoft YaHei";background:#f3f7f9;color:#193541;max-width:1450px;margin:24px auto;padding:0 18px}h1{font-size:27px}h2{font-size:20px}p{line-height:1.7}section{background:white;margin:24px 0;padding:22px;border-radius:12px;overflow:auto}audio{width:75%}button{background:#edf6f3;border:1px solid #b2c9c5;border-radius:7px;margin:3px;padding:7px 10px;cursor:pointer}button b{font:23px system-ui}small{display:block;font:14px system-ui;margin-top:3px}.active{background:#addbd2}img{width:100%;min-width:1000px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #dde6e9;text-align:left}summary{cursor:pointer;padding:10px}</style><h1>逐字声学五度 · H/L目标候选</h1><p>沿用音素约束字界和全轨F0范围。五度三位数取该字最长可靠有声段20/50/80%位置；???表示F0不足，数字后?表示字界待补。H/L表示局部高/低目标候选，不要求每字都有目标，·为未检出；H?/L?为片段/有声段边缘或弱字待核。颜色背景是五度Hz分档，与H/L判定独立。</p><p>参考Roberts（2020）§4.1–4.3：突出目标通常为局部极值，弱目标可被相邻目标掩盖。这里尚未确定AP和目标的音系角色，因此不自动标H*、L*、La或域界。</p><a href="../character_pilot_v2/review.html">返回汉字/音素对照</a>'''
    script='''<script>const stops={};function playChar(i,b){const a=document.getElementById('a'+i);document.querySelectorAll('audio').forEach(x=>{if(x!==a)x.pause()});a.currentTime=Math.max(0,+b.dataset.start-.04);stops[i]=+b.dataset.end+.04;a.play()}function tick(){document.querySelectorAll('audio').forEach((a,i)=>{if(stops[i]!=null&&a.currentTime>=stops[i]){a.pause();delete stops[i]}document.querySelectorAll('#c'+i+' button').forEach(b=>b.classList.toggle('active',!a.paused&&a.currentTime>=+b.dataset.start&&a.currentTime<+b.dataset.end))});requestAnimationFrame(tick)}requestAnimationFrame(tick)</script></html>'''
    (OUT/'review.html').write_text(header+''.join(sections)+script,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);GEN.mkdir(parents=True,exist_ok=True)
    meta=json.loads(META.read_text(encoding='utf-8'));d=np.load(F0ROOT/'track1_f0.npz')
    filtered=parselmouth.read(str(F0ROOT/'track1_filtered_ac.Pitch'));ft=filtered.xs();ff=filtered.selected_array['frequency']
    samples=json.loads((SOURCE/'results.json').read_text(encoding='utf-8'));results=[]
    for sample in samples:
        origin=sample['clip_start_global_s'];m=(d['time_s']>=origin)&(d['time_s']<origin+sample['duration_s'])
        global_t=d['time_s'][m];t=global_t-origin;f=d['f0_raw_hz'][m];valid=d['reliable'][m].copy()
        ix=np.clip(np.rint((global_t-ft[0])/filtered.dx).astype(int),0,len(ft)-1)
        comparable=(ff[ix]>0)&(abs(ft[ix]-global_t)<=.0051)&(f>0)
        disagreement=np.zeros(len(t),bool);disagreement[comparable]=abs(12*np.log2(ff[ix][comparable]/f[comparable]))>1.
        valid &= ~disagreement
        sm=smooth_track(t,f,valid);events=targets(t,sm,valid)
        for e in events:
            c=next((c for c in sample['characters'] if c['start_s']<=e['time_s']<c['end_s']),None)
            e.update(time_global_s=origin+e['time_s'],char_index=c['index'] if c else None,char=c['char'] if c else None,
                     mapping_status=c['annotation_status'] if c else 'unassigned_gap',phonological_role='undetermined')
        chars=[measure_character(c,t,f,valid,meta['normalization'],events,origin) for c in sample['characters']]
        r={**sample,'characters':chars,'hl_events':events,'normalization':meta['normalization'],'parameters':PARAMS,
           'filtered_algorithm_disagreement_frames':int(disagreement.sum()),'HL_method':'local acoustic targets informed by Roberts; not full Sh_ToBI',
           'status':'tone5_and_HL_machine_candidates','AP_boundaries_inferred':False}
        np.savez_compressed(GEN/(r['utterance_id']+'_f0.npz'),time_clip_s=t,time_global_s=global_t,f0_hz=f,valid=valid,smoothed_st_re100=sm,algorithm_disagreement=disagreement)
        shutil.copy2(SOURCE/r['clip_file'],OUT/r['clip_file']);write_grid(r);plot(r,t,f,valid,sm)
        save(OUT/(r['utterance_id']+'.json'),r);results.append(r)
        print(r['utterance_id'],' '.join(c['char']+c['tone5_label']+'/'+c['hl_label'] for c in chars),flush=True)
    save(OUT/'results.json',results)
    summary=dict(samples=5,characters=72,tone5_candidates=sum(c['tone5_candidate'] is not None for r in results for c in r['characters']),
                 tone5_without_weak_boundary=sum(c['tone5_candidate'] is not None and c['annotation_status']!='manual_boundary_needed' for r in results for c in r['characters']),
                 insufficient_f0=sum(c['tone5_candidate'] is None for r in results for c in r['characters']),
                 hl_targets=sum(len(r['hl_events']) for r in results),
                 hl_uncertain=sum(e['edge_context'] or e['mapping_status']!='automatic_character_candidate' for r in results for e in r['hl_events']),
                 plateau_targets=sum(e['plateau'] for r in results for e in r['hl_events']),
                 original_alignment_sha256=hashlib.sha256((SOURCE/'results.json').read_bytes()).hexdigest(),
                 normalization_sha256=hashlib.sha256(META.read_bytes()).hexdigest(),parameters=PARAMS,
                 roberts_source='generated/roberts2020.txt; printed pp.60-75, especially 62-63 and 73-75',
                 numerical_thresholds='project engineering heuristics; not prescribed by Roberts',human_validation=False)
    save(OUT/'summary.json',summary);page(results);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
