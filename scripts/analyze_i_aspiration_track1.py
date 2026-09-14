"""Full indexed track-1 onset-F0 audit. No correction or annotation writes.

Run with the qwen3-asr Python environment (numpy, parselmouth, matplotlib).
All cutoffs are transparent screening conventions, not perceptual thresholds.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
import sys

import numpy as np
import parselmouth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wugniu import Wugniu

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/annotations/draft/aligner_v1/tone_hl_track1_v3'
F0ROOT = ROOT / 'generated/recording_11_20260909/tone_pilot'
OUT = ROOT / 'generated/i_aspiration_track1_20260914'
REVIEW = ROOT / 'data/annotations/draft/i_aspiration_track1_20260914'
I_NUC = {'i', 'in', 'iq'}
I_MED = {'ia', 'iau', 'ieu', 'ie', 'ian', 'iaon', 'ion', 'iaq', 'ioq'}
ASP = {'ph', 'th', 'tsh', 'ch', 'kh'}
PLAIN = {'p', 't', 'ts', 'c', 'k'}
GROUPS = ['no_i_plain', 'i_plain', 'no_i_asp', 'i_asp']
LABELS = ['无/i/·非送气', '含/i/·非送气', '无/i/·送气', '含/i/·送气']
CONFIG = dict(version='20260914-v1', primary_early_s=[0, .03], primary_body_s=[.06, .12],
              primary_min_rime_s=.12, min_early_frames=2, min_body_frames=3,
              adaptive_span='T=min(candidate_rime_duration,120ms); head=[0,.3T); body=[.6T,.9T)',
              adaptive_min_frames_per_window=2, fixed_window_sensitivity_offsets_s=[-.02, .02],
              screening_drop_st=[1, 2, 3], agreement_st=1, strict_max_gap_s=.020001,
              strict_channel_dominance_db=6, bootstrap_unit='utterance', bootstrap_reps=2000,
              seed=20260914, no_interpolation=True, no_pitch_correction=True)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(p, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def finite(v):
    return v is not None and math.isfinite(v)


def onset_class(initial):
    if initial in ASP: return 'aspirated_obstruent'
    if initial in PLAIN: return 'voiceless_unaspirated_obstruent'
    if initial in {'b', 'd', 'g', 'j', 'z', 'v', 'zh', 'gh'}: return 'voiced_obstruent_or_h'
    if initial in {'f', 's', 'sh', 'h'}: return 'voiceless_fricative_or_h'
    if not initial: return 'zero'
    return 'sonorant_or_glottal_sonorant'


def candidate_window(c, parsed):
    onset = int(bool(parsed['initial']))
    rime = c['phone_support'][onset:]
    voiced = [p for p in rime if p['phone'] != 'ʔ']
    if not voiced: return None, None, None
    a = max(c['start_s'], voiced[0]['start_s']) if onset else c['start_s']
    b = min([c['end_s']] + [p['start_s'] for p in rime if p['phone'] == 'ʔ'])
    # Sensitivity only: these are emission supports, not measured coda boundaries.
    vb = min([b] + [p['start_s'] for p in rime[1:] if p['phone'] in {'n', 'ŋ', 'm', 'l', 'ʔ'}])
    return (a, b, max(a, vb)) if b > a else (None, None, None)


def measure(t, f, a, b, mode='fixed'):
    if a is None or b is None: return dict(d_st=None, reason='no_window')
    if mode == 'fixed':
        if b-a < .12-1e-8: return dict(d_st=None, reason='rime_shorter_than_120ms')
        h0,h1,l0,l1 = a,a+.03,a+.06,a+.12
        nh,nl = 2,3
    else:
        span = min(b-a, .12)
        h0,h1,l0,l1 = a,a+.3*span,a+.6*span,a+.9*span
        nh,nl = 2,2
    head=(t>=h0-1e-9)&(t<h1-1e-9)&np.isfinite(f)&(f>0)
    body=(t>=l0-1e-9)&(t<l1-1e-9)&np.isfinite(f)&(f>0)
    result = dict(d_st=None, reason='insufficient_voicing', head_n=int(head.sum()), body_n=int(body.sum()))
    if head.sum()<nh or body.sum()<nl: return result
    hs=12*np.log2(f[head]/100);bs=12*np.log2(f[body]/100)
    ht=float(np.median(t[head]));bt=float(np.median(t[body]));d=float(np.median(hs)-np.median(bs))
    between=(t>=t[head][0]-1e-9)&(t<=t[body][-1]+1e-9)&np.isfinite(f)&(f>0)
    result.update(d_st=d, reason='measured', head_hz=float(100*2**(np.median(hs)/12)),
                  body_hz=float(100*2**(np.median(bs)/12)), elapsed_ms=1000*(bt-ht),
                  descent_st_s=d/(bt-ht), first_delay_ms=1000*(t[head][0]-a),
                  gap_ms=float(np.max(np.diff(t[between]))*1000), head_time_s=ht, body_time_s=bt,
                  head_first_global_s=float(t[head][0]),head_last_global_s=float(t[head][-1]),
                  body_first_global_s=float(t[body][0]),body_last_global_s=float(t[body][-1]))
    return result


def add_measure(row, prefix, result):
    for k,v in result.items(): row[prefix+'_'+k]=v


def main():
    OUT.mkdir(parents=True, exist_ok=True); REVIEW.mkdir(parents=True, exist_ok=True)
    paths=sorted(SOURCE.glob('recording_11_20260909_[0-9][0-9][0-9][0-9].json'))
    protected=paths+[F0ROOT/'track1_f0.npz', F0ROOT/'track1_filtered_ac.Pitch',
                     ROOT/'generated/recording_11_20260909/track_1_16k.wav',
                     ROOT/'data/metadata/recording_11_segments.csv',
                     ROOT/'data/metadata/recording_11_track1_tone_normalization.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    save(OUT/'protocol.json',CONFIG)
    data=np.load(F0ROOT/'track1_f0.npz');t=data['time_s'];f=data['f0_raw_hz'].copy()
    f[f<=0]=np.nan
    pitch=parselmouth.read(str(F0ROOT/'track1_filtered_ac.Pitch'))
    ft=pitch.xs();ff=pitch.selected_array['frequency'].copy();ff[ff<=0]=np.nan
    ix=np.rint((t-ft[0])/pitch.dx).astype(int)
    assert (ix>=0).all() and (ix<len(ft)).all() and np.max(abs(ft[ix]-t))<.0051
    filtered=ff[ix]; reliable=data['reliable'].astype(bool)
    dominance=data['rms_dbfs']-data['track2_rms_dbfs']
    with np.errstate(invalid='ignore',divide='ignore'): diff=12*np.log2(filtered/f)
    agreement=reliable & np.isfinite(filtered) & (abs(diff)<=1)
    good=np.where(reliable,f,np.nan); agreed=np.where(agreement,f,np.nan)
    converter=Wugniu();rows=[];traces={};coverage=[]
    for path in paths:
        r=json.loads(path.read_text(encoding='utf-8'));origin=r['clip_start_global_s']
        coverage.append((origin,origin+r['duration_s']))
        for c in r['characters']:
            p=converter.parse_syllable(c['spelling']);ik='nucleus' if p['final'] in I_NUC else 'medial' if p['final'] in I_MED else 'none'
            asp=p['initial'] in ASP;group=('i' if ik!='none' else 'no_i')+('_asp' if asp else '_plain')
            a,b,vb=candidate_window(c,p)
            row=dict(uid=r['utterance_id'], sentence=r['source_sentence_id'], index=c['index'], char=c['char'],
                     spelling=c['spelling'], initial=p['initial'], final=p['final'], ipa=p['ipa'], i_kind=ik,
                     aspirated=asp, group=group, onset_class=onset_class(p['initial']),
                     text=r['text'], source_word=c['reading_source'].get('source_word',''),
                     reading_note=c['reading_source'].get('note',''), tone_suffix=p['tone_suffix_raw'],
                     char_start_s=c['start_s'], char_end_s=c['end_s'], clip_start_global_s=origin,
                     char_start_global_s=origin+c['start_s'], char_duration_ms=1000*(c['end_s']-c['start_s']),
                     rime_start_s=a,rime_end_s=b,rime_duration_ms=1000*(b-a) if a is not None else None,
                     vowel_support_end_s=vb, rime_start_global_s=origin+a if a is not None else None,
                     issues=c['issues'], annotation_status=c['annotation_status'], alignment_score=c['alignment_support_score'],
                     alignment_clean=not c['issues'] and c['annotation_status']!='manual_boundary_needed',
                     original_window_mode=c['f0_window'].get('mode'),
                     sentence_position='initial' if c['index']==1 else 'final' if c['index']==len(r['characters']) else 'internal',
                     checked=p['final'].endswith('q'), nasal_coda=any(q in {'n','ŋ','m'} for q in p['phones'][1:]))
            if a is None:
                for name in ['raw','filtered','reliable','agreed','adaptive_raw','adaptive_agreed','vowel','shift_minus20','shift_plus20']:
                    add_measure(row,name,dict(d_st=None,reason='no_window'))
                rows.append(row);continue
            ga,gb=origin+a,origin+b
            lo=np.searchsorted(t,ga-.03);hi=np.searchsorted(t,gb+.03)
            tt=t[lo:hi]; vals={'raw':f[lo:hi],'filtered':filtered[lo:hi],'reliable':good[lo:hi],'agreed':agreed[lo:hi]}
            for name,value in vals.items(): add_measure(row,name,measure(tt,value,ga,gb))
            for name,value in vals.items():
                vv=(tt>=ga)&(tt<gb)&np.isfinite(value)&(value>0)
                row[name+'_first_voiced_delay_ms']=float((tt[vv][0]-ga)*1000) if vv.any() else None
                row[name+'_early_available_n']=int((vv&(tt<ga+.03)).sum())
                row[name+'_body_available_n']=int((vv&(tt>=ga+.06)&(tt<ga+.12)).sum())
            for name in ['raw','agreed']:add_measure(row,'adaptive_'+name,measure(tt,vals[name],ga,gb,'adaptive'))
            add_measure(row,'vowel',measure(tt,vals['agreed'],ga,origin+vb))
            for off,name in [(-.02,'shift_minus20'),(.02,'shift_plus20')]:
                shifted=max(origin+c['start_s'],ga+off)
                row[name+'_actual_ms']=1000*(shifted-ga)
                add_measure(row,name,measure(tt,vals['agreed'],shifted,gb))
            win=(tt>=ga)&(tt<min(gb,ga+.12)); local_raw=vals['raw'][win]; local_t=tt[win]
            voiced=np.isfinite(local_raw);vi=np.flatnonzero(voiced)
            steps=[]
            for l,h in zip(vi[:-1],vi[1:]):
                if local_t[h]-local_t[l]<=.010001:steps.append(float(12*np.log2(local_raw[l]/local_raw[h])))
            row.update(raw_frames=int(voiced.sum()), agreement_frames=int(np.isfinite(vals['agreed'][win]).sum()),
                       reliable_frames=int(np.isfinite(vals['reliable'][win]).sum()),
                       maximum_adjacent_drop_st=max(steps) if steps else None,
                       adjacent_jump_ge7=any(abs(v)>=7 for v in steps),
                       raw_filtered_disagreement_frames=int((np.isfinite(diff[lo:hi][win])&(abs(diff[lo:hi][win])>1)).sum()),
                       channel_dominance_median_db=float(np.median(dominance[lo:hi][win])) if win.any() else None)
            # High-support sensitivity retains measured drops; it does not trim onset frames.
            row['strict']=bool(finite(row['agreed_d_st']) and row['alignment_clean'] and
                               row.get('agreed_gap_ms',999)<=20.001 and not row['adjacent_jump_ge7'] and
                               row['channel_dominance_median_db']>=6)
            rows.append(row)
            offsets=np.arange(0,.12,.01)+.005
            inds=np.rint((ga+offsets-t[0])/.01).astype(int)
            ok=(inds>=0)&(inds<len(t))&(t[np.clip(inds,0,len(t)-1)]<gb)
            trace=np.full(len(offsets),np.nan)
            if finite(row['agreed_d_st']):trace[ok]=12*np.log2(agreed[inds[ok]]/row['agreed_body_hz'])
            traces[(row['uid'],row['index'])]=trace
    table(OUT/'tokens.csv',rows);save(OUT/'tokens.json',rows)
    cohorts=[]
    variants=[('raw','raw_d_st'),('filtered','filtered_d_st'),('reliable','reliable_d_st'),('agreement','agreed_d_st'),
              ('strict','agreed_d_st'),('adaptive_raw','adaptive_raw_d_st'),('adaptive_agreement','adaptive_agreed_d_st'),
              ('vowel_support','vowel_d_st'),('shift_minus20','shift_minus20_d_st'),('shift_plus20','shift_plus20_d_st')]
    for variant,key in variants:
        for group in GROUPS:
            population=[r for r in rows if r['group']==group]
            eligible=[r for r in population if finite(r.get(key)) and (variant!='strict' or r.get('strict',False))]
            arr=np.array([r[key] for r in eligible]);counts=Counter(r['char'] for r in eligible)
            q=dict(variant=variant,group=group,total_tokens=len(population),measured_tokens=len(eligible),
                   measured_fraction=len(eligible)/len(population), utterances=len(set(r['uid'] for r in eligible)),
                   char_types=len(counts), spelling_types=len(set(r['spelling'] for r in eligible)),
                   median_d_st=float(np.median(arr)) if len(arr) else None, mean_d_st=float(np.mean(arr)) if len(arr) else None,
                   p90_d_st=float(np.quantile(arr,.9)) if len(arr) else None, top5_chars=counts.most_common(5))
            for cutoff in [1,2,3]:
                q['drop_ge'+str(cutoff)+'_n']=int((arr>=cutoff).sum())
                q['drop_ge'+str(cutoff)+'_fraction']=float((arr>=cutoff).mean()) if len(arr) else None
            cohorts.append(q)
    table(OUT/'cohorts.csv',cohorts)
    # Sentence cluster bootstrap: sampling sentences (including empty cohorts), not frames.
    uids=sorted(set(r['uid'] for r in rows)); umap={u:i for i,u in enumerate(uids)}
    rng=np.random.default_rng(CONFIG['seed']); weights=rng.multinomial(len(uids),np.ones(len(uids))/len(uids),size=CONFIG['bootstrap_reps'])
    inference=[]
    for variant,key in [('agreement','agreed_d_st'),('strict','agreed_d_st')]:
        means={};rates={}
        for group in GROUPS:
            eligible=[r for r in rows if r['group']==group and finite(r.get(key)) and (variant!='strict' or r.get('strict',False))]
            counts=np.zeros(len(uids));sums=counts.copy();hits=counts.copy()
            for r in eligible:
                j=umap[r['uid']];counts[j]+=1;sums[j]+=r[key];hits[j]+=r[key]>=2
            denom=weights@counts;means[group]=(sums.sum()/counts.sum(),(weights@sums)/denom)
            rates[group]=(hits.sum()/counts.sum(),(weights@hits)/denom)
        for a,b,label in [('i_plain','no_i_plain','i_effect_nonasp'),('i_asp','no_i_asp','i_effect_asp'),
                          ('no_i_asp','no_i_plain','asp_effect_no_i'),('i_asp','i_plain','asp_effect_i')]:
            for kind,d in [('mean_drop_st',means),('drop_ge2_rate_difference',rates)]:
                boot=d[a][1]-d[b][1]
                inference.append(dict(variant=variant,contrast=label,metric=kind,estimate=d[a][0]-d[b][0],ci95=np.quantile(boot,[.025,.975]).tolist()))
        for kind,d in [('mean_drop_st',means),('drop_ge2_rate_difference',rates)]:
            boot=d['i_asp'][1]-d['no_i_asp'][1]-d['i_plain'][1]+d['no_i_plain'][1]
            inference.append(dict(variant=variant,contrast='interaction',metric=kind,
                                  estimate=d['i_asp'][0]-d['no_i_asp'][0]-d['i_plain'][0]+d['no_i_plain'][0],ci95=np.quantile(boot,[.025,.975]).tolist()))
    table(OUT/'cluster_contrasts.csv',inference)
    breakdown=[]
    for field in ['i_kind','initial','final','onset_class','sentence_position','char']:
        buckets=defaultdict(list)
        for r in rows:
            if finite(r.get('agreed_d_st')): buckets[(r['group'],str(r[field]))].append(r)
        for (group,value),rs in buckets.items():
            ds=np.array([r['agreed_d_st'] for r in rs])
            breakdown.append(dict(field=field,group=group,value=value,n=len(rs),utterances=len(set(r['uid'] for r in rs)),
                                  median_d_st=float(np.median(ds)),mean_d_st=float(np.mean(ds)),drop_ge2=int((ds>=2).sum()),fraction_ge2=float((ds>=2).mean())))
    table(OUT/'breakdowns.csv',breakdown)
    merged=[]
    for a,b in sorted(coverage):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    summary=dict(config=CONFIG,total_sentences=len(paths),total_tokens=len(rows),full_track_duration_s=float(pitch.xmax),
                 indexed_clips_union_s=sum(b-a for a,b in merged),raw_voiced_frames=int(np.isfinite(f).sum()),
                 rime_window_missing=sum(r['rime_start_s'] is None for r in rows),
                 candidate_classification=Counter(r['group'] for r in rows),
                 i_kind_counts=Counter(r['i_kind'] for r in rows),tone_suffix_present=sum(bool(r['tone_suffix']) for r in rows),
                 user_examples=[r for r in rows if (r['sentence'],r['index']) in {(34,5),(34,16),(36,6)}],
                 cohorts=cohorts,cluster_contrasts=inference)
    save(OUT/'summary.json',summary)
    plot(rows,cohorts,traces)
    review(rows)
    after={p:sha(ROOT/p) for p in before}
    assert before==after
    assert len(rows)==9362 and len(paths)==593
    save(OUT/'verification.json',dict(originals_unchanged=True,input_hashes=before,
                                      script_sha256=sha(Path(__file__)),n_rows=len(rows),unique_tokens=len({(r['uid'],r['index']) for r in rows}),
                                      matched_grid_max_offset_ms=float(np.max(abs(ft[ix]-t))*1000),
                                      note='Input immutability and numeric checks do not validate pronunciation, timing or track attribution.'))
    print(json.dumps({k:summary[k] for k in ['total_sentences','total_tokens','indexed_clips_union_s','candidate_classification','i_kind_counts']},ensure_ascii=False))
    for q in cohorts:
        if q['variant'] in {'agreement','strict'}:print(json.dumps(q,ensure_ascii=False))


def plot(rows,cohorts,traces):
    font=Path('C:/Windows/Fonts/msyh.ttc');font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'axes.unicode_minus':False,'font.size':10})
    colors=['#778899','#237f92','#bf8740','#bd4d4b']
    fig,axs=plt.subplots(1,3,figsize=(16,5.4),layout='constrained')
    for i,g in enumerate(GROUPS):
        ds=np.array([r['agreed_d_st'] for r in rows if r['group']==g and finite(r.get('agreed_d_st'))])
        axs[0].plot(np.sort(ds),np.arange(1,len(ds)+1)/len(ds),color=colors[i],label=f'{LABELS[i]} (n={len(ds)})')
        q=next(q for q in cohorts if q['variant']=='agreement' and q['group']==g)
        s=next(q for q in cohorts if q['variant']=='strict' and q['group']==g)
        axs[1].bar(i-.17,q['drop_ge2_fraction']*100,.32,color=colors[i]);axs[1].bar(i+.17,s['drop_ge2_fraction']*100,.32,color=colors[i],alpha=.4)
        axs[1].text(i,q['drop_ge2_fraction']*100+1,f'{q["drop_ge2_n"]}/{q["measured_tokens"]}',ha='center',fontsize=9)
        ts=np.array([traces[(r['uid'],r['index'])] for r in rows if r['group']==g and finite(r.get('agreed_d_st'))])
        axs[2].plot(np.arange(12)*10+5,np.nanmedian(ts,axis=0),color=colors[i],label=LABELS[i])
    axs[0].axvline(2,color='black',ls=':',alpha=.5);axs[0].set_xlim(-6,8);axs[0].set_xlabel('起始高出后段 D（半音，正值＝下降）');axs[0].set_ylabel('累计比例');axs[0].legend(fontsize=8);axs[0].set_title('D分布：图限−6至8ST，统计保留全值')
    axs[1].set_xticks(range(4),[l.replace('·','\n') for l in LABELS]);axs[1].set_ylabel('D ≥ 2 ST 的比例（%）');axs[1].set_title('深色：双算法支持；浅色：严格子集')
    axs[2].axhline(0,color='black',lw=.5);axs[2].set_xlabel('距候选韵母起点（毫秒）');axs[2].set_ylabel('相对各字60–120ms后段（半音）');axs[2].set_title('分组逐时点中位数（点间可用样本变化）')
    for ax in axs:ax.grid(alpha=.15)
    fig.suptitle('音轨1全量候选：/i/ 与送气声母的初始F0下降\n固定比较0–30ms与60–120ms；排除不足120ms窗口；非因果、非感知标签',fontsize=14)
    fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)


def review(rows):
    # Include every indexed token, including missing values and negative controls.
    fields=['uid','sentence','index','char','spelling','ipa','i_kind','aspirated','group','char_start_s','char_end_s','text',
            'agreed_d_st','raw_d_st','filtered_d_st','adaptive_agreed_d_st','strict','issues','agreed_reason','rime_duration_ms']
    payload=[{k:r.get(k) for k in fields} for r in rows]
    rel=lambda p: os.path.relpath(p,REVIEW).replace(os.sep,'/')
    src=rel(SOURCE);fig=rel(OUT/'overview.png')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>音轨1：i与送气起始F0</title>
<style>body{font:16px/1.6 system-ui;margin:28px auto;max-width:1280px;color:#253644}img{width:100%}input,select,button{font:inherit;padding:5px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;text-align:left;padding:7px}th{background:#eef3f5}small{color:#63717b}.controls{display:flex;gap:12px;flex-wrap:wrap;position:sticky;top:0;background:white;padding:10px 0}audio{width:100%}</style>
<h1>音轨1：/i/ 与送气声母的起始 F0</h1><p>593句、9362字候选。D＝候选韵母起点0–30ms音高 − 60–120ms音高（半音），D正值表示下降。主结果要求韵母至少120ms，raw可靠帧且filtered差≤1ST。2ST是本次筛查规则；缺测不表示没有下降。“体”短窗另看自适应值。所有读音/字界均未人工验收；“非送气”包含浊音、擦音等。</p>
<img src="FIG" alt="全量统计"><audio id="audio" controls></audio><p id="now">选择逐字片段试听（前后各留120ms），或打开原句完整图。</p>
<div class="controls"><input id="query" placeholder="搜索句号、汉字、拼音"><select id="group"><option value="all">全部四组</option><option value="i_asp">含i＋送气</option><option value="i_plain">含i＋非送气</option><option value="no_i_asp">无i＋送气</option><option value="no_i_plain">无i＋非送气</option></select><select id="filter"><option value="examples">用户三例</option><option value="drop">主指标D≥2ST</option><option value="all">全部（含缺测）</option><option value="strict">严格子集D≥2ST</option></select><select id="sort"><option value="time">按时间</option><option value="drop">按下降幅度</option></select><span id="count"></span></div>
<table><thead><tr><th>句／字</th><th>读音</th><th>类型</th><th>D/ST</th><th>raw／filtered</th><th>自适应D</th><th>质量／试听</th></tr></thead><tbody id="body"></tbody></table><p>最多显示前500条；全量明细见generated目录的tokens.csv。原始音频与标注保留。</p>
<script>const rows=PAYLOAD; const source=SOURCE; const audio=document.getElementById('audio');let stopAt=null;let requestId=0;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const num=x=>x==null?'缺测':x.toFixed(2);
audio.addEventListener('timeupdate',()=>{if(stopAt!==null&&audio.currentTime>=stopAt){audio.pause();stopAt=null;}});
async function play(n){const r=rows[n],id=++requestId;audio.pause();stopAt=null;audio.src=source+'/'+r.uid+'_track1.wav';await new Promise((resolve,reject)=>{audio.onloadedmetadata=resolve;audio.onerror=reject;audio.load();});if(id!==requestId)return;audio.currentTime=Math.max(0,r.char_start_s-.12);stopAt=Math.min(audio.duration,r.char_end_s+.12);await audio.play();document.getElementById('now').textContent=r.sentence+' 第'+r.index+'字 '+r.char+'：'+r.text;}
function render(){const q=document.getElementById('query').value.trim(),g=document.getElementById('group').value,f=document.getElementById('filter').value;let rs=rows.map((r,n)=>({...r,n})).filter(r=>(g==='all'||r.group===g)&&(!q||(r.uid+' '+r.char+' '+r.spelling+' '+r.text).includes(q))&&(f==='all'||f==='examples'&&((r.sentence===34&&[5,16].includes(r.index))||(r.sentence===36&&r.index===6))||f==='drop'&&r.agreed_d_st!==null&&r.agreed_d_st>=2||f==='strict'&&r.strict&&r.agreed_d_st>=2));if(document.getElementById('sort').value==='drop')rs.sort((a,b)=>(b.agreed_d_st??-999)-(a.agreed_d_st??-999));document.getElementById('count').textContent=rs.length+'条';document.getElementById('body').innerHTML=rs.slice(0,500).map(r=>`<tr><td><a href="${source}/${r.uid}.html">${r.sentence}／${r.index} ${esc(r.char)}</a></td><td>${esc(r.spelling)} /${esc(r.ipa)}/</td><td>${esc(r.i_kind)}${r.aspirated?'＋送气':''}</td><td>${num(r.agreed_d_st)}</td><td>${num(r.raw_d_st)}／${num(r.filtered_d_st)}</td><td>${num(r.adaptive_agreed_d_st)}</td><td><button onclick="play(${r.n})">试听</button> ${r.strict?'严格子集':esc(r.issues.join('; ')||r.agreed_reason||'待核')}</td></tr>`).join('');}
for(const id of ['query','group','filter','sort'])document.getElementById(id).addEventListener('input',render);render();</script></html>'''
    page=page.replace('FIG',html.escape(fig,quote=True)).replace('PAYLOAD',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')).replace('SOURCE',json.dumps(src))
    details='<details><summary>展开0034“体、系”与0036“请”的波形／逐周期核查</summary><p>这些是自动脉冲及波形检查，尚未人工逐周期验收。系的完整频率图保留摩擦段和衰减尾端高频误追；请的后段可能含/n/。重点例使用的共同可观察区间不同于全量固定窗D。</p>'
    for name in ['0034_05_diagnostic.png','0034_16_diagnostic.png','0034_16_f0_full_range.png','0036_06_diagnostic.png']:
        details+='<img src="'+html.escape(rel(OUT/'examples'/name),quote=True)+'" alt="'+name+'">'
    page=page.replace('<audio id="audio"',details+'</details><audio id="audio"')
    (REVIEW/'review.html').write_text(page,encoding='utf-8')


if __name__=='__main__': main()
