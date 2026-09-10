"""Practical five-sentence preannotation: tolerate voicing, queue weak characters.

Reuses original emissions. Does not retrain, change greedy ASR, or label weak
forced intervals as acoustically established character boundaries.
"""
import base64
import collections
import html
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

import test_character_alignment as pilot
from plot_long_phone_comparison import edits

ROOT=pilot.ROOT
OUT=ROOT/'data/annotations/draft/aligner_v1/character_pilot_v2'
GEN=ROOT/'generated/aligner_v1/character_pilot_v2'
PAIRS=[('p','b'),('t','d'),('k','ɡ'),('s','z'),('ɕ','ʑ'),('tɕ','dʑ')]


def tolerance(logp,vocab):
    result=logp.copy()
    for a,b in PAIRS:
        if a in vocab and b in vocab:
            # A voiced/voiceless alternative contributes half its probability.
            # This is a alignment score, not a recalibrated ASR distribution.
            result[:,vocab[a]]=np.logaddexp(logp[:,vocab[a]],logp[:,vocab[b]]+math.log(.5))
            result[:,vocab[b]]=np.logaddexp(logp[:,vocab[b]],logp[:,vocab[a]]+math.log(.5))
    return result


def greedy(logp,centers,step,duration,vocab):
    reverse={v:k for k,v in vocab.items()};out=[];labels=logp.argmax(-1);i=0
    while i<len(labels):
        j=i+1
        while j<len(labels) and labels[j]==labels[i]:j+=1
        if labels[i]:
            peak=i+int(np.argmax(logp[i:j,labels[i]]))
            out.append(dict(phone=reverse[int(labels[i])],start_s=max(0.,float(centers[i]-step/2)),
                            end_s=min(duration,float(centers[j-1]+step/2)),anchor_s=float(centers[peak])))
        i=j
    return out


def grid(result):
    rows=result['characters'];duration=result['duration_s']
    tiers={'syllable':[(c['start_s'],c['end_s'],c['char']) for c in rows if c['annotation_status']!='manual_boundary_needed'],
           'word':[], 'AP':[], 'misc':[(c['start_s'],c['end_s'],'待人工补界:'+c['char']) for c in rows if c['annotation_status']=='manual_boundary_needed'],
           'char_candidate':[(c['start_s'],c['end_s'],c['char']) for c in rows],
           'phone_support':[(p['start_s'],p['end_s'],p['phone']) for c in rows for p in c['phone_support']],
           'free_phone':[(p['start_s'],p['end_s'],p['phone']) for p in result['greedy_timed_phones']]}
    tg=call('Create TextGrid...',0,duration,' '.join(tiers),'')
    for ti,items in enumerate(tiers.values(),1):
        for cut in sorted({v for a,b,_ in items for v in (a,b) if 0<v<duration}):call(tg,'Insert boundary',ti,cut)
        for a,b,text in items:call(tg,'Set interval text',ti,call(tg,'Get interval at time',ti,(a+b)/2),text)
    path=OUT/(result['utterance_id']+'.TextGrid')
    call(tg,'Save as text file',str(path));parselmouth.read(str(path))


def plot(result,wave,sr):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(17,4.3),layout='constrained')
    ax.set_xlim(0,result['duration_s']);ax.set_ylim(-.1,3.45)
    t=np.arange(len(wave))/sr;scale=max(float(abs(wave).max()),.01)
    ax.plot(t[::8],2.8+wave[::8]/scale*.32,lw=.55,color='#677e88')
    for c in result['characters']:
        weak=c['annotation_status']=='manual_boundary_needed';mid=(c['start_s']+c['end_s'])/2
        color='#9b5d22' if weak else '#087e73'
        ax.plot([c['start_s'],c['end_s']],[3.10,3.10],color=color,lw=2,ls='--' if weak else '-')
        ax.text(mid,3.2,c['char']+('?' if weak else ''),ha='center',fontsize=12,color=color)
        ax.axvline(c['start_s'],ymin=.71,ymax=.9,color=color,alpha=.35,ls='--' if weak else '-')
    reference=[p for c in result['characters'] for p in c['phone_support']]
    actual=result['greedy_timed_phones']
    ops,_=edits([p['phone'] for p in reference],[p['phone'] for p in actual])
    equivalence={frozenset(p) for p in PAIRS}
    colors={'match':'#087e73','substitution':'#b14d34','insertion':'#c08012','deletion':'#80528d','voicing':'#2b75ad'}
    for op,i,j in ops:
        if op=='substitution' and frozenset([reference[i]['phone'],actual[j]['phone']]) in equivalence:op='voicing'
        if i is not None:reference[i]['comparison']=op
        if j is not None:actual[j]['comparison']=op
    for phones,y in [(reference,1.6),(actual,.55)]:
        for k,p in enumerate(phones):
            mid=p.get('anchor_s',(p['start_s']+p['end_s'])/2);level=y+(k%3)*.2;color=colors[p['comparison']]
            ax.plot([p['start_s'],p['end_s']],[y-.1,y-.1],color=color,lw=2)
            ax.annotate(p['phone'],xy=(mid,y-.12),xytext=(mid,level),ha='center',fontsize=10,fontfamily='DejaVu Sans',color=color,arrowprops=dict(arrowstyle='-',lw=.5,color=color))
    ax.set_yticks([.8,1.9,2.8],['原始自由识别','文本读音候选','波形 / 汉字'])
    ax.set_xlabel('片段内秒；? 表示缺少可靠声学支撑，留待人工补界')
    ax.set_title(result['text']+'\n绿：一致　蓝：清浊差异（容忍）　红/紫/橙：其他序列差异',fontsize=13)
    ax.grid(axis='x',alpha=.13)
    fig.savefig(GEN/(result['utterance_id']+'.png'),dpi=180);plt.close(fig)


def page(results):
    sections=[]
    for i,r in enumerate(results):
        audio=base64.b64encode((OUT/r['clip_file']).read_bytes()).decode()
        picture=base64.b64encode((GEN/(r['utterance_id']+'.png')).read_bytes()).decode()
        buttons=''.join(f'<button data-start="{c["start_s"]:.6f}" data-end="{c["end_s"]:.6f}" onclick="playChar({i},this)" class="{"weak" if c["annotation_status"]=="manual_boundary_needed" else ""}" title="{c["annotation_status"]}">{c["char"]}{"?" if c["annotation_status"]=="manual_boundary_needed" else ""}</button>' for c in r['characters'])
        manual='、'.join(f'{c["index"]}:{c["char"]}' for c in r['characters'] if c['annotation_status']=='manual_boundary_needed') or '无明显弱支撑字；仍是机器预标注'
        sections.append(f'<section><h2>{r["utterance_id"].split("_")[-1]} · {html.escape(r["text"])}</h2><audio id="a{i}" controls src="data:audio/wav;base64,{audio}"></audio><select onchange="document.getElementById(\'a{i}\').playbackRate=+this.value"><option value="1">1倍速</option><option value="0.8">0.8倍速</option><option value="0.65">0.65倍速</option></select><div id="c{i}">{buttons}</div><p>待人工补界（字序:汉字）：{manual}。点击?字播放的是估计范围，不能作为已确定字界。</p><img src="data:image/png;base64,{picture}"></section>')
    header='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>5句汉字预标注：弱读留人工</title><style>body{font:16px system-ui,"Microsoft YaHei";background:#f4f7fa;color:#173342;max-width:1450px;margin:24px auto;padding:0 18px}h1{font-size:26px}h2{font-size:21px}p{line-height:1.6}section{background:white;margin:24px 0;padding:22px;border-radius:12px;overflow-x:auto}audio{width:75%}button{font:23px system-ui;background:#e6f4ef;border:1px solid #abcac0;border-radius:6px;margin:4px;padding:6px 12px;cursor:pointer}.weak{background:#fff0dc;border-style:dashed}.active{background:#087e73;color:white}img{width:100%;min-width:900px}</style><h1>5句汉字预标注 · 弱读留人工</h1><p>先检查字序与主要位置。蓝色清浊差异可容忍；带?的字仅保留候选范围，TextGrid的自动syllable层留空、转入人工补界队列。其他字也是机器预标注，并非已听校。</p><a href="../long_phone_pilot_v1/review.html">查看32秒原始自由识别对照</a><p>点击单字试听（前后40毫秒），也可整句播放或放慢。</p>'''
    js='''<script>const stops={};function playChar(i,b){const a=document.getElementById('a'+i);document.querySelectorAll('audio').forEach(x=>{if(x!==a)x.pause()});a.currentTime=Math.max(0,+b.dataset.start-.04);stops[i]=+b.dataset.end+.04;a.play()}function tick(){document.querySelectorAll('audio').forEach((a,i)=>{if(stops[i]!=null&&a.currentTime>=stops[i]){a.pause();delete stops[i]}document.querySelectorAll('#c'+i+' button').forEach(b=>b.classList.toggle('active',!a.paused&&a.currentTime>=+b.dataset.start&&a.currentTime<+b.dataset.end))});requestAnimationFrame(tick)}requestAnimationFrame(tick)</script></html>'''
    (OUT/'review.html').write_text(header+''.join(sections)+js,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);GEN.mkdir(parents=True,exist_ok=True)
    vocab=json.loads((pilot.MODEL/'phone_vocab.json').read_text(encoding='utf-8'))
    plans=json.loads((pilot.OUT/'reading_candidates.json').read_text(encoding='utf-8'))
    prior=json.loads((pilot.OUT/'results.json').read_text(encoding='utf-8'))
    results=[];queue=[]
    for plan,old in zip(plans,prior):
        cache=np.load(pilot.GEN/(old['utterance_id']+'_emissions.npz'))
        logp=cache['logp'];sr=int(cache['sr']);step=float(cache['stride'])/sr
        centers=(np.arange(len(logp))*int(cache['stride'])+(int(cache['receptive_field'])-1)/2)/sr
        scores=tolerance(logp,vocab)
        nodes,path,value=pilot.lattice_align(scores,plan['characters'],vocab)
        rows=pilot.evidence(nodes,path,plan['characters'],scores,centers,step,old['duration_s'])
        wave,_=sf.read(pilot.OLD/old['clip_file'],dtype='float32')
        pilot.boundaries(rows,wave,sr,old['duration_s'])
        for c in rows:
            weak=c['mean_emission_posterior']<.15 or max(p['peak_posterior'] for p in c['phone_support'])<.3
            c['annotation_status']='manual_boundary_needed' if weak else 'automatic_character_candidate'
            c['alignment_support_score']=c.pop('mean_emission_posterior')
            c['support_score_note']='voicing-tolerant weighted score; not a calibrated correctness probability'
            c['review_notes']='声学支撑弱；时间仅供定位，人工补界' if weak else '字序/位置机器候选；不因单个phone或清浊不同自动退回'
            c['start_global_s']=old['clip_start_global_s']+c['start_s'];c['end_global_s']=old['clip_start_global_s']+c['end_s']
            if weak:queue.append(dict(utterance_id=old['utterance_id'],index=c['index'],char=c['char'],candidate_start_s=c['start_s'],candidate_end_s=c['end_s'],status='manual_boundary_needed'))
        result={**old,'characters':rows,'greedy_timed_phones':greedy(logp,centers,step,old['duration_s'],vocab),
                'voicing_pairs':PAIRS,'voicing_alternative_weight':.5,'ctc_path_score':value,
                'status':'practical_preannotation_weak_chars_deferred','human_listening_performed':False}
        assert all(0<=c['start_s']<c['end_s']<=old['duration_s'] for c in rows)
        assert all(a['end_s']<=b['start_s']+1e-7 for a,b in zip(rows,rows[1:]))
        shutil.copy2(pilot.OLD/old['clip_file'],OUT/old['clip_file'])
        grid(result);plot(result,wave,sr)
        pilot.save(OUT/(old['utterance_id']+'.json'),result);results.append(result)
    pilot.save(OUT/'results.json',results);pilot.save(OUT/'manual_review_queue.json',queue)
    summary=dict(samples=5,characters=72,manual_boundary_needed=len(queue),automatic_character_candidates=72-len(queue),
                 human_validated_characters=0,training_performed=False,raw_greedy_output_changed=False,voicing_pairs=PAIRS,
                 input_emissions='reused character_pilot_v1 original model outputs')
    pilot.save(OUT/'summary.json',summary);page(results)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
