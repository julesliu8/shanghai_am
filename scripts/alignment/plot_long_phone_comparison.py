"""Single-pass 32.148-second phone decoding with a fixed text-IPA comparison.

Greedy predictions never receive text. Fixed dictionary hypotheses are separately
forced onto the same emissions for display; their timings and phones are not gold.
"""
import base64
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC
import test_character_alignment as pilot

ROOT=pilot.ROOT
OUT=ROOT/'data/annotations/draft/aligner_v1/long_phone_pilot_v1'
GEN=ROOT/'generated/aligner_v1/long_phone_pilot_v1'
START,END=2003.090,2035.238


def edits(reference,predicted):
    n,m=len(reference),len(predicted)
    dp=np.zeros((n+1,m+1),dtype=int)
    dp[:,0]=np.arange(n+1);dp[0,:]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i,j]=min(dp[i-1,j-1]+(reference[i-1]!=predicted[j-1]),dp[i-1,j]+1,dp[i,j-1]+1)
    i,j=n,m;ops=[]
    while i or j:
        if i and j and dp[i,j]==dp[i-1,j-1]+(reference[i-1]!=predicted[j-1]):
            ops.append(('match' if reference[i-1]==predicted[j-1] else 'substitution',i-1,j-1));i-=1;j-=1
        elif i and dp[i,j]==dp[i-1,j]+1:
            ops.append(('deletion',i-1,None));i-=1
        else:
            ops.append(('insertion',None,j-1));j-=1
    return list(reversed(ops)),int(dp[n,m])


def annotate_phones(ax,rows,y,color_by_op=True):
    last=[-1e9]*3
    for r in rows:
        center=r['anchor_s']
        if not ax.get_xlim()[0]<=center<ax.get_xlim()[1]:continue
        lane=min(range(3),key=lambda k:last[k])
        last[lane]=center
        level=y+lane*.20
        color={'match':'#087f72','substitution':'#bf4c32','deletion':'#7c3fad','insertion':'#bb7300'}.get(r['comparison'],'#394c63')
        ax.plot([r['start_s'],r['end_s']],[y-.10,y-.10],color=color,lw=2.5)
        ax.annotate(r['phone'],xy=(center,y-.11),xytext=(center,level),ha='center',va='bottom',fontsize=10,
                    fontfamily='DejaVu Sans',color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.5))


def render(result,wave,sr):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    panels=math.ceil(result['duration_s']/6)
    fig,axs=plt.subplots(panels,1,figsize=(20,panels*3.35),layout='constrained')
    pieces=[]
    def fill(ax,lo,hi):
        ax.set_xlim(lo,hi);ax.set_ylim(-.1,3.5)
        a,b=int(lo*sr),min(len(wave),int(hi*sr))
        x=np.arange(a,b,8)/sr;v=wave[a:b:8]
        scale=max(float(np.quantile(abs(wave),.995)),.01)
        ax.plot(x,2.83+v/scale*.35,color='#64748b',lw=.5)
        for c in result['characters']:
            mid=(c['start_s']+c['end_s'])/2
            if lo<=mid<hi:
                ax.text(mid,3.26,c['char'],ha='center',fontsize=11)
                ax.plot([c['start_s'],c['end_s']],[3.16,3.16],lw=2,color='#8295a5')
        annotate_phones(ax,result['reference_phones'],1.67)
        annotate_phones(ax,result['greedy_phones'],.55)
        ax.set_yticks([.85,1.97,2.83],['自由识别\n（无文字输入）','文字IPA候选\n（强制定位）','波形 / 汉字'])
        ax.set_xticks(np.arange(math.ceil(lo),hi+.001,.5),minor=True)
        ax.grid(axis='x',which='both',alpha=.12)
        ax.set_xlabel(f'片段内时间 / 秒    [全轨时间 = 此处 + {START:.3f}秒]')
        ax.set_title(f'{lo:.0f}–{hi:.3f}秒',loc='left',fontsize=12)
        for spine in ['top','right']:ax.spines[spine].set_visible(False)
    for i,ax in enumerate(np.atleast_1d(axs)):
        lo,hi=i*6,min((i+1)*6,result['duration_s']);fill(ax,lo,hi)
        sub,subax=plt.subplots(figsize=(20,3.35),layout='constrained');fill(subax,lo,hi)
        p=GEN/f'phones_{i+1:02d}.png';sub.savefig(p,dpi=180);plt.close(sub);pieces.append(p)
    fig.suptitle('32.148秒连续对话 · 上海话音素模型自由识别与文字IPA候选对照\n'
                 '绿色：序列一致　红色：替换　紫色：文字侧未匹配　橙色：模型侧额外音素\n'
                 '整段一次前向推理；分行仅为排版。文字读音与强制时间均非人工真值，差异不等于已确认错误。',fontsize=17)
    fig.savefig(GEN/'long_phone_comparison.png',dpi=180)
    fig.savefig(GEN/'long_phone_comparison.pdf')
    plt.close(fig)
    return pieces


def main():
    OUT.mkdir(parents=True,exist_ok=True);GEN.mkdir(parents=True,exist_ok=True)
    sentences=json.loads((ROOT/'data/annotations/draft/recording_11_20260909/revised_20260910/transcript_revised.json').read_text(encoding='utf-8'))['sentences']
    sentences=[s for s in sentences if 351<=s['id']<=356]
    assert len(sentences)==6 and all(START<s['start']<s['end']<END for s in sentences)
    text=''.join(s['text'] for s in sentences)
    vocab=json.loads((pilot.MODEL/'phone_vocab.json').read_text(encoding='utf-8'))
    converter=pilot.Wugniu(ROOT/'data/lexicons/shanghainese_alignment_v1/wugniu_scheme.json')
    chars=pilot.candidates({'text':text},pilot.lexicon(),converter,vocab)
    # Freeze first-ranked readings before any model inference. Alternative
    # acoustic selection is deliberately not used to inflate the agreement.
    fixed=[{**c,'options':c['options'][:1]} for c in chars]
    pilot.save(OUT/'reading_candidates.json',dict(text=text,characters=fixed))
    source=ROOT/'generated/recording_11_20260909/track_1_16k.wav'
    with sf.SoundFile(source) as f:
        assert f.samplerate==16000 and f.channels==1
        sr=f.samplerate;f.seek(round(START*sr));wave=f.read(round((END-START)*sr),dtype='float32')
    sf.write(OUT/'long_track1.wav',wave,sr,subtype='PCM_16')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Wav2Vec2ForCTC.from_pretrained(pilot.MODEL,local_files_only=True).to(device).eval()
    stride,rf=1,1
    for k,s in zip(model.config.conv_kernel,model.config.conv_stride):rf+=(k-1)*stride;stride*=s
    tensor=torch.from_numpy((wave-wave.mean())/np.sqrt(wave.var()+1e-7)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logp=model(tensor,attention_mask=torch.ones_like(tensor,dtype=torch.long)).logits[0].log_softmax(-1).cpu().numpy()
    np.savez_compressed(GEN/'long_emissions.npz',logp=logp,sr=sr,stride=stride,receptive_field=rf)
    centers=(np.arange(len(logp))*stride+(rf-1)/2)/sr
    nodes,path,score=pilot.lattice_align(logp,fixed,vocab,first_only=True)
    rows=pilot.evidence(nodes,path,fixed,logp,centers,stride/sr,len(wave)/sr)
    pilot.boundaries(rows,wave,sr,len(wave)/sr)
    ref=[]
    for c in rows:
        for p in c['phone_support']:
            ref.append({**p,'char':c['char'],'char_index':c['index'],'anchor_s':(p['start_s']+p['end_s'])/2})
    reverse={v:k for k,v in vocab.items()}
    greedy=[];ids=logp.argmax(-1);t=0
    while t<len(ids):
        end=t+1
        while end<len(ids) and ids[end]==ids[t]:end+=1
        if ids[t]:
            peak=t+int(np.argmax(logp[t:end,ids[t]]))
            greedy.append(dict(phone=reverse[int(ids[t])],start_s=max(0,float(centers[t]-stride/sr/2)),
                               end_s=min(len(wave)/sr,float(centers[end-1]+stride/sr/2)),anchor_s=float(centers[peak]),
                               peak_posterior=float(np.exp(logp[peak,ids[t]]))))
        t=end
    operations,distance=edits([r['phone'] for r in ref],[r['phone'] for r in greedy])
    for op,i,j in operations:
        if i is not None:ref[i]['comparison']=op
        if j is not None:greedy[j]['comparison']=op
    result=dict(start_global_s=START,end_global_s=END,duration_s=len(wave)/sr,text=text,sentences=sentences,
                model=str(pilot.MODEL.relative_to(ROOT)),model_input='complete waveform only; one forward pass, no transcript',
                characters=rows,reference_phones=ref,greedy_phones=greedy,comparison_operations=operations,
                fixed_reading_edit_distance=distance,fixed_reading_phone_count=len(ref),greedy_phone_count=len(greedy),
                comparison_edit_ratio=distance/len(ref),warning='Fixed candidate pronunciation is not verified IPA gold. Ratio is diagnostic, not ASR accuracy.',
                human_listening_performed=False,test_manifest_used=False)
    pilot.save(OUT/'long_phone_results.json',result)
    pieces=render(result,wave,sr)
    audio=base64.b64encode((OUT/'long_track1.wav').read_bytes()).decode()
    cards=''.join(f'<section><h2>{i*6}–{min((i+1)*6,len(wave)/sr):.3f} 秒 <button onclick="jump({i*6})">从这里播放</button></h2><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"></section>' for i,p in enumerate(pieces))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>32秒连续对话：音素对照</title><style>body{font:16px system-ui,"Microsoft YaHei";margin:24px;background:#f5f8fa;color:#183243}header{position:sticky;top:0;background:#f5f8fa;padding:12px;z-index:2;border-bottom:1px solid #b9cad2}section{background:white;padding:16px;margin:20px 0;border-radius:12px}img{width:100%;min-width:1100px}section{overflow-x:auto}h1{font-size:25px}h2{font-size:18px}p{line-height:1.6}audio{width:70%}button,select{font:16px system-ui;padding:5px 12px}</style><header><h1>32.148秒连续对话 · 模型音素对照</h1><audio id="audio" controls src="data:audio/wav;base64,'''+audio+'''"></audio> <select onchange="a.playbackRate=+this.value"><option value="1">1倍速</option><option value="0.8">0.8倍速</option><option value="0.65">0.65倍速</option></select><p>绿：一致　红：替换　紫：文字侧未匹配　橙：模型侧额外音素。颜色按整段序列编辑对照产生，不代表人工确认错误。</p></header><p>自由识别仅输入整段音频，32.148秒一次推理。文字IPA在推理前固定候选读音，另做强制定位；不是人工音素真值。录音全轨时间：2003.090–2035.238秒。</p><details><summary>文字稿</summary><p>'''+html.escape(text)+'''</p></details>'''+cards+'''<script>const a=document.getElementById('audio');function jump(t){a.currentTime=t;a.play()}</script></html>'''
    (OUT/'review.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['duration_s','fixed_reading_edit_distance','fixed_reading_phone_count','greedy_phone_count','comparison_edit_ratio']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
