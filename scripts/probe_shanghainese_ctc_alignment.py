"""Language-matched character CTC diagnostic, NOT a phoneme aligner or final boundaries."""
from pathlib import Path
import json, re, csv, os
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from analyze_track1_tones import ROOT,GEN,OUT as OLDOUT,save,load,configure_plot
OUT=ROOT/'data/annotations/draft/recording_11_20260909/alignment_pilot'
MODEL='spycsh/shanghainese-wav2vec-3800'
REV='26d9cadd1e75ac9393b56da1fe43ce5ef0d5a7fb'

def force_ctc(logp,targets,blank=0):
    states=np.full(2*len(targets)+1,blank,dtype=int);states[1::2]=targets
    steps,nstates=logp.shape[0],len(states)
    assert steps>=len(targets)
    dp=np.full((steps,nstates),-np.inf);back=np.zeros((steps,nstates),np.int8)
    dp[0,0]=logp[0,blank];dp[0,1]=logp[0,targets[0]]
    skip=np.zeros(nstates,bool)
    skip[2:]=(states[2:]!=blank)&(states[2:]!=states[:-2])
    for t in range(1,steps):
        choices=np.stack([dp[t-1],np.r_[-np.inf,dp[t-1,:-1]],np.r_[-np.inf,-np.inf,dp[t-1,:-2]]])
        choices[2,~skip]=-np.inf
        best=choices.argmax(axis=0);back[t]=best
        dp[t]=choices[best,np.arange(nstates)]+logp[t,states]
    end=nstates-1 if dp[-1,-1]>dp[-1,-2] else nstates-2
    assert np.isfinite(dp[-1,end])
    path=np.zeros(steps,int);path[-1]=end
    for t in range(steps-1,0,-1):path[t-1]=path[t]-back[t,path[t]]
    return path,states,float(dp[-1,end])

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def run():
    OUT.mkdir(parents=True,exist_ok=True);os.environ['HF_HUB_OFFLINE']='1';torch.set_num_threads(4)
    processor=Wav2Vec2Processor.from_pretrained(MODEL,revision=REV,local_files_only=True)
    model=Wav2Vec2ForCTC.from_pretrained(MODEL,revision=REV,local_files_only=True).to('cuda').eval()
    vocab=processor.tokenizer.get_vocab();all_results=[];plt=configure_plot()
    stride=1;rf=1
    for k,s in zip(model.config.conv_kernel,model.config.conv_stride):rf+=(k-1)*stride;stride*=s
    for sample in load(OLDOUT/'pilot_samples.json'):
        text=''.join(re.findall(r'[\u3400-\u9fff]',sample['text']));uid=sample['utterance_id']
        missing=[ch for ch in text if ch not in vocab]
        if missing:
            all_results.append(dict(utterance_id=uid,text=text,status='unsupported_characters',missing=missing));continue
        y,sr=sf.read(OLDOUT/sample['clip_file'],dtype='float32')
        x=processor(y,sampling_rate=sr,return_tensors='pt',padding=True)
        with torch.inference_mode():logits=model(**{k:v.to('cuda') for k,v in x.items()}).logits[0]
        logp=logits.log_softmax(-1).cpu().numpy();greedy=processor.decode(logits.argmax(-1).cpu().numpy())
        targets=[vocab[ch] for ch in text];path,states,score=force_ctc(logp,targets,model.config.pad_token_id)
        rows=[];offset=sample['clip_start_global_s']
        for i,ch in enumerate(text):
            frames=np.flatnonzero(path==2*i+1);assert len(frames)>0
            # This is acoustic CTC evidence duration, not a phonetic interval.
            centers=(frames*stride+(rf-1)/2)/sr
            a=max(0,float(centers[0]-.5*stride/sr));b=min(len(y)/sr,float(centers[-1]+.5*stride/sr))
            posterior=np.exp(logp[frames,targets[i]])
            peak_idx=frames[np.argmax(posterior)];peak=(peak_idx*stride+(rf-1)/2)/sr
            rows.append(dict(char_index=i+1,char=ch,ctc_start_clip_s=a,ctc_end_clip_s=b,
                ctc_start_global_s=offset+a,ctc_end_global_s=offset+b,ctc_peak_global_s=offset+peak,
                mean_target_posterior=float(np.mean(posterior)),max_target_posterior=float(np.max(posterior)),
                ctc_frames=int(len(frames)),status='ctc_emission_support_not_syllable_boundary'))
        result=dict(utterance_id=uid,text=text,model=MODEL,revision=REV,source_wav=str((OLDOUT/sample['clip_file']).relative_to(ROOT)),
            clip_origin_s=offset,greedy_recognition=greedy,character_edit_distance=edit_distance(text,greedy),
            target_characters=len(text),frame_stride_s=stride/sr,receptive_field_s=rf/sr,
            ctc_path_log_score=score,characters=rows,status='diagnostic_only; phonetic_boundaries_not_confirmed')
        save(OUT/f'{uid}_ctc_diagnostic.json',result);all_results.append(result)
        fig,ax=plt.subplots(figsize=(14,3.5),layout='constrained');times=np.arange(len(y))/sr+offset
        ax.plot(times[::8],y[::8],lw=.5,color='#616b77');ylim=max(abs(y).max(),.1)
        previous=-100.;lane=0
        for c in rows:
            mid=c['ctc_peak_global_s']
            if mid-previous<(len(y)/sr)*.03:lane=1-lane
            else:lane=0
            ax.axvspan(c['ctc_start_global_s'],c['ctc_end_global_s'],alpha=.12,color='#00867d')
            ax.annotate(c['char'],xy=(mid,0),xytext=(mid,ylim*(1.05+.22*lane)),fontsize=11,ha='center',
                arrowprops=dict(arrowstyle='-',color='#318b82',lw=.7))
            previous=mid
        ax.set_ylim(-ylim*1.05,ylim*1.5);ax.set_title(f'{uid}\n上海话字符CTC：标出字符声学证据峰；阴影不是已确认音节区间')
        ax.set_xlabel('全轨绝对时间（秒）');ax.set_ylabel('波形幅度');fig.savefig(GEN/f'{uid}_ctc_diagnostic.png',dpi=160);plt.close(fig)
        print(uid,repr(greedy),result['character_edit_distance'],flush=True)
    save(OUT/'ctc_diagnostic_summary.json',all_results)

if __name__=='__main__':run()
