"""Five-clip, text-constrained phone-lattice alignment. All boundaries are drafts.

Uses the trained phone vocabulary, NOT a Chinese-text tokenizer. Allows complete
dictionary pronunciation alternatives per character; CTC cannot skip characters.
    Blank gaps yield explicit uncertain boundary regions, not measured phone gold.
"""
from pathlib import Path
import base64
import collections
import csv
import ctypes
import hashlib
import html
import json
import re
import shutil

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC
from wugniu import Wugniu

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'data/annotations/draft/recording_11_20260909/tone_pilot'
OUT = ROOT / 'data/annotations/draft/aligner_v1/character_pilot_v1'
GEN = ROOT / 'generated/aligner_v1/character_pilot_v1'
MODEL = ROOT / 'models/aligner_v1/v3_best_step006750'
DICT = ROOT.parent / 'shanghai_app/processed_results.csv'
RIME = GEN / 'wugniu_zaonhe.dict.yaml'


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def simplify(text):
    # Windows' character conversion changes lookup keys only, never the transcript.
    f = ctypes.windll.kernel32.LCMapStringEx
    f.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_int,
                  ctypes.c_wchar_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    buf = ctypes.create_unicode_buffer(len(text) * 2 + 2)
    n = f('zh-CN', 0x02000000, text, len(text), buf, len(buf), None, None, 0)
    if not n:
        raise OSError('Simplified lookup conversion failed')
    return buf[:n]


def lexicon():
    entries = []
    for line, r in enumerate(csv.reader(DICT.open(encoding='utf-8-sig')), 1):
        if len(r) < 6:
            continue
        word = r[1].strip('【】')
        syllables = re.findall(r'[a-z]+[0-9]*', r[-1])
        if re.fullmatch(r'[\u3400-\u9fff]+', word) and len(word) == len(syllables):
            entries.append((word, syllables, f'processed_results.csv:{line}'))
    for line, row in enumerate(RIME.read_text(encoding='utf-8').splitlines(), 1):
        cells = row.split('\t')
        if len(cells) < 2:
            continue
        word, syllables = simplify(cells[0]), cells[1].split()
        if re.fullmatch(r'[\u3400-\u9fff]+', word) and len(word) == len(syllables):
            entries.append((word, syllables, f'wugniu_zaonhe.dict.yaml:{line}'))
    return entries


def candidates(sample, entries, converter, vocab):
    chars = re.findall(r'[\u3400-\u9fff]', sample['text'])
    text = ''.join(chars)
    rows = []
    # User transcript spellings can stand for colloquial particles. These are
    # explicit hypotheses, not silent rewrites or dictionary attestations.
    colloquial = {'的': ['geq', 'ku'], '么': ['meq', 'm'], '了': ['leq', 'la'], '呃': ['eq']}
    for i, ch in enumerate(chars):
        pool = []
        for word, syllables, source in entries:
            for j, c in enumerate(word):
                if c != ch:
                    continue
                contextual = 0 <= i-j and text[i-j:i-j+len(word)] == word
                rank = -10-len(word) if contextual and len(word) > 1 else len(word)
                pool.append((rank, word, syllables[j], source))
        options, seen = [], set()
        for rank, word, spelling, source in sorted(pool):
            try:
                p = converter.parse_syllable(spelling)
            except ValueError:
                continue
            phones = p['phones']
            note = 'dictionary'
            # Tight sonorant onsets are absent from training; approximate with
            # the corresponding sonorant instead of inventing an unknown ID.
            phones = [x[1:] if x not in vocab and x.startswith('ʔ') and x[1:] in vocab else x for x in phones]
            if phones != p['phones']:
                note = 'unseen_tight_onset_approximated_by_sonorant'
            if any(x not in vocab for x in phones):
                continue
            key = tuple(phones)
            if key in seen:
                continue
            seen.add(key)
            options.append(dict(spelling=p['spelling'], phones=phones, source=source,
                                source_word=word, note=note, rank=rank))
            if len(options) >= 6:
                break
        for spelling in colloquial.get(ch, []):
            p = converter.parse_syllable(spelling)
            if tuple(p['phones']) not in seen:
                seen.add(tuple(p['phones']))
                options.insert(0, dict(spelling=p['spelling'], phones=p['phones'],
                                      source='explicit_colloquial_transcript_hypothesis', source_word=ch,
                                      note='unconfirmed_colloquial_reading', rank=-20))
        # Permit unreleased/weakened glottal codas, but never delete a whole syllable.
        for option in list(options):
            phones = option['phones']
            if len(phones) > 1 and phones[-1] == 'ʔ' and tuple(phones[:-1]) not in seen:
                seen.add(tuple(phones[:-1]))
                options.append({**option, 'phones': phones[:-1], 'note': option['note']+';optional_glottal_coda_omitted'})
        if not options:
            raise ValueError(f'No supported reading for {ch} at {i}')
        rows.append(dict(index=i+1, char=ch, options=options))
    return rows


def lattice_align(logp, chars, vocab, first_only=False):
    """Viterbi on a CTC DAG with whole-pronunciation branches per character."""
    nodes = [dict(label=0, char=-1, option=-1, phone=-1)]
    predecessors = [[0]]
    start = [0]
    previous_ends = []
    boundary = 0
    for ci, row in enumerate(chars):
        ends = []
        options = row['options'][:1] if first_only else row['options']
        for oi, option in enumerate(options):
            previous_phone = None
            previous_blank = boundary
            for pi, phone in enumerate(option['phones']):
                label = vocab[phone]
                n = len(nodes)
                nodes.append(dict(label=label, char=ci, option=oi, phone=pi))
                pred = [n, previous_blank]
                if pi == 0:
                    pred += [e for e in previous_ends if nodes[e]['label'] != label]
                    if ci == 0:
                        start.append(n)
                elif nodes[previous_phone]['label'] != label:
                    pred.append(previous_phone)
                predecessors.append(pred)
                previous_phone = n
                if pi < len(option['phones'])-1:
                    b = len(nodes)
                    nodes.append(dict(label=0, char=ci, option=oi, phone=-1))
                    predecessors.append([b, n])
                    previous_blank = b
            ends.append(previous_phone)
        boundary = len(nodes)
        nodes.append(dict(label=0, char=ci, option=-1, phone=-1))
        predecessors.append([boundary] + ends)
        previous_ends = ends
    labels = np.array([n['label'] for n in nodes])
    width = max(map(len, predecessors))
    incoming = np.full((len(nodes), width), len(nodes), dtype=np.int32)
    for i, pred in enumerate(predecessors):
        incoming[i, :len(pred)] = pred
    score = np.full(len(nodes)+1, -np.inf)
    score[start] = logp[0, labels[start]]
    back = np.zeros((len(logp), len(nodes)), dtype=np.int32)
    for t in range(1, len(logp)):
        best = np.argmax(score[incoming], axis=1)
        chosen = incoming[np.arange(len(nodes)), best]
        back[t] = chosen
        score[:-1] = score[chosen] + logp[t, labels]
    final = [boundary] + previous_ends
    end = final[int(np.argmax(score[final]))]
    if not np.isfinite(score[end]):
        raise ValueError('No complete CTC path; do not fabricate boundaries')
    value = float(score[end])
    path = np.empty(len(logp), dtype=np.int32)
    path[-1] = end
    for t in range(len(logp)-1, 0, -1):
        path[t-1] = back[t, path[t]]
    selected = [nodes[i] for i in path]
    for ci in range(len(chars)):
        assert any(n['char'] == ci and n['label'] for n in selected)
    return nodes, path, value


def self_test():
    vocab = {'a': 1, 'b': 2}
    chars = [{'options': [{'phones': ['b']}, {'phones': ['a']}]}, {'options': [{'phones': ['a']}]}]
    p = np.full((5, 3), -12.)
    p[np.arange(5), [0, 1, 0, 1, 0]] = 0
    nodes, path, _ = lattice_align(p, chars, vocab)
    first = [nodes[n] for n in path if nodes[n]['char'] == 0 and nodes[n]['label']]
    assert first and all(n['option'] == 1 for n in first)
    assert any(nodes[n]['label'] == 0 for n in path[2:3])
    # Impossible duplicate tokens in two frames cannot skip the required blank.
    try:
        lattice_align(p[:2], [{'options': [{'phones': ['a']}]}, {'options': [{'phones': ['a']}]}], vocab)
    except ValueError:
        pass
    else:
        raise AssertionError('Repeated label skipped blank')


def evidence(nodes, path, rows, logp, centers, stride_s, duration):
    result = []
    for ci, row in enumerate(rows):
        frames = np.array([t for t, n in enumerate(path) if nodes[n]['char'] == ci and nodes[n]['label']])
        option_ids = {nodes[path[t]]['option'] for t in frames}
        assert len(option_ids) == 1
        option = row['options'][option_ids.pop()]
        post = np.array([np.exp(logp[t, nodes[path[t]]['label']]) for t in frames])
        phone_rows = []
        for pi, phone in enumerate(option['phones']):
            f = np.array([t for t in frames if nodes[path[t]]['phone'] == pi])
            assert len(f)
            phone_rows.append(dict(phone=phone, start_s=max(0., float(centers[f[0]]-stride_s/2)),
                                   end_s=min(duration, float(centers[f[-1]]+stride_s/2)),
                                   peak_posterior=float(max(np.exp(logp[t, nodes[path[t]]['label']]) for t in f))))
        result.append(dict(index=ci+1, char=row['char'], spelling=option['spelling'],
                           phones=option['phones'], reading_source=option,
                           support_start_s=max(0., float(centers[frames[0]]-stride_s/2)),
                           support_end_s=min(duration, float(centers[frames[-1]]+stride_s/2)),
                           anchor_s=float(np.average(centers[frames], weights=post+1e-8)),
                           mean_emission_posterior=float(post.mean()),
                           phone_support=phone_rows))
    return result


def boundaries(rows, wave, sr, duration):
    """Conservative heuristic: energy valley between adjacent CTC support spans.

    Only acoustically quiet blank gaps remain unassigned pauses. Bounds are evidence windows, not
    probabilistic confidence intervals; connected speech may have no energy dip.
    """
    width = max(1, int(sr*.01))
    rms = np.sqrt(np.convolve(wave.astype(float)**2, np.ones(width)/width, mode='same'))
    threshold = max(1e-5, float(np.quantile(rms, .95))*.035)
    quiet = rms < threshold
    changes = np.diff(np.r_[False, quiet, False].astype(np.int8))
    quiet_spans = [(a/sr,b/sr) for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)) if (b-a)/sr>=.05]
    first = rows[0]['support_start_s']
    before = [(a,b) for a,b in quiet_spans if b<=first and b>=first-.3]
    rows[0]['start_s'] = max(0., before[-1][1]-.005 if before else first-.03)
    last = rows[-1]['support_end_s']
    after = [(a,b) for a,b in quiet_spans if a>=last and a<=last+.3]
    rows[-1]['end_s'] = min(duration, after[0][0]+.005 if after else last+.03)
    for a, b in zip(rows, rows[1:]):
        left, right = a['support_end_s'], b['support_start_s']
        assert right >= left-1e-7
        pauses = [(max(left,x),min(right,y)) for x,y in quiet_spans if min(right,y)-max(left,x)>=.05]
        if pauses:
            # A CTC blank may be a sustained vowel: only measured quiet earns
            # an unlabelled pause, never the duration of blank alone.
            qleft,qright = max(pauses,key=lambda xy:xy[1]-xy[0])
            a['end_s'], b['start_s'] = qleft, qright
            method = 'rms_confirmed_quiet_gap_unassigned'
        elif right-left >= .025:
            lo, hi = int(left*sr), min(len(rms), int(right*sr))
            split = (lo + int(np.argmin(rms[lo:hi]))) / sr
            a['end_s'] = b['start_s'] = split
            method = 'minimum_10ms_rms_inside_blank_gap'
        else:
            a['end_s'] = b['start_s'] = (left+right)/2
            method = 'short_gap_midpoint'
        a['right_boundary_evidence_window_s'] = [left, right]
        a['right_boundary_method'] = method
    for r in rows:
        r['issues'] = []
        if r['mean_emission_posterior'] < .15:
            r['issues'].append('weak_acoustic_support')
        if r['end_s']-r['start_s'] < .06:
            r['issues'].append('very_short_character')
        if r['end_s']-r['start_s'] > .60:
            r['issues'].append('long_character_or_internal_pause')
        if any(p['peak_posterior'] < .05 for p in r['phone_support']):
            r['issues'].append('one_or_more_phones_mismatch')
        r['status'] = 'machine_candidate_needs_listening'
        r['quiet_rms_threshold'] = threshold


def textgrid(path, result):
    import parselmouth
    from parselmouth.praat import call
    duration = result['duration_s']
    tiers = {'syllable': [(r['start_s'], r['end_s'], r['char']) for r in result['characters']],
             'word': [], 'AP': [], 'misc': [],
             'phone_support': [(p['start_s'], p['end_s'], p['phone']) for r in result['characters'] for p in r['phone_support']],
             'old_qwen': [(r['start_time'], r['end_time'], r['text']) for r in result['old_tokens'] if r['end_time']>r['start_time']],
             'review': [(r['start_s'], r['end_s'], ';'.join(r['issues'])) for r in result['characters'] if r['issues']]}
    grid = call('Create TextGrid...', 0, duration, ' '.join(tiers), '')
    for ti, items in enumerate(tiers.values(), 1):
        cuts = sorted({v for a, b, _ in items for v in [a, b] if 0 < v < duration})
        for cut in cuts:
            call(grid, 'Insert boundary', ti, cut)
        for a, b, label in items:
            interval = call(grid, 'Get interval at time', ti, (a+b)/2)
            call(grid, 'Set interval text', ti, interval, label)
    call(grid, 'Save as text file', str(path))
    parselmouth.read(str(path))


def plot(result, wave, sr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(3, 1, figsize=(15, 6), sharex=True, layout='constrained', height_ratios=[2, 2, 1])
    t = np.arange(len(wave))/sr
    axs[0].plot(t[::8], wave[::8], lw=.5, color='#475569')
    axs[1].specgram(wave, NFFT=256, Fs=sr, noverlap=200, cmap='magma', vmin=-110, vmax=-30)
    axs[1].set_ylim(0, 5000)
    for r in result['characters']:
        mid=(r['start_s']+r['end_s'])/2
        color = '#cf6a20' if r['issues'] else '#008577'
        axs[0].axvline(r['start_s'], color=color, alpha=.6, lw=.7)
        axs[0].text(mid, 1.02, r['char'], transform=axs[0].get_xaxis_transform(), ha='center', fontsize=10)
        axs[1].axvline(r['start_s'], color='white', alpha=.6, lw=.5)
        axs[2].plot([r['start_s'], r['end_s']], [.75, .75], lw=8, color=color)
        axs[2].text(mid, .94, r['char'], ha='center', fontsize=10)
    for r in result['old_tokens']:
        axs[2].plot([r['start_time'], r['end_time']], [.15, .15], lw=7, color='#94a3b8')
        axs[2].text((r['start_time']+r['end_time'])/2, -.08, r['text'], ha='center', fontsize=9)
    axs[0].set_title(result['text']+'\n新候选：橙色需重点核听；竖线为估计字起点', pad=30)
    axs[1].set_ylabel('频率 Hz')
    axs[2].set_ylim(-.35, 1.2); axs[2].set_yticks([.15,.75], ['旧Qwen','新phone']); axs[2].set_xlabel('片段内时间（秒）')
    axs[2].set_xlim(0, result['duration_s'])
    path = GEN/(result['utterance_id']+'.png')
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def review_page(results):
    sections=[]
    for i, r in enumerate(results):
        wav=base64.b64encode((OLD/r['clip_file']).read_bytes()).decode()
        picture=base64.b64encode((GEN/(r['utterance_id']+'.png')).read_bytes()).decode()
        buttons=''.join(f'<button data-start="{c["start_s"]:.6f}" data-end="{c["end_s"]:.6f}" onclick="playChar({i},this)" title="{html.escape(c["spelling"])} {c["start_s"]:.3f}–{c["end_s"]:.3f}秒" class="{"flag" if c["issues"] else ""}">{c["char"]}</button>' for c in r['characters'])
        rows=''.join(f'<tr><td>{c["index"]}</td><td>{c["char"]}</td><td>{html.escape(c["spelling"])}</td><td>{c["start_s"]:.3f}–{c["end_s"]:.3f}</td><td>{c["old_start_s"]:.3f}–{c["old_end_s"]:.3f}</td><td>{html.escape(", ".join(c["issues"]) or "—")}</td></tr>' for c in r['characters'])
        sections.append(f'<section><h2>{r["utterance_id"].split("_")[-1]} · {html.escape(r["text"])}</h2><audio id="a{i}" controls src="data:audio/wav;base64,{wav}"></audio><label>速度 <select onchange="document.getElementById(\'a{i}\').playbackRate=+this.value"><option value="1">1×</option><option value="0.8">0.8×</option><option value="0.65">0.65×</option></select></label><div class="chars" id="c{i}">{buttons}</div><p>点击汉字播放该字及前后40毫秒。整句播放时高亮当前字；橙色是自动疑点，不等同于错字。</p><img src="data:image/png;base64,{picture}"><details><summary>逐字时间与旧结果对照</summary><table><tr><th>序号</th><th>汉字</th><th>选用读音</th><th>新时间/秒</th><th>旧时间/秒</th><th>自动疑点</th></tr>{rows}</table></details></section>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>上海话：5段汉字对齐试听</title><style>body{font:16px system-ui,"Microsoft YaHei";max-width:1200px;margin:30px auto;padding:0 20px;background:#f4f7fa;color:#172a3a}section{background:white;padding:24px;margin:24px 0;border-radius:14px}h1{font-size:26px}h2{font-size:21px}p{line-height:1.7;color:#526171}button{font:24px system-ui;margin:4px;padding:7px 13px;border:1px solid #bccbd2;border-radius:7px;background:#edf8f5;cursor:pointer}.flag{background:#fff0da}button.active{background:#0c756e;color:white}img{width:100%}audio{width:75%;vertical-align:middle}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #dde3e8;padding:8px}summary{cursor:pointer;padding:12px 0}</style><h1>上海话 · 汉字对齐试听</h1><a href="../long_phone_pilot_v1/review.html">查看32秒长段音素对照</a><p>使用给定文字与吴拼→IPA读音候选，约束已训练的音素模型寻找单调路径。无需先把所有音素识别正确。字区间是候选：CTC声学支撑与空白段结合能量谷估计，仅声学静音保留间隔；尚未逐字人工验收。所有时间为当前片段内时间，完整录音时间另见JSON。原字幕仅作对照。</p>'''+''.join(sections)+'''<script>
const stops={};function playChar(i,b){const a=document.getElementById('a'+i);document.querySelectorAll('audio').forEach(other=>{if(other!==a)other.pause()});a.currentTime=Math.max(0,+b.dataset.start-.04);stops[i]=+b.dataset.end+.04;a.play()}
function tick(){document.querySelectorAll('audio').forEach((a,i)=>{if(stops[i]!=null&&a.currentTime>=stops[i]){a.pause();delete stops[i]}document.querySelectorAll('#c'+i+' button').forEach(b=>b.classList.toggle('active',a.currentTime>=+b.dataset.start&&a.currentTime<+b.dataset.end))});requestAnimationFrame(tick)}requestAnimationFrame(tick);
</script></html>'''
    (OUT/'review.html').write_text(page, encoding='utf-8')


def main():
    self_test()
    OUT.mkdir(parents=True, exist_ok=True); GEN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, local_files_only=True).eval()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    vocab=json.loads((MODEL/'phone_vocab.json').read_text(encoding='utf-8'))
    converter=Wugniu(ROOT/'data/lexicons/shanghainese_alignment_v1/wugniu_scheme.json')
    entries=lexicon()
    samples=json.loads((OLD/'pilot_samples.json').read_text(encoding='utf-8'))
    plans=[dict(utterance_id=s['utterance_id'],text=s['text'],characters=candidates(s,entries,converter,vocab)) for s in samples]
    save(OUT/'reading_candidates.json', plans)
    stride, rf=1,1
    for k,s in zip(model.config.conv_kernel,model.config.conv_stride):
        rf+=(k-1)*stride;stride*=s
    results=[]
    for sample, plan in zip(samples,plans):
        wave,sr=sf.read(OLD/sample['clip_file'],dtype='float32')
        assert sr==16000 and wave.ndim==1
        duration=len(wave)/sr
        x=(wave-wave.mean())/np.sqrt(wave.var()+1e-7)
        tensor=torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.inference_mode():
            logp=model(tensor,attention_mask=torch.ones_like(tensor,dtype=torch.long)).logits[0].log_softmax(-1).cpu().numpy()
        np.savez_compressed(GEN/(sample['utterance_id']+'_emissions.npz'),logp=logp,sr=sr,stride=stride,receptive_field=rf)
        centers=(np.arange(len(logp))*stride+(rf-1)/2)/sr
        nodes,path,score=lattice_align(logp,plan['characters'],vocab)
        rows=evidence(nodes,path,plan['characters'],logp,centers,stride/sr,duration)
        # Independent fixed-first-reading control: compare anchors, not to gold.
        fixed_nodes,fixed_path,fixed_score=lattice_align(logp,plan['characters'],vocab,first_only=True)
        fixed=evidence(fixed_nodes,fixed_path,plan['characters'],logp,centers,stride/sr,duration)
        boundaries(rows,wave,sr,duration)
        for row,old,control in zip(rows,sample['tokens'],fixed):
            assert row['char']==old['text']
            row.update(start_global_s=sample['clip_start_global_s']+row['start_s'],end_global_s=sample['clip_start_global_s']+row['end_s'],
                       old_start_s=old['start_time'],old_end_s=old['end_time'],fixed_reading_anchor_s=control['anchor_s'],
                       reading_sensitivity_s=abs(row['anchor_s']-control['anchor_s']))
            if row['reading_sensitivity_s']>.10:
                row['issues'].append('reading_sensitive_over_100ms')
            assert 0<=row['start_s']<row['end_s']<=duration
        assert all(a['end_s']<=b['start_s']+1e-7 for a,b in zip(rows,rows[1:]))
        greedy=[];previous=None
        reverse={v:k for k,v in vocab.items()}
        for idx in logp.argmax(-1):
            if idx!=previous and idx:greedy.append(reverse[int(idx)])
            previous=idx
        result=dict(utterance_id=sample['utterance_id'],text=sample['text'],clip_file=sample['clip_file'],
                    clip_start_global_s=sample['clip_start_global_s'],duration_s=duration,characters=rows,
                    old_tokens=sample['tokens'],greedy_phones=greedy,ctc_path_score=score,fixed_reading_score=fixed_score,
                    model=str(MODEL.relative_to(ROOT)),frame_stride_s=stride/sr,receptive_field_s=rf/sr,
                    status='machine_character_alignment_candidate',human_listening_performed=False)
        save(OUT/(sample['utterance_id']+'.json'),result)
        shutil.copy2(OLD/sample['clip_file'], OUT/sample['clip_file'])
        textgrid(OUT/(sample['utterance_id']+'.TextGrid'),result)
        plot(result,wave,sr)
        results.append(result)
        print(sample['utterance_id'],''.join(r['char'] for r in rows),'flagged',sum(bool(r['issues']) for r in rows),flush=True)
    save(OUT/'results.json',results)
    summary=dict(model=str(MODEL.relative_to(ROOT)),samples=len(results),characters=sum(len(r['characters']) for r in results),
                 old_zero_duration_characters=sum(t['end_time']==t['start_time'] for r in results for t in r['old_tokens']),
                 new_zero_duration_characters=0,flagged_characters=sum(bool(c['issues']) for r in results for c in r['characters']),
                 reading_sensitive_characters=sum(c['reading_sensitivity_s']>.1 for r in results for c in r['characters']),
                 boundary_accuracy='not_measured_no_reference',test_manifest_used=False,
                 dictionary_url='https://raw.githubusercontent.com/NGLI/rime-wugniu_zaonhe/master/wugniu_zaonhe.dict.yaml',
                 dictionary_sha256=hashlib.sha256(RIME.read_bytes()).hexdigest(),
                 dictionary_license='Rime source credits GPLv3; derived reading candidates remain local',
                 model_sha256=hashlib.sha256((MODEL/'model.safetensors').read_bytes()).hexdigest())
    save(OUT/'summary.json',summary)
    review_page(results)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
