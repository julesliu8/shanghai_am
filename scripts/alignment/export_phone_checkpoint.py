"""Export weights from a trusted project checkpoint for an explicit optimizer reset.

Never load checkpoints received from untrusted sources: these contain pickle state.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new export directory')
    source = args.checkpoint.resolve()
    state = torch.load(source, map_location='cpu', weights_only=False)
    run = json.loads((source.parent / 'run_config.json').read_text())
    for key, expected in state['manifest_sha256'].items():
        with Path(run[key]).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, key
    assert json.loads(Path(run['vocab']).read_text()) == state['vocab']
    cfg = Wav2Vec2Config.from_pretrained(source.parent / 'model_config', local_files_only=True)
    model = Wav2Vec2ForCTC(cfg)
    model.load_state_dict(state['model'], strict=True)
    assert model.lm_head.out_features == len(state['vocab'])
    temp = args.output.with_name(args.output.name + '.tmp')
    temp.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temp, safe_serialization=True)
    Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True, return_attention_mask=True).save_pretrained(temp)
    (temp / 'phone_vocab.json').write_text(json.dumps(state['vocab'], ensure_ascii=False, indent=2), encoding='utf-8')
    with source.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    provenance = dict(source_checkpoint=str(source), source_sha256=digest,
                      source_step=state['step'], source_best_group_macro_per=state['best'],
                      manifest_sha256=state['manifest_sha256'],
                      export='model weights only; optimizer, scheduler and RNG will be reset',
                      test_used=False)
    (temp / 'source_checkpoint.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    os.replace(temp, args.output)
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
