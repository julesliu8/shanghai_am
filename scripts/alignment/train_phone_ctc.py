"""Train phone CTC for an alignment pilot. CTC emissions are NOT phone boundaries.

Uses only train/dev JSONL. The test manifest is neither opened nor used for selection.
CPU --smoke-test uses a tiny RANDOM model; it cannot produce a usable speech model.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

DEFAULT_MODEL = "TencentGameMate/chinese-wav2vec2-base"
DEFAULT_REVISION = "3991242c806928916fff4a8c0e4f76acf661b743"


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_manifest(path, audio_root, vocab, expected_split):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != expected_split:
                raise ValueError(f"Wrong split for {r['id']}: {r.get('split')}")
            ids = [vocab[p] for p in r["phones"]]  # Unknown phones raise, never become <unk>.
            if not ids or 0 in ids or ids != r["phone_ids"]:
                raise ValueError(f"Invalid phone IDs for {r['id']}")
            wav = (audio_root / r["audio_path"]).resolve()
            if not wav.is_relative_to(audio_root):
                raise ValueError(f"Audio path escapes corpus root: {wav}")
            y, sr = sf.read(wav, dtype="float32")
            if sr != 16000 or y.ndim != 1 or not len(y) or not np.isfinite(y).all():
                raise ValueError(f"Expected finite 16 kHz mono WAV: {wav}")
            y = (y - y.mean()) / np.sqrt(y.var() + 1e-7)
            rows.append({**r, "wave": y, "ids": ids})
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Empty manifest or duplicate IDs: {path}")
    return rows


def collate(rows, device):
    lengths = torch.tensor([len(r["wave"]) for r in rows], dtype=torch.long)
    x = torch.zeros(len(rows), int(lengths.max()), dtype=torch.float32)
    mask = torch.zeros_like(x, dtype=torch.long)
    labels = torch.full((len(rows), max(len(r["ids"]) for r in rows)), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, :lengths[i]] = torch.from_numpy(r["wave"])
        mask[i, :lengths[i]] = 1
        labels[i, :len(r["ids"])] = torch.tensor(r["ids"])
    return {"input_values": x.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}, lengths


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def collapse(ids):
    out, previous = [], None
    for value in ids:
        if value != previous and value != 0:
            out.append(value)
        previous = value
    return out


def evaluate(model, rows, batch_size, device, id2phone, autocast_context):
    model.eval()
    errors = references = 0
    groups, predictions = {}, []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            inputs, lengths = collate(batch, device)
            with autocast_context():
                logits = model(input_values=inputs["input_values"], attention_mask=inputs["attention_mask"]).logits
            output_lengths = model._get_feat_extract_output_lengths(lengths).tolist()
            greedy = logits.argmax(-1).cpu().tolist()
            for r, seq, length in zip(batch, greedy, output_lengths):
                predicted = collapse(seq[:length])
                error, count = edit_distance(r["ids"], predicted), len(r["ids"])
                errors += error
                references += count
                key = r["group_id"]
                group = groups.setdefault(key, [0, 0])
                group[0] += error
                group[1] += count
                predictions.append({"id": r["id"], "group_id": key, "reference_phones": r["phones"],
                                    "predicted_phones": [id2phone[x] for x in predicted],
                                    "edit_distance": error, "reference_phone_count": count})
    return {"phone_error_rate": errors / references,
            "group_macro_per": float(np.mean([e / n for e, n in groups.values()])),
            "edit_distance": errors, "reference_phones": references,
            "utterances": len(rows), "groups": len(groups)}, predictions


def make_model(args, vocab):
    options = dict(vocab_size=len(vocab), pad_token_id=0, bos_token_id=None, eos_token_id=None,
                   ctc_loss_reduction="mean", ctc_zero_infinity=False,
                   mask_time_prob=0.0, mask_feature_prob=0.0, layerdrop=0.0)
    if args.smoke_test:
        cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=64, conv_dim=(16, 16, 16), conv_stride=(5, 2, 2),
                            conv_kernel=(10, 3, 3), num_conv_pos_embeddings=16,
                            num_conv_pos_embedding_groups=2, **options)
        return Wav2Vec2ForCTC(cfg)
    # The pretrained self-supervised output heads are intentionally discarded;
    # lm_head is newly initialized for this project's exact phone inventory.
    return Wav2Vec2ForCTC.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only,
        ignore_mismatched_sizes=True, **options)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--audio-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--unfreeze-last", type=int, default=2)
    p.add_argument("--head-only-steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--max-wall-seconds", type=int, default=7200)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=.001)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", type=Path, help="Only a trusted checkpoint created by this script")
    p.add_argument("--initial-eval", action="store_true", help="Evaluate and preserve starting weights as a best-model baseline")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Random tiny CPU model, one optimizer step; no download")
    a = p.parse_args()
    if not 1 <= a.max_steps <= 30000 or not 1 <= a.max_wall_seconds <= 7200:
        p.error("Hard limits: max-steps <= 30000 and max-wall-seconds <= 7200")
    if min(a.batch_size, a.grad_accum, a.eval_every, a.checkpoint_every, a.patience) < 1:
        p.error("Batch/accum/evaluation/checkpoint/patience must be positive")
    if a.head_lr <= 0 or a.encoder_lr <= 0 or a.clip_grad <= 0 or a.threads < 1:
        p.error("Learning rates, clip-grad and threads must be positive")
    if min(a.unfreeze_last, a.head_only_steps, a.warmup_steps, a.min_delta) < 0:
        p.error("Unfreeze count, warmup/head-only steps and min-delta cannot be negative")
    if a.smoke_test:
        a.device, a.max_steps, a.batch_size, a.grad_accum = "cpu", 1, 2, 1
        a.head_only_steps, a.eval_every, a.checkpoint_every = 0, 1, 1
    return a


def run(args):
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_config.json").exists() and not args.resume:
        raise ValueError("Output already contains a run; use a new directory or --resume")
    status = {"state": "starting", "started_utc": utc(), "step": 0,
              "smoke_test": args.smoke_test, "boundary_validation": "not_performed",
              "warning": "CTC training only; emissions are not complete phone/syllable boundaries"}
    atomic_json(output / "status.json", status)
    stop = {"requested": False}
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda signum, frame: stop.update(requested=True))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    audio_root = (args.audio_root or args.train.parent.parent).resolve()
    vocab = json.loads(args.vocab.read_text(encoding="utf-8-sig"))
    if vocab.get("<blank>") != 0 or sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("vocab must have <blank>:0 and contiguous unique IDs")
    id2phone = {i: p for p, i in vocab.items()}
    train = read_manifest(args.train, audio_root, vocab, "train")
    dev = read_manifest(args.dev, audio_root, vocab, "dev")
    for field in ["id", "group_id"]:
        if {r[field] for r in train} & {r[field] for r in dev}:
            raise ValueError(f"Train/dev overlap in {field}")
    if set(vocab.values()) - {0} - {x for r in train for x in r["ids"]}:
        raise ValueError("Training split does not cover all phones")
    if args.smoke_test:
        train, dev = train[:4], dev[:4]
    model = make_model(args, vocab).to(device)
    layers = model.wav2vec2.encoder.layers
    if not 0 <= args.unfreeze_last <= len(layers):
        raise ValueError("Invalid --unfreeze-last")
    model.requires_grad_(False)
    model.lm_head.requires_grad_(True)
    encoder_params = [p for layer in list(layers)[-args.unfreeze_last:] for p in layer.parameters()] if args.unfreeze_last else []
    optimizer = torch.optim.AdamW([
        {"params": list(model.lm_head.parameters()), "lr": args.head_lr},
        {"params": encoder_params, "lr": args.encoder_lr}], weight_decay=.01)
    def scale(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        return max(0., (args.max_steps - step) / max(1, args.max_steps - args.warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    mixed = device.type == "cuda"
    dtype = torch.bfloat16 if mixed and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=mixed and dtype == torch.float16)
    def autocast_context():
        return torch.autocast("cuda", dtype=dtype) if mixed else nullcontext()
    for r in train + dev:
        frames = int(model._get_feat_extract_output_lengths(len(r["wave"])))
        required = len(r["ids"]) + sum(a == b for a, b in zip(r["ids"], r["ids"][1:]))
        if frames < required:
            raise ValueError(f"CTC impossible: {r['id']} has {frames} frames for {required} required")
    hashes = {"train": sha(args.train), "dev": sha(args.dev), "vocab": sha(args.vocab)}
    architecture = {key: getattr(args, key) for key in ["model", "revision", "smoke_test", "unfreeze_last",
                    "head_only_steps", "head_lr", "encoder_lr", "seed", "batch_size", "grad_accum",
                    "warmup_steps", "max_steps"]}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(manifest_sha256=hashes, train_items=len(train), dev_items=len(dev),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  eventual_trainable_parameters=sum(p.numel() for p in model.lm_head.parameters()) + sum(p.numel() for p in encoder_params),
                  selected_metric="dev.group_macro_per", test_used=False,
                  versions={"torch": torch.__version__, "numpy": np.__version__})
    atomic_json(output / "run_config.json", config)
    atomic_json(output / "vocab.json", vocab)
    model.config.save_pretrained(output / "model_config")
    step, best, stale, elapsed_before, order, position, epoch = 0, float("inf"), 0, 0., [], 0, 0
    if args.resume:
        # Only use local checkpoints produced by this run; pickle state includes RNG/optimizer.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["manifest_sha256"] != hashes or state["vocab"] != vocab:
            raise ValueError("Resume data/vocabulary mismatch")
        if state.get("architecture", architecture) != architecture:
            raise ValueError("Resume backbone/training-stage mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, best, stale = state["step"], state["best"], state["stale"]
        elapsed_before, order, position, epoch = state["elapsed_seconds"], state["order"], state["position"], state["epoch"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if mixed and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    def elapsed():
        return elapsed_before + time.monotonic() - started
    def log(event):
        event = {"utc": utc(), "step": step, "elapsed_seconds": round(elapsed(), 3), **event}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
        print(json.dumps(event, ensure_ascii=False), flush=True)
    def checkpoint(name):
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                       step=step, best=best, stale=stale, elapsed_seconds=elapsed(),
                       order=order, position=position, epoch=epoch, manifest_sha256=hashes, vocab=vocab,
                       architecture=architecture,
                       python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if mixed else None)
        tmp = output / (name + ".tmp")
        with tmp.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output / name)
    reason = "max_steps"
    last_eval = None
    try:
        if args.initial_eval and not args.resume:
            last_eval, predictions = evaluate(model, dev, args.batch_size, device, id2phone, autocast_context)
            best = last_eval["group_macro_per"]
            atomic_json(output / "best_dev_predictions.json", predictions)
            atomic_json(output / "last_dev_metrics.json", {"step": step, **last_eval})
            atomic_json(output / "initial_dev_metrics.json", {"step": step, **last_eval})
            checkpoint("best.pt")
            log({"event": "dev", "baseline": True, **last_eval, "improved": False, "stale_evaluations": 0})
        while step < args.max_steps:
            if stop["requested"] or elapsed() >= args.max_wall_seconds:
                reason = "interrupted" if stop["requested"] else "wall_time_limit"
                break
            train_encoder = step >= args.head_only_steps
            for p in encoder_params:
                p.requires_grad_(train_encoder)
            # Frozen modules stay in evaluation mode, avoiding randomized frozen features.
            model.eval()
            model.lm_head.train()
            if train_encoder:
                for layer in list(layers)[-args.unfreeze_last:] if args.unfreeze_last else []:
                    layer.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for _ in range(args.grad_accum):
                if position >= len(order):
                    order = list(range(len(train)))
                    random.shuffle(order)
                    position, epoch = 0, epoch + 1
                batch = [train[j] for j in order[position:position + args.batch_size]]
                position += len(batch)
                inputs, _ = collate(batch, device)
                with autocast_context():
                    loss = model(**inputs).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite CTC loss; stop rather than hiding invalid targets")
                loss_sum += float(loss.detach())
                scaler.scale(loss / args.grad_accum).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_grad)
            if not torch.isfinite(grad):
                raise RuntimeError("Non-finite gradients")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            log({"event": "train", "loss": loss_sum / args.grad_accum, "epoch": epoch,
                 "grad_norm_before_clip": float(grad), "encoder_unfrozen": train_encoder,
                 "head_lr": optimizer.param_groups[0]["lr"], "encoder_lr": optimizer.param_groups[1]["lr"]})
            status.update(state="training", step=step, elapsed_seconds=round(elapsed(), 3),
                          updated_utc=utc(), best_dev_group_macro_per=None if not np.isfinite(best) else best)
            atomic_json(output / "status.json", status)
            if stop["requested"] or elapsed() >= args.max_wall_seconds:
                reason = "interrupted" if stop["requested"] else "wall_time_limit"
                break
            if step % args.eval_every == 0 or step == args.max_steps:
                last_eval, predictions = evaluate(model, dev, args.batch_size, device, id2phone, autocast_context)
                value = last_eval["group_macro_per"]
                improved = value < best - args.min_delta
                if improved:
                    best, stale = value, 0
                    atomic_json(output / "best_dev_predictions.json", predictions)
                    checkpoint("best.pt")
                elif step > args.head_only_steps:
                    stale += 1
                log({"event": "dev", **last_eval, "improved": improved, "stale_evaluations": stale})
                atomic_json(output / "last_dev_metrics.json", {"step": step, **last_eval})
                if stale >= args.patience:
                    reason = "early_stopping"
                    break
            if step % args.checkpoint_every == 0:
                checkpoint("last.pt")
        checkpoint("last.pt")
        checkpoint("final.pt")
        final_dir = output / f"final_model_step{step:06d}"
        temporary = output / f"final_model_step{step:06d}.tmp"
        if not final_dir.exists():
            temporary.mkdir(exist_ok=True)
            model.save_pretrained(temporary, safe_serialization=True)
            Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True, return_attention_mask=True).save_pretrained(temporary)
            atomic_json(temporary / "phone_vocab.json", vocab)
            os.replace(temporary, final_dir)
        atomic_json(output / "final_model.json", {"path": final_dir.name, "step": step,
                    "selection": "last; best validation checkpoint is best.pt", "smoke_test": args.smoke_test})
        status.update(state="finished", stop_reason=reason, step=step, updated_utc=utc(),
                      elapsed_seconds=round(elapsed(), 3),
                      best_dev_group_macro_per=None if not np.isfinite(best) else best,
                      latest_dev=last_eval, training_performed=step > 0,
                      trained_model_requires_manual_evaluation=True)
        atomic_json(output / "status.json", status)
        log({"event": "finished", "stop_reason": reason})
    except BaseException as exc:
        try:
            checkpoint("last.pt")
        finally:
            status.update(state="failed", step=step, updated_utc=utc(), error=repr(exc), traceback=traceback.format_exc())
            atomic_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    args = arguments()
    if (args.output / "run_config.json").exists() and not args.resume:
        raise SystemExit("Output already contains a run; use a new directory or --resume. Existing status preserved.")
    try:
        run(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "status.json"
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        current.update(state="failed", updated_utc=utc(), error=repr(exc), traceback=traceback.format_exc())
        atomic_json(path, current)
        raise
