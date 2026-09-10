"""Prepare the approved numeric-ID, toneless Shanghai phone training corpus.

Run with the existing qwen3-asr Python. Originals are read only. The primary
index contains all numeric IDs, including missing/conflicting exclusions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf

from wugniu import BLANK, DEFAULT_SCHEME, PROJECT_ROOT, Wugniu, training_syllables


SCHEMA_VERSION = "shanghai_phone_dataset_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_package(package: Path) -> dict[str, Any]:
    """Validate serialized manifests, phone spans and all packaged file hashes."""
    manifest = json.loads((package / "package_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (package / item["path"]).resolve()
        if not path.is_relative_to(package.resolve()) or sha256(path) != item["sha256"]:
            raise AssertionError(f"Package path/hash mismatch: {item['path']}")
    vocab = json.loads((package / "dataset/vocab.json").read_text(encoding="utf-8"))
    assert vocab[BLANK] == 0 and sorted(vocab.values()) == list(range(len(vocab)))
    ids = set()
    count = 0
    for split in ("train", "dev", "test"):
        for line in (package / "dataset" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            assert item["id"] not in ids and item["split"] == split
            ids.add(item["id"])
            assert item["phone_ids"] == [vocab[p] for p in item["phones"]]
            assert 0 not in item["phone_ids"]
            previous = 0
            for syllable in item["syllables"]:
                assert "raw_token" not in syllable and "tone_suffix_raw" not in syllable
                start, end = syllable["phone_span"]
                assert start == previous and start < end <= len(item["phones"])
                assert "".join(item["phones"][start:end]) == syllable["ipa"]
                assert not any(char.isdigit() for char in syllable["spelling"])
                previous = end
            assert previous == len(item["phones"])
            info = sf.info(package / item["audio_path"])
            assert info.samplerate == 16000 and info.channels == 1
            assert info.frames == item["num_samples"] and info.duration == item["duration_s"]
            count += 1
    return {"passed": True, "rows_checked": count, "package_files_hash_checked": len(manifest["files"]),
            "checks": ["unique IDs", "all package file hashes", "portable relative paths", "CTC blank excluded from targets",
                       "vocab ID consistency", "contiguous syllable phone spans", "no tone targets", "audio header/duration consistency"]}


def group_split(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Select whole toneless-reading groups without exhausting any train phone."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["converted"]["toneless_text"]].append(record)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    train_phone_counts = Counter(p for r in records for p in r["converted"]["phones"])
    group_counts = {
        key: Counter(p for r in group for p in r["converted"]["phones"])
        for key, group in grouped.items()
    }
    assignments = {key: "train" for key in keys}
    target = round(len(records) * 0.1)
    decisions = []
    for split in ("dev", "test"):
        count = 0
        for key in keys:
            group = grouped[key]
            if assignments[key] != "train" or count + len(group) > target:
                continue
            if any(train_phone_counts[p] - n <= 0 for p, n in group_counts[key].items()):
                continue
            assignments[key] = split
            train_phone_counts.subtract(group_counts[key])
            count += len(group)
            if count == target:
                break
        decisions.append({"split": split, "target_rows": target, "actual_rows": count})
    groups = []
    for key in sorted(grouped):
        group_id = "reading_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        for record in grouped[key]:
            record["split"] = assignments[key]
            record["group_id"] = group_id
        groups.append({
            "group_id": group_id,
            "toneless_text": key,
            "split": assignments[key],
            "ids": [r["id"] for r in grouped[key]],
        })
    return {"seed": seed, "group_key": "whole_utterance_toneless_spelling", "groups": groups,
            "holdout_decisions": decisions, "train_covers_every_observed_phone": True}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    metadata = args.metadata.resolve()
    audio_root = args.audio_root.resolve()
    package = args.package_root.resolve()
    project = args.project_root.resolve()
    scheme_path = args.scheme.resolve()
    index_path = project / "data/metadata/speaker_readings_v1.csv"
    lexicon_root = project / "data/lexicons/shanghainese_alignment_v1"
    split_root = project / "data/splits/aligner_v1"
    converter = Wugniu(scheme_path)
    previous_annotations = {}
    if index_path.exists():
        with index_path.open(encoding="utf-8-sig", newline="") as handle:
            previous_annotations = {row["id"]: row for row in csv.DictReader(handle)}
    converter.validate_examples()
    original_metadata_hash = sha256(metadata)
    original_scheme_hash = sha256(scheme_path)
    original_segment_index = project / "data/metadata/recording_11_segments.csv"
    segment_hash = sha256(original_segment_index) if original_segment_index.exists() else None
    all_records = []
    seen = set()
    audio_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["file_name", "text"]:
            raise ValueError(f"Unexpected metadata schema: {reader.fieldnames!r}")
        for line, row in enumerate(reader, 2):
            ident = row["file_name"]
            if not re.match(r"^[0-9]", ident):
                continue
            if not re.fullmatch(r"[0-9][A-Za-z0-9_-]*", ident) or ident in seen:
                raise ValueError(f"Unsafe or duplicate ID at line {line}: {ident!r}")
            seen.add(ident)
            converted = converter.convert(row["text"])
            audio = audio_root / f"{ident}.wav"
            record = {
                "id": ident, "metadata_line": line, "raw_text": row["text"],
                "converted": converted, "source_audio": audio,
                "status": "ready", "exclusion_reason": "", "split": "excluded",
            }
            if not audio.is_file():
                record.update(status="excluded_missing_audio", exclusion_reason="same_name_wav_missing")
            else:
                info = sf.info(audio)
                if info.channels != 1 or info.frames <= 0:
                    raise ValueError(f"Expected nonempty mono audio: {audio} ({info})")
                record.update(source_sha256=sha256(audio), source_sample_rate=info.samplerate,
                              source_frames=info.frames, source_duration_s=info.duration,
                              source_subtype=info.subtype)
                audio_by_hash[record["source_sha256"]].append(record)
            all_records.append(record)
    conflicting_groups = []
    for digest, group in audio_by_hash.items():
        if len({r["converted"]["toneless_text"] for r in group}) > 1:
            conflicting_groups.append({"sha256": digest, "ids": [r["id"] for r in group]})
            for record in group:
                record.update(status="excluded_duplicate_audio_conflict",
                              exclusion_reason="identical_audio_different_toneless_transcript")
    if conflicting_groups and args.conflict_policy == "error":
        raise ValueError(f"Conflicting duplicate audio: {conflicting_groups!r}")
    for record in all_records:
        if record["id"] in set(args.exclude_id) and record["status"] == "ready":
            record.update(status="excluded_explicit", exclusion_reason="command_line_exclusion")
    included = [r for r in all_records if r["status"] == "ready"]
    if not included:
        raise ValueError("No included recordings")
    assignments = group_split(included, args.seed)
    phone_list = sorted({p for r in included for p in r["converted"]["phones"]})
    if BLANK in phone_list or any(any(c.isdigit() for c in p) for p in phone_list):
        raise ValueError("Blank or tone digits leaked into phone inventory")
    vocab = {BLANK: 0, **{phone: i for i, phone in enumerate(phone_list, 1)}}
    dataset_records = {split: [] for split in ("train", "dev", "test")}
    hashes = []
    corpus_root = package / "corpus_16k"
    corpus_root.mkdir(parents=True, exist_ok=True)
    for number, record in enumerate(included, 1):
        source = record["source_audio"]
        audio, sample_rate = sf.read(source, dtype="float64")
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Invalid waveform: {source}")
        gcd = math.gcd(sample_rate, 16000)
        derived = resample_poly(audio, 16000 // gcd, sample_rate // gcd)
        output = corpus_root / f"{record['id']}.wav"
        # FLOAT preserves the resampling output without silent clipping/normalization.
        sf.write(output, derived, 16000, subtype="FLOAT")
        info = sf.info(output)
        if info.samplerate != 16000 or info.channels != 1 or info.frames != len(derived):
            raise AssertionError(f"Bad derived audio: {output}")
        if abs(info.duration - record["source_duration_s"]) > 1 / 16000:
            raise AssertionError(f"Duration changed beyond one output sample: {output}")
        record.update(derived_relative=output.relative_to(package).as_posix(),
                      derived_sha256=sha256(output), duration_s=info.duration,
                      derived_frames=info.frames,
                      derived_abs_peak=float(np.max(np.abs(derived))))
        converted = record["converted"]
        item = {
            "id": record["id"], "audio_path": record["derived_relative"],
            "phones": converted["phones"], "phone_ids": [vocab[p] for p in converted["phones"]],
            "syllables": training_syllables(converted), "duration_s": info.duration,
            "sample_rate": 16000, "num_samples": info.frames, "split": record["split"],
            "speaker_id": "speaker_numeric_01", "group_id": record["group_id"],
        }
        dataset_records[record["split"]].append(item)
        hashes.append({"id": record["id"], "source_path": str(source),
                       "source_sha256": record["source_sha256"],
                       "audio_path": record["derived_relative"],
                       "derived_sha256": record["derived_sha256"]})
        if number % 250 == 0:
            print(f"Prepared {number}/{len(included)} recordings", flush=True)
    observed = set(phone_list)
    train_phones = {p for item in dataset_records["train"] for p in item["phones"]}
    if train_phones != observed:
        raise AssertionError("Train does not cover all observed phones")
    group_sets = {split: {r["group_id"] for r in items} for split, items in dataset_records.items()}
    if any(group_sets[a] & group_sets[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise AssertionError("Reading-group leakage")
    split_by_id = {r["id"]: r["split"] for r in included}
    for group in audio_by_hash.values():
        assigned = {split_by_id[r["id"]] for r in group if r["id"] in split_by_id}
        if len(assigned) > 1:
            raise AssertionError("Exact audio duplicates cross splits")
    for record in all_records:
        if "source_sha256" in record and sha256(record["source_audio"]) != record["source_sha256"]:
            raise AssertionError(f"Original audio changed: {record['source_audio']}")
    if sha256(metadata) != original_metadata_hash or sha256(scheme_path) != original_scheme_hash:
        raise AssertionError("Original metadata/scheme changed")
    if segment_hash is not None and sha256(original_segment_index) != segment_hash:
        raise AssertionError("Recording-11 main index changed")
    dataset_dir = package / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, items in dataset_records.items():
        with (dataset_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        write_json(split_root / f"{split}.json", {"split": split, "ids": [r["id"] for r in items]})
    write_json(split_root / "group_assignments.json", assignments)
    convention = {
        "schema_version": SCHEMA_VERSION, "blank_token": BLANK, "blank_id": 0,
        "vocab": vocab, "id_to_phone": [BLANK] + phone_list,
        "phone_span_convention": "zero-based half-open [start,end)",
        "audio_path_base": "package root (parent of dataset directory)",
        "tone_policy": "No tone targets. Original digit suffixes preserved only in primary metadata index.",
        "tokenization": "Whole IPA onset incl. affrication/aspiration; rime base plus attached combining diacritics; NFC; zero onset emits no token.",
        "unknown_policy": "Fail on unknown or ambiguous spelling; no UNK label.",
        "scheme_sha256": original_scheme_hash,
    }
    write_json(dataset_dir / "vocab.json", vocab)
    write_json(dataset_dir / "phone_inventory.json", convention)
    write_json(lexicon_root / "vocab.json", vocab)
    write_json(lexicon_root / "phone_inventory.json", convention)
    write_json(lexicon_root / "wugniu_scheme.json", converter.scheme)
    index_fields = ["id", "source_metadata_path", "source_metadata_sha256", "metadata_line",
                    "source_audio_path", "source_audio_sha256", "source_sample_rate", "source_frames",
                    "source_duration_s", "raw_wugniu", "toneless_wugniu", "tone_suffixes_raw_json",
                    "group_id", "split", "status", "exclusion_reason", "derived_audio_path",
                    "derived_audio_sha256", "derived_duration_s", "speaker_id", "transcript_status",
                    "phone_conversion_status", "alignment_status", "tone_boundary_status"]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fields)
        writer.writeheader()
        for r in all_records:
            writer.writerow({
                "id": r["id"], "source_metadata_path": str(metadata),
                "source_metadata_sha256": original_metadata_hash, "metadata_line": r["metadata_line"],
                "source_audio_path": str(r["source_audio"]), "source_audio_sha256": r.get("source_sha256", ""),
                "source_sample_rate": r.get("source_sample_rate", ""), "source_frames": r.get("source_frames", ""),
                "source_duration_s": r.get("source_duration_s", ""), "raw_wugniu": r["raw_text"],
                "toneless_wugniu": r["converted"]["toneless_text"],
                "tone_suffixes_raw_json": json.dumps(r["converted"]["tone_suffixes_raw"], ensure_ascii=False),
                "group_id": r.get("group_id", ""), "split": r["split"], "status": r["status"],
                "exclusion_reason": r["exclusion_reason"],
                "derived_audio_path": str(package / r["derived_relative"]) if "derived_relative" in r else "",
                "derived_audio_sha256": r.get("derived_sha256", ""), "derived_duration_s": r.get("duration_s", ""),
                "speaker_id": "speaker_numeric_01", "transcript_status": "user_confirmed_clean",
                "phone_conversion_status": "strict_scheme_parsed",
                "alignment_status": previous_annotations.get(r["id"], {}).get("alignment_status", "not_started"),
                "tone_boundary_status": previous_annotations.get(r["id"], {}).get("tone_boundary_status", "not_started_HL_future"),
            })
    pool_counts = {}
    for split, items in dataset_records.items():
        counts = Counter(p for item in items for p in item["phones"])
        pool_counts[split] = {
            "rows": len(items), "groups": len(group_sets[split]),
            "duration_s": sum(item["duration_s"] for item in items),
            "syllables": sum(len(item["syllables"]) for item in items),
            "phone_tokens": sum(counts.values()), "phone_counts": counts,
            "phone_types": len(counts), "missing_observed_phones": sorted(observed - set(counts)),
            "phones_fewer_than_5_tokens": {p: n for p, n in counts.items() if n < 5},
        }
    summary = {
        "schema_version": SCHEMA_VERSION, "approval_scope": "local preparation of user-confirmed clean numeric recordings; no tone targets; no audio alignment or model training here",
        "source_metadata": {"path": str(metadata), "sha256": original_metadata_hash},
        "source_scheme": {"path": str(scheme_path), "sha256": original_scheme_hash},
        "original_recording_11_index_sha256": segment_hash,
        "all_numeric_rows": len(all_records), "included_rows": len(included),
        "status_counts": Counter(r["status"] for r in all_records),
        "conflicting_audio_groups": conflicting_groups, "split_counts": pool_counts,
        "observed_phone_types": len(phone_list), "vocab_size_including_blank": len(vocab),
        "phone_counts_all": Counter(p for r in included for p in r["converted"]["phones"]),
        "seed": args.seed, "train_covers_all_observed_phones": train_phones == observed,
        "source_hashes_verified_unchanged": True, "group_leakage": False,
        "exact_audio_hash_leakage": False,
        "resampling": {"method": "scipy.signal.resample_poly", "sample_rate": 16000,
                       "subtype": "FLOAT", "reason": "preserve resampling peaks without amplitude normalization or PCM clipping",
                       "max_abs_peak": max(r["derived_abs_peak"] for r in included),
                       "original_waveforms_unchanged": True},
        "software": {"numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
        "preparation_source_sha256": {"prepare_phone_corpus.py": sha256(Path(__file__)),
                                      "wugniu.py": sha256(Path(__file__).with_name("wugniu.py"))},
        "training_requires_hanzi": False, "training_includes_tone_labels": False,
        "warnings": ["Phone sequences are text-derived, not acoustic time boundaries.",
                     "Rare phones retained in training; test coverage is reported, not fabricated.",
                     "Held-out word reading does not establish conversational alignment quality."],
    }
    write_json(package / "source_hashes.json", {"metadata_sha256": original_metadata_hash,
               "scheme_sha256": original_scheme_hash, "included_audio": hashes,
               "excluded_existing_audio": [{"id": r["id"], "source_path": str(r["source_audio"]),
                   "source_sha256": r["source_sha256"], "status": r["status"]}
                   for r in all_records if r["status"] != "ready" and "source_sha256" in r]})
    write_json(package / "dataset_summary.json", summary)
    manifest = []
    for artifact in sorted(list(dataset_dir.glob("*.json*")) + [package / "dataset_summary.json", package / "source_hashes.json"]):
        manifest.append({"path": artifact.relative_to(package).as_posix(), "sha256": sha256(artifact), "bytes": artifact.stat().st_size})
    for r in included:
        artifact = package / r["derived_relative"]
        manifest.append({"path": r["derived_relative"], "sha256": r["derived_sha256"], "bytes": artifact.stat().st_size})
    write_json(package / "package_manifest.json", {"schema_version": SCHEMA_VERSION,
               "files": manifest, "note": "Manifest excludes itself; verifies portable dataset and derived audio."})
    consistency = verify_package(package)
    consistency_path = package / "package_consistency.json"
    write_json(consistency_path, consistency)
    manifest.append({"path": "package_consistency.json", "sha256": sha256(consistency_path),
                     "bytes": consistency_path.stat().st_size})
    write_json(package / "package_manifest.json", {"schema_version": SCHEMA_VERSION,
               "files": manifest, "note": "Manifest excludes itself; verifies portable dataset and derived audio."})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT.parent / "metadata.csv")
    parser.add_argument("--audio-root", type=Path, default=PROJECT_ROOT.parent / "wavs")
    parser.add_argument("--scheme", type=Path, default=DEFAULT_SCHEME)
    parser.add_argument("--package-root", type=Path, default=PROJECT_ROOT / "generated/aligner_v1")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--conflict-policy", choices=("exclude", "error"), default="exclude")
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(verify_package(args.package_root.resolve()), ensure_ascii=False, indent=2))
        return
    summary = prepare(args)
    print(json.dumps({"included_rows": summary["included_rows"],
                      "status_counts": summary["status_counts"],
                      "split_counts": {k: {q: v[q] for q in ("rows", "duration_s", "phone_types", "missing_observed_phones")}
                                       for k, v in summary["split_counts"].items()},
                      "vocab_size": summary["vocab_size_including_blank"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
