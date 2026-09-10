"""Build the approved, portable phone-CTC training bundle; never edits sources."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PROJECT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "wuu_phone_ctc_v1"
REVISION = "3991242c806928916fff4a8c0e4f76acf661b743"
MODEL_ID = "TencentGameMate/chinese-wav2vec2-base"
WEIGHT_SHA256 = "be2da40c9e7ae26bfc904a3ed79ebb9e8f060bec6dba85d6a6ae86114bc38901"
WEIGHT_BYTES = 380261837
MODEL_FILES = ("config.json", "preprocessor_config.json", "README.md", "pytorch_model.bin")
DATASET_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl", "vocab.json", "phone_inventory.json")
SCRIPT_FILES = ("train_phone_ctc.py", "requirements_phone_ctc.txt", "README_phone_ctc.md",
                "verify_cloud_package.py", "phone_ctc_model_source.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    rel = PurePosixPath(value)
    if not value or "\\" in value or rel.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError(f"Non-portable relative path: {value!r}")
    if re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"Windows absolute path is not portable: {value!r}")
    return rel


def reject_private_paths(value, context="dataset"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("source_hashes", "source_path", "original_path", "absolute_path"):
                raise ValueError(f"Unexpected source/private-path field in {context}: {key}")
            reject_private_paths(child, context)
    elif isinstance(value, list):
        for child in value:
            reject_private_paths(child, context)
    elif isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("/", "\\\\"))):
        raise ValueError(f"Absolute path in {context}: {value!r}")


def collect_inputs(model_dir: Path):
    generated = PROJECT / "generated" / "aligner_v1"
    dataset = generated / "dataset"
    scripts = PROJECT / "scripts" / "alignment"
    payload: dict[str, Path] = {}
    rows_by_split = {}
    all_ids, all_groups, audio_paths = set(), set(), set()
    for split in ("train", "dev", "test"):
        path = dataset / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        split_groups = set()
        for row in rows:
            reject_private_paths(row, f"{split}/{row.get('id')}")
            if row["split"] != split or row["id"] in all_ids:
                raise ValueError(f"Duplicate ID or wrong split: {row.get('id')}")
            all_ids.add(row["id"])
            split_groups.add(row["group_id"])
            rel = safe_relative(row["audio_path"])
            if rel.parts[0] != "corpus_16k" or rel.suffix.lower() != ".wav":
                raise ValueError(f"Unexpected audio path: {rel}")
            audio = (generated / Path(*rel.parts)).resolve()
            if not audio.is_relative_to((generated / "corpus_16k").resolve()) or not audio.is_file():
                raise ValueError(f"Missing or escaping audio: {rel}")
            if str(rel) in audio_paths:
                raise ValueError(f"Audio referenced more than once: {rel}")
            audio_paths.add(str(rel))
            payload[str(rel)] = audio
        if all_groups & split_groups:
            raise ValueError(f"Reading group crosses split: {split}")
        all_groups.update(split_groups)
        rows_by_split[split] = len(rows)
    expected = {"train": 1064, "dev": 133, "test": 133}
    if rows_by_split != expected or len(audio_paths) != 1330:
        raise ValueError(f"Dataset differs from approved frozen scope: {rows_by_split}")
    for name in DATASET_FILES:
        path = dataset / name
        if name.endswith(".json"):
            reject_private_paths(json.loads(path.read_text(encoding="utf-8-sig")), name)
        payload[f"dataset/{name}"] = path
    for name in SCRIPT_FILES:
        payload[name] = scripts / name
        if name.endswith(".json"):
            reject_private_paths(json.loads((scripts / name).read_text(encoding="utf-8-sig")), name)
    for name in MODEL_FILES:
        payload[f"base_model/{name}"] = model_dir / name
    for rel, path in payload.items():
        safe_relative(rel)
        if not path.is_file():
            raise FileNotFoundError(f"Required package input missing: {rel} ({path})")
    weights = payload["base_model/pytorch_model.bin"]
    if weights.stat().st_size != WEIGHT_BYTES or sha256(weights) != WEIGHT_SHA256:
        raise ValueError("Base-model weights do not match pinned official LFS size/SHA-256")
    return dict(sorted(payload.items())), rows_by_split


def file_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(f"{PACKAGE_NAME}/{name}")
    info.size, info.mode, info.mtime = size, 0o644, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def verify_archive(path: Path, expected: dict):
    found = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            prefix = PACKAGE_NAME + "/"
            if not member.name.startswith(prefix) or not member.isfile():
                raise ValueError(f"Unexpected archive member: {member.name}")
            name = member.name[len(prefix):]
            safe_relative(name)
            if name in found:
                raise ValueError(f"Duplicate archive member: {name}")
            found.add(name)
            if name == "portable_manifest.json":
                continue
            if name not in expected:
                raise ValueError(f"Unlisted archive payload: {name}")
            digest = hashlib.sha256()
            stream = archive.extractfile(member)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if member.size != expected[name]["bytes"] or digest.hexdigest() != expected[name]["sha256"]:
                raise ValueError(f"Archive payload changed during packaging: {name}")
    if found != set(expected) | {"portable_manifest.json"}:
        raise ValueError("Archive member set is incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path.home() / ".cache/huggingface/hub/models--TencentGameMate--chinese-wav2vec2-base/snapshots" / REVISION)
    parser.add_argument("--output", type=Path, default=PROJECT / "generated/aligner_v1/cloud_package/wuu_phone_ctc_v1.tar.gz")
    parser.add_argument("--check-inputs-only", action="store_true", help="Validate and hash inputs without producing an archive")
    args = parser.parse_args()
    payload, rows_by_split = collect_inputs(args.model_dir.resolve())
    entries = {name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in payload.items()}
    manifest = {
        "schema": "wuu_phone_ctc_portable_v1", "package_root": PACKAGE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": {"id": MODEL_ID, "revision": REVISION, "license": "MIT"},
        "dataset_rows": rows_by_split, "audio_files": 1330,
        "files": entries,
        "manifest_scope": "Every regular payload file is listed; portable_manifest.json excludes itself. Archive SHA-256 is recorded outside the archive.",
        "excluded": ["source_hashes.json", "personal absolute source paths", "original unselected audio", "other models", "training outputs"],
        "training_state": "Prepared package only; no cloud instance or training start implied.",
    }
    report = {"payload_files": len(entries), "payload_bytes": sum(x["bytes"] for x in entries.values()),
              "dataset_rows": rows_by_split, "model_weight_sha256": WEIGHT_SHA256,
              "source_script_sha256": {name: entries[name]["sha256"] for name in SCRIPT_FILES},
              "builder_sha256": sha256(Path(__file__)), "status": "inputs_verified"}
    if args.check_inputs_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    output = args.output.resolve()
    source_roots = [PROJECT / "data", PROJECT / "scripts", PROJECT.parent / "wavs", args.model_dir.resolve()]
    if any(output.is_relative_to(x.resolve()) for x in source_roots):
        raise ValueError("Output must not overwrite a source tree")
    if output.exists():
        raise FileExistsError("Archive already exists; choose a new --output path to preserve prior provenance")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Partial archive already exists: {temporary}")
    manifest_data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temporary.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", compresslevel=3, mtime=0, fileobj=raw) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, path in payload.items():
                with path.open("rb") as stream:
                    archive.addfile(file_info(name, entries[name]["bytes"]), stream)
            archive.addfile(file_info("portable_manifest.json", len(manifest_data)), io.BytesIO(manifest_data))
    verify_archive(temporary, entries)
    temporary.replace(output)
    report.update(status="archive_verified", archive_name=output.name, archive_bytes=output.stat().st_size,
                  archive_sha256=sha256(output), portable_manifest_sha256=hashlib.sha256(manifest_data).hexdigest())
    output.with_name(output.name + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
