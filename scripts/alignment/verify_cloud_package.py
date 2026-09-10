"""Read-only package hash and Torch/CUDA preflight; run from the extracted bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-cuda", action="store_true", help="Fail if Torch>=2.6 or CUDA is unavailable")
    parser.add_argument("--skip-runtime-check", action="store_true", help="Verify package files without importing Torch")
    args = parser.parse_args()
    if args.require_cuda and args.skip_runtime_check:
        parser.error("--require-cuda conflicts with --skip-runtime-check")
    root = args.root.resolve()
    manifest_path = root / "portable_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "wuu_phone_ctc_portable_v1":
        raise ValueError("Unsupported portable manifest schema")
    failures = []
    total_bytes = 0
    for name, expected in manifest["files"].items():
        rel = PurePosixPath(name)
        if (not name or "\\" in name or rel.is_absolute() or re.match(r"^[A-Za-z]:", name)
                or any(x in ("", ".", "..") for x in name.split("/"))):
            raise ValueError(f"Unsafe manifest path: {name!r}")
        path = root.joinpath(*rel.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            failures.append({"path": name, "error": "missing, symlink, or outside bundle"})
            continue
        size = path.stat().st_size
        if size != expected["bytes"] or sha256(path) != expected["sha256"]:
            failures.append({"path": name, "error": "size or SHA-256 mismatch"})
        total_bytes += size
    runtime = {"python": platform.python_version(), "platform": platform.system(), "checked": False}
    if not args.skip_runtime_check:
        try:
            import torch
            version = str(torch.__version__)
            numbers = re.match(r"^(\d+)\.(\d+)", version)
            supported = bool(numbers and tuple(map(int, numbers.groups())) >= (2, 6))
            available = torch.cuda.is_available()
            runtime.update(checked=True, torch=version, torch_at_least_2_6=supported,
                           torch_cuda_build=torch.version.cuda, cuda_available=available,
                           gpu_count=torch.cuda.device_count() if available else 0)
            if available:
                runtime["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                                   for i in range(torch.cuda.device_count())]
            if args.require_cuda and not (available and supported):
                failures.append({"error": "CUDA training requires Torch>=2.6 and an available CUDA GPU"})
        except ImportError as exc:
            runtime["import_error"] = str(exc)
            if args.require_cuda:
                failures.append({"error": "Torch unavailable"})
    print(json.dumps({"status": "passed" if not failures else "failed", "verified_payload_files": len(manifest["files"]),
                      "payload_bytes": total_bytes, "dataset_rows": manifest.get("dataset_rows"),
                      "source_model": manifest.get("source_model"), "failures": failures,
                      "runtime": runtime, "note": "No training, writes, downloads or instance shutdown performed."},
                     ensure_ascii=False, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
