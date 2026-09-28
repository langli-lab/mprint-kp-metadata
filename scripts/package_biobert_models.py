#!/usr/bin/env python3
"""Package the six downloaded checkpoints for a GitHub Release. No uploads."""

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "6_models_MPRINT"
OUTPUT = ROOT / "release-assets" / "biobert-v1"
MODELS = ("Biomarker", "CT", "FBNSTP", "PE", "PK", "VC")
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
OPTIONAL_FILES = ("added_tokens.json",)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    packages = []
    for model in MODELS:
        folder = SOURCE / f"{model}_checkpoint"
        archive = OUTPUT / f"biobert-{model}.zip"
        if folder.is_symlink():
            raise SystemExit(f"Expected a local checkpoint directory: {folder}")
        files = [folder / name for name in REQUIRED_FILES]
        files += [folder / name for name in OPTIONAL_FILES if (folder / name).exists()]
        for path in files:
            if not path.is_file() or path.is_symlink():
                raise SystemExit(f"Missing regular checkpoint file: {path}")
        packages.append((archive, files))

    # Check every destination before writing; existing packages are never replaced.
    checksum_path = OUTPUT / "SHA256SUMS.txt"
    for path in [archive for archive, _ in packages] + [checksum_path]:
        if path.exists():
            raise SystemExit(f"Output already exists; keep it or move it before rebuilding: {path}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    checksums = []
    for archive, files in packages:
        # Stored ZIPs avoid spending CPU recompressing large floating-point weights.
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
            for path in files:
                bundle.write(path, arcname=path.relative_to(SOURCE).as_posix())
        with zipfile.ZipFile(archive) as bundle:
            bad_file = bundle.testzip()
            if bad_file is not None:
                raise SystemExit(f"Archive integrity check failed: {archive}: {bad_file}")
        checksums.append(f"{sha256(archive)}  {archive.name}\n")
        print(f"Verified {archive.name}: {archive.stat().st_size:,} bytes", flush=True)

    with checksum_path.open("x") as stream:
        stream.writelines(checksums)
    print(f"Prepared six archives and {checksum_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
