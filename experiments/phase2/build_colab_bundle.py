"""Build a privacy-scoped Colab ZIP from the validated Phase 2 manifest."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reports" / "phase2" / "dataset_manifest.csv"
DEFAULT_OUTPUT = ROOT / "SpeechSense-Colab.zip"
ARCHIVE_ROOT = Path("Speech-Emotion-Recognition")
REPORT_ALLOWLIST = (
    "dataset_manifest.csv",
    "split_assignments.json",
    "dataset_validation.json",
    "DATASET_VALIDATION.md",
)


def eligible_audio_paths() -> list[Path]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        paths = [
            ROOT / row["audio_path"]
            for row in rows
            if row["model_eligible"].lower() == "true"
        ]
    if len(paths) != 8_491 or len(paths) != len(set(paths)):
        raise RuntimeError("Expected 8,491 unique model-eligible audio files")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing eligible audio file: {missing[0]}")
    return paths


def source_files() -> list[Path]:
    phase2 = ROOT / "experiments" / "phase2"
    files = [
        path
        for path in phase2.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {".pyc", ".pyo"}
    ]
    files.extend(ROOT / "reports" / "phase2" / name for name in REPORT_ALLOWLIST)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")

    files = eligible_audio_paths() + source_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(path.stat().st_size for path in files)
    print(
        f"Bundling {len(files):,} files ({total_bytes / 1024**3:.2f} GiB source)",
        flush=True,
    )
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(ROOT)
            archive.write(path, (ARCHIVE_ROOT / relative).as_posix())
            if index % 500 == 0:
                print(f"Added {index:,}/{len(files):,}", flush=True)

    print(f"Created: {output}")
    print(f"Archive size: {output.stat().st_size / 1024**3:.2f} GiB")
    print("Private real-world recordings and runtime files were not included.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
