"""Extract the Phase 2 handcrafted baseline features with resumable caching."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import librosa
import numpy as np
import sklearn

try:
    from .handcrafted_features import (
        FEATURE_CONFIG,
        FEATURE_NAMES,
        extract_handcrafted_features,
    )
except ImportError:  # direct script execution
    from handcrafted_features import (
        FEATURE_CONFIG,
        FEATURE_NAMES,
        extract_handcrafted_features,
    )


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "reports" / "phase2" / "dataset_manifest.csv"
CACHE_DIR = ROOT / "models" / "phase2" / "cache"
FEATURES_PATH = CACHE_DIR / "handcrafted_features.csv"
METADATA_PATH = CACHE_DIR / "handcrafted_features_metadata.json"
ERRORS_PATH = CACHE_DIR / "handcrafted_feature_errors.json"
OUTPUT_FIELDS = ("audio_path",) + FEATURE_NAMES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}. Run dataset_validation.py first."
        )
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["model_eligible"].lower() == "true"]


def completed_paths() -> set[str]:
    if not FEATURES_PATH.is_file():
        return set()
    with FEATURES_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {row["audio_path"] for row in csv.DictReader(handle)}


def check_existing_metadata(manifest_hash: str) -> None:
    if not METADATA_PATH.is_file():
        return
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("manifest_sha256") != manifest_hash:
        raise RuntimeError(
            "The cached features belong to a different manifest. Move the Phase 2 "
            "cache aside before extracting again."
        )
    if metadata.get("feature_config") != FEATURE_CONFIG:
        raise RuntimeError("The cached feature configuration does not match this script.")


def write_metadata(manifest_hash: str, total: int, completed: int, elapsed: float) -> None:
    metadata = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_hash,
        "feature_config": FEATURE_CONFIG,
        "eligible_recordings": total,
        "completed_recordings": completed,
        "complete": completed == total,
        "elapsed_seconds_this_run": elapsed,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "librosa": librosa.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def extract_one(row: dict[str, str]) -> tuple[str, np.ndarray | None, str | None]:
    """Process-pool-safe extraction wrapper."""
    try:
        values = extract_handcrafted_features(ROOT / row["audio_path"])
        return row["audio_path"], values, None
    except Exception as exc:
        return row["audio_path"], None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Extract at most this many new rows.")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes. Use 2 on an 8 GB laptop.",
    )
    args = parser.parse_args()

    rows = load_manifest()
    manifest_hash = file_sha256(MANIFEST_PATH)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    check_existing_metadata(manifest_hash)
    completed = completed_paths()
    pending = [row for row in rows if row["audio_path"] not in completed]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        pending = pending[: args.limit]
    if args.jobs < 1 or args.jobs > 4:
        parser.error("--jobs must be between 1 and 4")

    print(f"Eligible recordings: {len(rows)}")
    print(f"Already cached: {len(completed)}")
    print(f"Extracting this run: {len(pending)}", flush=True)
    if not pending:
        write_metadata(manifest_hash, len(rows), len(completed), 0.0)
        return 0

    errors: list[dict[str, str]] = []
    start = time.perf_counter()
    needs_header = not FEATURES_PATH.exists() or FEATURES_PATH.stat().st_size == 0
    with FEATURES_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if needs_header:
            writer.writeheader()
        if args.jobs == 1:
            iterator = map(extract_one, pending)
            executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=args.jobs)
            iterator = executor.map(extract_one, pending, chunksize=8)
        try:
            for index, (audio_path, values, error) in enumerate(iterator, start=1):
                if error is None and values is not None:
                    writer.writerow(
                        {"audio_path": audio_path}
                        | {name: f"{value:.12g}" for name, value in zip(FEATURE_NAMES, values)}
                    )
                    completed.add(audio_path)
                else:
                    errors.append({"audio_path": audio_path, "error": error or "unknown error"})
                if index % args.progress_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    elapsed = time.perf_counter() - start
                    print(
                        f"Processed {index}/{len(pending)} new rows "
                        f"({index / elapsed:.2f} files/second)",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

    elapsed = time.perf_counter() - start
    write_metadata(manifest_hash, len(rows), len(completed), elapsed)
    ERRORS_PATH.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(f"Cached: {len(completed)}/{len(rows)}")
    print(f"Elapsed: {elapsed:.1f} seconds")
    print(f"Errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
