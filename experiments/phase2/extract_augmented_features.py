"""Cache one deterministic gain-augmented feature vector per training recording."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .handcrafted_features import FEATURE_CONFIG, FEATURE_NAMES, extract_handcrafted_features
    from .extract_features import MANIFEST_PATH, ROOT, file_sha256
except ImportError:
    from handcrafted_features import FEATURE_CONFIG, FEATURE_NAMES, extract_handcrafted_features
    from extract_features import MANIFEST_PATH, ROOT, file_sha256


CACHE_DIR = ROOT / "models" / "phase2" / "cache"
OUTPUT_PATH = CACHE_DIR / "volume_augmented_features.csv"
METADATA_PATH = CACHE_DIR / "volume_augmented_features_metadata.json"
ERRORS_PATH = CACHE_DIR / "volume_augmented_feature_errors.json"
GAINS = (0.5, 0.75, 1.5, 2.0)
RANDOM_SEED = 42
FIELDS = ("audio_path", "gain") + FEATURE_NAMES


def assigned_gain(audio_path: str) -> float:
    digest = hashlib.sha256(f"{RANDOM_SEED}:{audio_path}".encode("utf-8")).digest()
    return GAINS[int.from_bytes(digest[:4], "big") % len(GAINS)]


def load_training_rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row["model_eligible"].lower() == "true" and row["data_split"] == "train"
    ]


def extract_one(row: dict[str, str]) -> tuple[str, float, np.ndarray | None, str | None]:
    gain = assigned_gain(row["audio_path"])
    try:
        values = extract_handcrafted_features(ROOT / row["audio_path"], gain=gain)
        return row["audio_path"], gain, values, None
    except Exception as exc:
        return row["audio_path"], gain, None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 4:
        parser.error("--jobs must be between 1 and 4")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_training_rows()
    manifest_hash = file_sha256(MANIFEST_PATH)
    if OUTPUT_PATH.exists() and METADATA_PATH.exists():
        existing_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if existing_metadata.get("manifest_sha256") != manifest_hash:
            raise RuntimeError(
                "The manifest changed after the augmented cache was created. "
                "Move the old cache aside and restart extraction."
            )
        if existing_metadata.get("feature_config") != FEATURE_CONFIG:
            raise RuntimeError(
                "The feature configuration changed after the augmented cache was "
                "created. Move the old cache aside and restart extraction."
            )
    completed = set()
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
            completed = {row["audio_path"] for row in csv.DictReader(handle)}
    pending = [row for row in rows if row["audio_path"] not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    errors = []
    started = time.perf_counter()
    needs_header = not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0

    with OUTPUT_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            iterator = executor.map(extract_one, pending)
            for index, (audio_path, gain, values, error) in enumerate(iterator, start=1):
                if error is None and values is not None:
                    writer.writerow(
                        {"audio_path": audio_path, "gain": gain}
                        | {name: f"{value:.12g}" for name, value in zip(FEATURE_NAMES, values)}
                    )
                    completed.add(audio_path)
                else:
                    errors.append({"audio_path": audio_path, "gain": gain, "error": error})
                if index % args.progress_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    elapsed = time.perf_counter() - started
                    print(
                        f"Processed {index}/{len(pending)} "
                        f"({index / elapsed:.2f} files/second)",
                        flush=True,
                    )

    elapsed = time.perf_counter() - started
    metadata = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "feature_config": FEATURE_CONFIG,
        "random_seed": RANDOM_SEED,
        "gain_choices": list(GAINS),
        "strategy": "one deterministic gain-augmented copy per original training recording",
        "training_recordings": len(rows),
        "completed_recordings": len(completed),
        "complete": len(completed) == len(rows),
        "elapsed_seconds_this_run": elapsed,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ERRORS_PATH.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(f"Cached: {len(completed)}/{len(rows)}")
    print(f"Elapsed: {elapsed:.1f} seconds; errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
