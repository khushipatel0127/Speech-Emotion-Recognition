"""Build and validate the leakage-safe SpeechSense Phase 2 dataset manifest.

This script intentionally uses only the Python standard library so data
validation can run before the machine-learning environment is installed.
Original audio is opened read-only and is never rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
CREMAD_DIR = DATASET_DIR / "CREMA-D"
REPORT_DIR = ROOT / "reports" / "phase2"
MANIFEST_PATH = REPORT_DIR / "dataset_manifest.csv"
SPLITS_PATH = REPORT_DIR / "split_assignments.json"
REPORT_JSON_PATH = REPORT_DIR / "dataset_validation.json"
REPORT_MD_PATH = REPORT_DIR / "DATASET_VALIDATION.md"

RANDOM_SEED = 42
EMOTIONS = ("angry", "disgust", "fearful", "happy", "neutral", "sad")
RAVDESS_CODES = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
}
CREMAD_CODES = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
PREVIOUSLY_EXAMINED_RAVDESS = {
    "RAVDESS_01",
    "RAVDESS_09",
    "RAVDESS_12",
    "RAVDESS_17",
    "RAVDESS_19",
}

MANIFEST_FIELDS = (
    "dataset",
    "speaker_id",
    "original_filename",
    "audio_path",
    "emotion",
    "sample_rate",
    "duration_seconds",
    "channels",
    "sample_width_bytes",
    "frame_count",
    "audio_sha256",
    "model_eligible",
    "exclusion_reason",
    "data_split",
)


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    duration_seconds: float
    channels: int
    sample_width_bytes: int
    frame_count: int
    audio_sha256: str


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def inspect_wav(path: Path) -> AudioInfo:
    """Read WAV metadata and hash decoded PCM frames without loading a corpus."""
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        if channels < 1 or sample_width < 1 or sample_rate < 1 or frame_count < 1:
            raise ValueError("empty or invalid WAV header")
        while True:
            frames = wav.readframes(65_536)
            if not frames:
                break
            digest.update(frames)
    return AudioInfo(
        sample_rate=sample_rate,
        duration_seconds=frame_count / sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=frame_count,
        audio_sha256=digest.hexdigest(),
    )


def add_audio_fields(row: dict[str, object], path: Path) -> dict[str, object]:
    info = inspect_wav(path)
    return {
        **row,
        "sample_rate": info.sample_rate,
        "duration_seconds": f"{info.duration_seconds:.6f}",
        "channels": info.channels,
        "sample_width_bytes": info.sample_width_bytes,
        "frame_count": info.frame_count,
        "audio_sha256": info.audio_sha256,
        "model_eligible": "true",
        "exclusion_reason": "",
        "data_split": "",
    }


def load_ravdess(issues: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    actor_dirs = sorted(DATASET_DIR.glob("Actor_[0-9][0-9]"))
    for actor_dir in actor_dirs:
        for path in sorted(actor_dir.glob("*.wav")):
            parts = path.stem.split("-")
            if len(parts) != 7:
                issues.append(f"RAVDESS filename has unexpected format: {relative_path(path)}")
                continue
            emotion = RAVDESS_CODES.get(parts[2])
            if emotion is None:
                # Calm and surprised are intentionally excluded from Phase 2.
                if parts[2] not in {"02", "08"}:
                    issues.append(f"RAVDESS has unknown emotion code: {relative_path(path)}")
                continue
            actor = parts[6]
            if actor_dir.name != f"Actor_{actor}":
                issues.append(f"RAVDESS actor folder mismatch: {relative_path(path)}")
                continue
            try:
                rows.append(
                    add_audio_fields(
                        {
                            "dataset": "RAVDESS",
                            "speaker_id": f"RAVDESS_{actor}",
                            "original_filename": path.name,
                            "audio_path": relative_path(path),
                            "emotion": emotion,
                        },
                        path,
                    )
                )
            except (OSError, EOFError, wave.Error, ValueError) as exc:
                issues.append(f"Unreadable WAV {relative_path(path)}: {exc}")
    return rows


def load_cremad(issues: list[str]) -> list[dict[str, object]]:
    metadata_path = CREMAD_DIR / "metadata.csv"
    if not metadata_path.is_file():
        issues.append(f"Missing CREMA-D metadata: {relative_path(metadata_path)}")
        return []

    rows: list[dict[str, object]] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file_path", "source_file", "speaker_id", "emotion", "dataset"}
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            issues.append(f"CREMA-D metadata is missing columns: {missing_columns}")
            return []

        for line_number, source in enumerate(reader, start=2):
            source_file = Path(source["source_file"]).name
            parts = source_file.split("_")
            if len(parts) != 4:
                issues.append(f"CREMA-D filename has unexpected format on line {line_number}: {source_file}")
                continue
            filename_emotion = CREMAD_CODES.get(parts[2])
            emotion = source["emotion"].strip().lower()
            speaker_id = source["speaker_id"].strip()
            if filename_emotion is None or emotion != filename_emotion:
                issues.append(f"CREMA-D label mismatch on line {line_number}: {source_file}")
                continue
            if speaker_id != f"CREMAD_{parts[0]}":
                issues.append(f"CREMA-D speaker mismatch on line {line_number}: {source_file}")
                continue

            path = ROOT / Path(source["file_path"])
            if not path.is_file():
                issues.append(f"CREMA-D metadata references missing file: {source['file_path']}")
                continue
            try:
                rows.append(
                    add_audio_fields(
                        {
                            "dataset": "CREMA-D",
                            "speaker_id": speaker_id,
                            "original_filename": source_file,
                            "audio_path": relative_path(path),
                            "emotion": emotion,
                        },
                        path,
                    )
                )
            except (OSError, EOFError, wave.Error, ValueError) as exc:
                issues.append(f"Unreadable WAV {relative_path(path)}: {exc}")
    return rows


def optimize_dataset_split(
    rows: list[dict[str, object]],
    train_speakers: int,
    validation_speakers: int,
    test_speakers: int,
    rng: random.Random,
    prohibited_test: set[str] | None = None,
) -> dict[str, str]:
    speakers = sorted({str(row["speaker_id"]) for row in rows})
    if len(speakers) != train_speakers + validation_speakers + test_speakers:
        raise ValueError("Split speaker counts do not cover the dataset")
    allowed_test = sorted(set(speakers) - (prohibited_test or set()))
    targets = {
        "train": train_speakers / len(speakers),
        "validation": validation_speakers / len(speakers),
        "test": test_speakers / len(speakers),
    }
    eligible_rows = [row for row in rows if row["model_eligible"] == "true"]
    overall = Counter(str(row["emotion"]) for row in eligible_rows)
    total = len(eligible_rows)
    speaker_classes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in eligible_rows:
        speaker_classes[str(row["speaker_id"])][str(row["emotion"])] += 1

    best_score = float("inf")
    best: dict[str, str] | None = None
    for _ in range(20_000):
        test = set(rng.sample(allowed_test, test_speakers))
        remaining = sorted(set(speakers) - test)
        validation = set(rng.sample(remaining, validation_speakers))
        candidate = {
            speaker: (
                "test" if speaker in test else "validation" if speaker in validation else "train"
            )
            for speaker in speakers
        }
        split_counts: Counter[str] = Counter()
        class_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for speaker, split in candidate.items():
            count = sum(speaker_classes[speaker].values())
            split_counts[split] += count
            class_counts[split].update(speaker_classes[speaker])
        score = 0.0
        for split, target in targets.items():
            score += 3.0 * abs(split_counts[split] / total - target)
            for emotion in EMOTIONS:
                global_rate = overall[emotion] / total
                split_rate = class_counts[split][emotion] / split_counts[split]
                score += abs(split_rate - global_rate)
        if score < best_score:
            best_score = score
            best = candidate
    if best is None:
        raise RuntimeError("Unable to create a dataset split")
    return best


def assign_splits(rows: list[dict[str, object]]) -> dict[str, str]:
    rng = random.Random(RANDOM_SEED)
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)

    assignments = {}
    assignments.update(
        optimize_dataset_split(
            by_dataset["RAVDESS"],
            train_speakers=16,
            validation_speakers=4,
            test_speakers=4,
            rng=rng,
            prohibited_test=PREVIOUSLY_EXAMINED_RAVDESS,
        )
    )
    assignments.update(
        optimize_dataset_split(
            by_dataset["CREMA-D"],
            train_speakers=64,
            validation_speakers=13,
            test_speakers=14,
            rng=rng,
        )
    )
    for row in rows:
        row["data_split"] = assignments[str(row["speaker_id"])]
    return assignments


def nested_counts(rows: Iterable[dict[str, object]], keys: tuple[str, ...]) -> dict[str, object]:
    tree: dict[str, object] = {}
    for row in rows:
        cursor = tree
        for key in keys[:-1]:
            cursor = cursor.setdefault(str(row[key]), {})  # type: ignore[assignment]
        leaf = str(row[keys[-1]])
        cursor[leaf] = int(cursor.get(leaf, 0)) + 1
    return tree


def find_duplicates(rows: list[dict[str, object]], key: str) -> list[list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        values[str(row[key])].append(str(row["audio_path"]))
    return [paths for paths in values.values() if len(paths) > 1]


def mark_duplicate_exclusions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Exclude ambiguous label conflicts and redundant exact copies from modeling."""
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_hash[str(row["audio_sha256"])].append(row)

    excluded: list[dict[str, object]] = []
    for group in by_hash.values():
        if len(group) < 2:
            continue
        emotions = {str(row["emotion"]) for row in group}
        if len(emotions) > 1:
            for row in group:
                row["model_eligible"] = "false"
                row["exclusion_reason"] = "exact_audio_duplicate_with_conflicting_labels"
                excluded.append(row)
        else:
            ordered = sorted(group, key=lambda row: str(row["audio_path"]))
            for row in ordered[1:]:
                row["model_eligible"] = "false"
                row["exclusion_reason"] = "redundant_exact_audio_duplicate"
                excluded.append(row)
    return excluded


def write_manifest(rows: list[dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["dataset"]), str(row["audio_path"]))))


def make_report(
    rows: list[dict[str, object]], assignments: dict[str, str], issues: list[str]
) -> dict[str, object]:
    eligible_rows = [row for row in rows if row["model_eligible"] == "true"]
    excluded_rows = [row for row in rows if row["model_eligible"] != "true"]
    duplicate_paths = find_duplicates(rows, "audio_path")
    duplicate_content = find_duplicates(rows, "audio_sha256")
    content_split_leaks = []
    for paths in duplicate_content:
        matching = [row for row in rows if str(row["audio_path"]) in paths]
        if len({str(row["data_split"]) for row in matching}) > 1:
            content_split_leaks.append(paths)

    speakers_by_split = {
        split: sorted(speaker for speaker, assigned in assignments.items() if assigned == split)
        for split in ("train", "validation", "test")
    }
    overlap = {
        "train_validation": sorted(set(speakers_by_split["train"]) & set(speakers_by_split["validation"])),
        "train_test": sorted(set(speakers_by_split["train"]) & set(speakers_by_split["test"])),
        "validation_test": sorted(set(speakers_by_split["validation"]) & set(speakers_by_split["test"])),
    }

    sample_rates = Counter(int(row["sample_rate"]) for row in rows)
    channels = Counter(int(row["channels"]) for row in rows)
    durations = [float(row["duration_seconds"]) for row in rows]
    validation_errors = list(issues)
    if duplicate_paths:
        validation_errors.append(f"Duplicate manifest audio paths: {len(duplicate_paths)} groups")
    if content_split_leaks:
        validation_errors.append(f"Duplicate audio content crosses splits: {len(content_split_leaks)} groups")
    if any(overlap.values()):
        validation_errors.append("Speaker overlap detected between data splits")
    if set(str(row["emotion"]) for row in rows) != set(EMOTIONS):
        validation_errors.append("Manifest emotion labels do not match the required six classes")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "status": "passed" if not validation_errors else "failed",
        "recordings": len(rows),
        "model_eligible_recordings": len(eligible_rows),
        "excluded_recordings": [
            {
                "audio_path": row["audio_path"],
                "emotion": row["emotion"],
                "reason": row["exclusion_reason"],
            }
            for row in excluded_rows
        ],
        "speakers": len(assignments),
        "recordings_by_dataset": dict(Counter(str(row["dataset"]) for row in rows)),
        "model_eligible_by_dataset": dict(
            Counter(str(row["dataset"]) for row in eligible_rows)
        ),
        "speakers_by_dataset": {
            dataset: len({str(row["speaker_id"]) for row in rows if row["dataset"] == dataset})
            for dataset in sorted({str(row["dataset"]) for row in rows})
        },
        "class_distribution": nested_counts(rows, ("dataset", "emotion")),
        "model_class_distribution": nested_counts(eligible_rows, ("dataset", "emotion")),
        "split_recordings": nested_counts(eligible_rows, ("data_split", "dataset")),
        "split_class_distribution": nested_counts(
            eligible_rows, ("data_split", "dataset", "emotion")
        ),
        "split_speaker_counts": {
            split: len(speakers) for split, speakers in speakers_by_split.items()
        },
        "split_speakers": speakers_by_split,
        "speaker_overlap": overlap,
        "sample_rates": {str(rate): count for rate, count in sorted(sample_rates.items())},
        "channels": {str(count): value for count, value in sorted(channels.items())},
        "duration_seconds": {
            "minimum": min(durations),
            "maximum": max(durations),
            "mean": statistics.fmean(durations),
            "median": statistics.median(durations),
            "total_hours": sum(durations) / 3600,
        },
        "duplicate_manifest_path_groups": duplicate_paths,
        "duplicate_audio_content_groups": duplicate_content,
        "duplicate_content_cross_split_groups": content_split_leaks,
        "validation_errors": validation_errors,
        "notes": [
            "RAVDESS calm and surprised are excluded; no label merging is performed.",
            "Previously examined RAVDESS actors are prohibited from the new final test split.",
            "The test split remains a public-dataset holdout and is not equivalent to natural-speech evaluation.",
            "Original sample rates are recorded here; model feature extraction will resample to 16 kHz mono.",
            "SHA-256 hashes cover decoded PCM frame bytes and identify exact audio-content duplicates.",
            "Conflicting-label duplicate groups are excluded in full; one redundant same-label copy is excluded.",
        ],
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(report: dict[str, object]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    split_recordings = report["split_recordings"]
    split_speaker_counts = report["split_speaker_counts"]
    duration = report["duration_seconds"]
    dataset_rows = [
        [
            dataset,
            report["recordings_by_dataset"][dataset],
            report["model_eligible_by_dataset"][dataset],
            report["speakers_by_dataset"][dataset],
        ]
        for dataset in sorted(report["recordings_by_dataset"])
    ]
    split_rows = [
        [
            split,
            split_recordings.get(split, {}).get("RAVDESS", 0),
            split_recordings.get(split, {}).get("CREMA-D", 0),
            sum(split_recordings.get(split, {}).values()),
            split_speaker_counts[split],
        ]
        for split in ("train", "validation", "test")
    ]
    class_rows = []
    for dataset, counts in report["model_class_distribution"].items():
        class_rows.extend([dataset, emotion, counts.get(emotion, 0)] for emotion in EMOTIONS)

    errors = report["validation_errors"]
    error_text = "None." if not errors else "\n".join(f"- {error}" for error in errors)
    notes = "\n".join(f"- {note}" for note in report["notes"])
    content = f"""# SpeechSense Phase 2 Dataset Validation

Generated: {report['generated_at_utc']}

Status: **{str(report['status']).upper()}**

## Dataset inventory

{markdown_table(['Dataset', 'Discovered recordings', 'Model-eligible recordings', 'Speakers'], dataset_rows)}

Total discovered six-class recordings: **{report['recordings']}**  
Total model-eligible recordings: **{report['model_eligible_recordings']}**  
Total dataset-prefixed speakers: **{report['speakers']}**

## Emotion distribution

{markdown_table(['Dataset', 'Emotion', 'Recordings'], class_rows)}

## Frozen speaker-independent split

{markdown_table(['Split', 'RAVDESS', 'CREMA-D', 'Total recordings', 'Speakers'], split_rows)}

Random seed: `{report['random_seed']}`. Speaker assignments are saved in `split_assignments.json`.
All recordings from a speaker are assigned to exactly one split.

## Audio properties

- Sample rates: `{json.dumps(report['sample_rates'], sort_keys=True)}`
- Channel counts: `{json.dumps(report['channels'], sort_keys=True)}`
- Duration range: {duration['minimum']:.3f}–{duration['maximum']:.3f} seconds
- Mean duration: {duration['mean']:.3f} seconds
- Median duration: {duration['median']:.3f} seconds
- Total duration: {duration['total_hours']:.2f} hours

## Integrity checks

- Duplicate manifest path groups: **{len(report['duplicate_manifest_path_groups'])}**
- Exact duplicate audio-content groups: **{len(report['duplicate_audio_content_groups'])}**
- Recordings excluded because of exact duplicates: **{len(report['excluded_recordings'])}**
- Duplicate-content groups crossing splits: **{len(report['duplicate_content_cross_split_groups'])}**
- Speaker overlap between splits: **{sum(len(value) for value in report['speaker_overlap'].values())}**

Validation errors:

{error_text}

## Method notes and limitations

{notes}
"""
    REPORT_MD_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-content-hash",
        action="store_true",
        help="Reserved for future fast checks; exact content hashing is required for the full report.",
    )
    args = parser.parse_args()
    if args.skip_content_hash:
        parser.error("The full Phase 2 validation requires content hashing.")

    issues: list[str] = []
    print("Inspecting RAVDESS WAV files...", flush=True)
    rows = load_ravdess(issues)
    print(f"Validated {len(rows)} eligible RAVDESS recordings.", flush=True)
    print("Inspecting CREMA-D WAV files...", flush=True)
    cremad_rows = load_cremad(issues)
    rows.extend(cremad_rows)
    print(f"Validated {len(cremad_rows)} CREMA-D recordings.", flush=True)

    if not rows:
        raise RuntimeError("No valid recordings were found")
    mark_duplicate_exclusions(rows)
    assignments = assign_splits(rows)
    report = make_report(rows, assignments, issues)
    write_manifest(rows)
    SPLITS_PATH.write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "strategy": "dataset-stratified, speaker-independent fixed holdout",
                "assignments": dict(sorted(assignments.items())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(report)

    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Split assignments: {SPLITS_PATH}")
    print(f"Validation report: {REPORT_MD_PATH}")
    print(f"Status: {report['status'].upper()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
