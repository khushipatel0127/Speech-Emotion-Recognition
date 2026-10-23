"""Evaluate manually labeled natural recordings through the deployed pipeline.

Ground-truth labels must be supplied by the person collecting the recordings.
The script never derives an expected label from a model prediction or filename.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from audio_converter import convert_to_mono_wav  # noqa: E402
from predictor import get_models, predict_from_path  # noqa: E402


REQUIRED_COLUMNS = {
    "speaker_id",
    "expected_emotion",
    "recording_device",
    "recording_format",
    "audio_path",
}
DEFAULT_MANIFEST = ROOT / "reports" / "phase2" / "real_world_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "phase2" / "real_world_evaluation"


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Copy "
            "experiments/phase2/real_world_manifest.template.csv and add "
            "manually labeled recordings."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]


def resolve_audio_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_relative = ROOT / path
    if project_relative.exists():
        return project_relative
    return manifest_path.parent / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    rows = read_manifest(manifest_path)
    if not rows:
        print(
            "No labeled recordings are listed. Real-world accuracy has not been "
            "established."
        )
        return 2

    models = get_models()
    supported = {
        str(value).lower() for value in models["encoder"].classes_.tolist()
    }
    predictions: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="speechsense-real-world-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, row in enumerate(rows, start=1):
            expected = row["expected_emotion"].lower()
            missing_values = sorted(
                column for column in REQUIRED_COLUMNS if not row[column]
            )
            if missing_values:
                errors.append(
                    {
                        "row": str(index),
                        "error": f"Missing required values: {missing_values}",
                    }
                )
                continue
            if expected not in supported:
                errors.append(
                    {
                        "row": str(index),
                        "audio_path": row["audio_path"],
                        "error": (
                            f"Unsupported expected label '{expected}'. "
                            f"Deployed labels: {sorted(supported)}"
                        ),
                    }
                )
                continue

            source = resolve_audio_path(row["audio_path"], manifest_path)
            if not source.is_file():
                errors.append(
                    {
                        "row": str(index),
                        "audio_path": row["audio_path"],
                        "error": "Audio file does not exist",
                    }
                )
                continue

            converted = temp_root / f"{index:04d}.wav"
            started = time.perf_counter()
            try:
                duration, _ = convert_to_mono_wav(source, converted, 180.0)
                result = predict_from_path(converted)
            except Exception as exc:
                errors.append(
                    {
                        "row": str(index),
                        "audio_path": row["audio_path"],
                        "error": str(exc),
                    }
                )
                continue

            predictions.append(
                {
                    **row,
                    "expected_emotion": expected,
                    "predicted_emotion": result["predicted_emotion"],
                    "correct": expected == result["predicted_emotion"],
                    "duration_seconds": duration,
                    "processing_time_ms": round(
                        (time.perf_counter() - started) * 1_000, 2
                    ),
                    "model_version": result["model_version"],
                    "is_uncertain": result["is_uncertain"],
                    "emotion_scores": json.dumps(result["emotion_scores"]),
                }
            )
            print(f"Evaluated {index}/{len(rows)}: {source.name}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    if predictions:
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)

    report: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path.as_posix(),
        "status": "complete" if not errors else "incomplete",
        "recordings_listed": len(rows),
        "recordings_evaluated": len(predictions),
        "speakers_evaluated": len({row["speaker_id"] for row in predictions}),
        "errors": errors,
        "ground_truth_note": "Expected emotions were manually supplied in the manifest.",
    }
    if predictions:
        expected = [str(row["expected_emotion"]) for row in predictions]
        predicted = [str(row["predicted_emotion"]) for row in predictions]
        labels = sorted(supported)
        report["model_version"] = predictions[0]["model_version"]
        report["accuracy"] = float(accuracy_score(expected, predicted))
        report["macro_f1"] = float(
            f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)
        )
        report["classification_report"] = classification_report(
            expected,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report["confusion_matrix_labels"] = labels
        report["confusion_matrix"] = confusion_matrix(
            expected, predicted, labels=labels
        ).tolist()
        report["uncertain_recordings"] = sum(
            bool(row["is_uncertain"]) for row in predictions
        )

    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    if not predictions:
        print("No valid labeled recordings were evaluated.")
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
