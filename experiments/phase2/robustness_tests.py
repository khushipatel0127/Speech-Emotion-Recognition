"""Evaluate gain sensitivity of the selected handcrafted validation model."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

try:
    from .handcrafted_features import extract_handcrafted_features
    from .train_baselines import OUTPUT_DIR, REPORT_DIR, ROOT, load_data
except ImportError:
    from handcrafted_features import extract_handcrafted_features
    from train_baselines import OUTPUT_DIR, REPORT_DIR, ROOT, load_data


GAINS = (0.5, 1.0, 2.0, 3.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    selection = json.loads((OUTPUT_DIR / "selection.json").read_text(encoding="utf-8"))
    model_name = selection["selected_validation_model"]
    model = joblib.load(ROOT / selection["model_path"])
    frame, matrix, labels, _, encoder = load_data()
    validation_mask = frame["data_split"].to_numpy() == "validation"
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    validation_labels = labels[validation_mask]
    validation_features = matrix[validation_mask]
    class_ids = {name: int(encoder.transform([name])[0]) for name in encoder.classes_}

    baseline_predictions = model.predict(validation_features)
    results = []
    for gain in GAINS:
        started = time.perf_counter()
        if gain == 1.0:
            features = validation_features
        else:
            extracted = []
            for index, relative_path in enumerate(validation_frame["audio_path"], start=1):
                extracted.append(
                    extract_handcrafted_features(ROOT / relative_path, gain=gain)
                )
                if index % args.progress_every == 0:
                    print(f"Gain {gain:.1f}x: {index}/{len(validation_frame)}", flush=True)
            features = np.asarray(extracted, dtype=np.float64)
        predictions = model.predict(features)
        recalls = recall_score(
            validation_labels,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            average=None,
            zero_division=0,
        )
        happy_mask = validation_labels == class_ids["happy"]
        happy_as_angry = int(
            np.sum(predictions[happy_mask] == class_ids["angry"])
        )
        results.append(
            {
                "gain": gain,
                "gain_db": float(20 * np.log10(gain)),
                "accuracy": float(accuracy_score(validation_labels, predictions)),
                "macro_f1": float(
                    f1_score(
                        validation_labels,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "agreement_with_original": float(
                    np.mean(predictions == baseline_predictions)
                ),
                "happy_recall": float(recalls[class_ids["happy"]]),
                "neutral_recall": float(recalls[class_ids["neutral"]]),
                "happy_as_angry_count": happy_as_angry,
                "happy_as_angry_rate": float(happy_as_angry / happy_mask.sum()),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"Gain {gain:.1f}x: accuracy={results[-1]['accuracy']:.4f}, "
            f"macro F1={results[-1]['macro_f1']:.4f}, "
            f"agreement={results[-1]['agreement_with_original']:.4f}"
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "partition": "validation speakers; final test untouched",
        "gain_method": "sample multiplication with clipping to [-1, 1]",
        "results": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "gain_robustness.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
