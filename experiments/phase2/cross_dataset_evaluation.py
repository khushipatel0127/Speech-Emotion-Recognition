"""Measure validation transfer between CREMA-D and RAVDESS speakers."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .train_baselines import (
        REPORT_DIR,
        build_model,
        load_data,
        metric_summary,
    )
except ImportError:
    from train_baselines import REPORT_DIR, build_model, load_data, metric_summary


MODEL_NAME = "calibrated_rbf_svm"


def main() -> int:
    frame, matrix, labels, groups, encoder = load_data()
    classes = [str(value) for value in encoder.classes_]
    comparisons = (
        ("CREMA-D", "RAVDESS"),
        ("RAVDESS", "CREMA-D"),
    )
    results = {}
    for source, target in comparisons:
        fit_mask = (
            (frame["dataset"].to_numpy() == source)
            & (frame["data_split"].to_numpy() == "train")
        )
        score_mask = (
            (frame["dataset"].to_numpy() == target)
            & (frame["data_split"].to_numpy() == "validation")
        )
        if set(groups[fit_mask]) & set(groups[score_mask]):
            raise RuntimeError("Speaker overlap detected in cross-dataset evaluation")
        model = build_model(MODEL_NAME, labels[fit_mask], groups[fit_mask])
        started = time.perf_counter()
        model.fit(matrix[fit_mask], labels[fit_mask])
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = model.predict(matrix[score_mask])
        inference_seconds = time.perf_counter() - started
        name = f"{source}_to_{target}"
        results[name] = {
            "training_recordings": int(fit_mask.sum()),
            "training_speakers": int(len(set(groups[fit_mask]))),
            "evaluation_recordings": int(score_mask.sum()),
            "evaluation_speakers": int(len(set(groups[score_mask]))),
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            **metric_summary(labels[score_mask], predictions),
            "classification_report": classification_report(
                labels[score_mask],
                predictions,
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                labels[score_mask], predictions
            ).tolist(),
        }
        print(
            f"{name}: accuracy={results[name]['accuracy']:.4f}, "
            f"macro F1={results[name]['macro_f1']:.4f}"
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "evaluation_partition": "validation speakers only; final test untouched",
        "results": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cross_dataset_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
