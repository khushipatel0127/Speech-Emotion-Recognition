"""Compare training-only volume augmentation for the strongest baselines."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .handcrafted_features import FEATURE_NAMES
    from .train_baselines import (
        OUTPUT_DIR,
        RANDOM_SEED,
        REPORT_DIR,
        build_model,
        evaluate_subsets,
        load_data,
        metric_summary,
    )
except ImportError:
    from handcrafted_features import FEATURE_NAMES
    from train_baselines import (
        OUTPUT_DIR,
        RANDOM_SEED,
        REPORT_DIR,
        build_model,
        evaluate_subsets,
        load_data,
        metric_summary,
    )


AUGMENTED_PATH = (
    OUTPUT_DIR.parent / "cache" / "volume_augmented_features.csv"
)
AUGMENTED_METADATA = (
    OUTPUT_DIR.parent / "cache" / "volume_augmented_features_metadata.json"
)
CANDIDATES = ("calibrated_rbf_svm", "hist_gradient_boosting")


def main() -> int:
    metadata = json.loads(AUGMENTED_METADATA.read_text(encoding="utf-8"))
    if not metadata.get("complete"):
        raise RuntimeError("Volume-augmented feature cache is incomplete")
    frame, matrix, labels, groups, encoder = load_data()
    train_mask = frame["data_split"].to_numpy() == "train"
    validation_mask = frame["data_split"].to_numpy() == "validation"
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    train_matrix = matrix[train_mask]
    train_labels = labels[train_mask]
    train_groups = groups[train_mask]

    augmented = pd.read_csv(AUGMENTED_PATH).set_index("audio_path")
    augmented = augmented.loc[train_frame["audio_path"]]
    augmented_matrix = augmented[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(augmented_matrix)):
        raise ValueError("Augmented feature cache contains invalid values")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    augmented_output = OUTPUT_DIR.parent / "volume_augmented"
    augmented_output.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATES:
        oof = np.full(train_labels.shape, -1, dtype=int)
        fold_results = []
        for fold, (fit_indices, score_indices) in enumerate(
            outer.split(train_matrix, train_labels, train_groups), start=1
        ):
            fit_matrix = np.concatenate(
                [train_matrix[fit_indices], augmented_matrix[fit_indices]]
            )
            fit_labels = np.concatenate([train_labels[fit_indices], train_labels[fit_indices]])
            fit_groups = np.concatenate([train_groups[fit_indices], train_groups[fit_indices]])
            model = build_model(name, fit_labels, fit_groups)
            model.fit(fit_matrix, fit_labels)
            predictions = model.predict(train_matrix[score_indices])
            oof[score_indices] = predictions
            fold_results.append(
                {"fold": fold, **metric_summary(train_labels[score_indices], predictions)}
            )
            print(
                f"{name} fold {fold}: macro F1={fold_results[-1]['macro_f1']:.4f}",
                flush=True,
            )

        full_matrix = np.concatenate([train_matrix, augmented_matrix])
        full_labels = np.concatenate([train_labels, train_labels])
        full_groups = np.concatenate([train_groups, train_groups])
        model = build_model(name, full_labels, full_groups)
        started = time.perf_counter()
        model.fit(full_matrix, full_labels)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = model.predict(matrix[validation_mask])
        inference_seconds = time.perf_counter() - started
        model_path = augmented_output / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        results[name] = {
            "cross_validation": {
                **metric_summary(train_labels, oof),
                "folds": fold_results,
            },
            "validation": {
                **metric_summary(labels[validation_mask], predictions),
                "per_dataset": evaluate_subsets(
                    validation_frame, labels[validation_mask], predictions
                ),
                "classification_report": classification_report(
                    labels[validation_mask],
                    predictions,
                    target_names=[str(value) for value in encoder.classes_],
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    labels[validation_mask], predictions
                ).tolist(),
            },
            "fit_seconds": fit_seconds,
            "validation_inference_ms_per_recording": float(
                inference_seconds / validation_mask.sum() * 1_000
            ),
            "model_path": model_path.relative_to(OUTPUT_DIR.parents[2]).as_posix(),
            "model_size_bytes": model_path.stat().st_size,
        }

    winner = max(
        results,
        key=lambda name: (
            results[name]["validation"]["macro_f1"],
            results[name]["cross_validation"]["macro_f1"],
        ),
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "training-only augmentation; validation original; final test untouched",
        "augmentation": metadata,
        "selected_validation_model": winner,
        "results": results,
    }
    (REPORT_DIR / "volume_augmented_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (augmented_output / "selection.json").write_text(
        json.dumps(
            {
                "selected_validation_model": winner,
                "model_path": results[winner]["model_path"],
                "test_evaluated": False,
                "deployed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Selected augmented validation model: {winner}")
    print("The final test split remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
