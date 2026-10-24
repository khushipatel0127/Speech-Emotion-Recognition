"""Freeze a selected embedding classifier and evaluate the final test split once.

Run this only after model family and hyperparameters have been selected using
grouped cross-validation and the validation speakers. The final estimator is
fit on the combined train and validation speakers; test speakers are used only
for the single evaluation written by this script.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from .train_baselines import MANIFEST_PATH, ROOT, evaluate_subsets, metric_summary
    from .train_embedding_classifier import build_embedding_model, load_embeddings
except ImportError:
    from train_baselines import MANIFEST_PATH, ROOT, evaluate_subsets, metric_summary
    from train_embedding_classifier import build_embedding_model, load_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    selection_report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    if selection_report.get("test_evaluation") is not None:
        raise RuntimeError(
            "This selection report already contains a final test evaluation. "
            "Refusing to reuse the holdout split."
        )
    if selection_report.get("selected_validation_model") != "logistic_regression":
        raise ValueError("This finalizer currently supports the selected logistic model only.")
    logistic_c = float(selection_report["logistic_c"])

    frame = pd.read_csv(MANIFEST_PATH, dtype={"model_eligible": str})
    frame = frame[frame["model_eligible"].str.lower() == "true"].reset_index(drop=True)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["emotion"])
    classes = [str(value) for value in encoder.classes_]
    groups = frame["speaker_id"].to_numpy()

    paths, embedding_matrix, checkpoint = load_embeddings(args.embedding_dir)
    if checkpoint["model_name"] != selection_report["embedding_model"]:
        raise ValueError("Embedding checkpoint does not match the selection report.")
    index_by_path = {path: index for index, path in enumerate(paths)}
    try:
        indices = np.asarray([index_by_path[path] for path in frame["audio_path"]])
    except KeyError as exc:
        raise ValueError(f"Embedding cache is missing manifest path: {exc.args[0]}") from exc
    matrix = embedding_matrix[indices]

    split = frame["data_split"].to_numpy()
    development_mask = np.isin(split, ["train", "validation"])
    test_mask = split == "test"
    if not development_mask.any() or not test_mask.any():
        raise ValueError("Manifest must contain development and test recordings.")
    development_speakers = set(frame.loc[development_mask, "speaker_id"])
    test_speakers = set(frame.loc[test_mask, "speaker_id"])
    overlap = development_speakers & test_speakers
    if overlap:
        raise ValueError(f"Speaker leakage into final test split: {sorted(overlap)}")

    model = build_embedding_model(
        "logistic_regression",
        labels[development_mask],
        groups[development_mask],
        logistic_c=logistic_c,
    )
    fit_started = time.perf_counter()
    model.fit(matrix[development_mask], labels[development_mask])
    fit_seconds = time.perf_counter() - fit_started

    inference_started = time.perf_counter()
    predictions = model.predict(matrix[test_mask])
    probabilities = model.predict_proba(matrix[test_mask])
    inference_seconds = time.perf_counter() - inference_started
    if list(model.classes_) != list(range(len(classes))):
        raise ValueError(f"Unexpected probability class ordering: {model.classes_}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Final model produced invalid probabilities.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "final_model.joblib"
    joblib.dump(model, model_path, compress=3)
    joblib.dump(encoder, args.output_dir / "label_encoder.joblib", compress=3)

    test_frame = frame.loc[test_mask].reset_index(drop=True)
    test_metrics = {
        **metric_summary(labels[test_mask], predictions),
        "per_dataset": evaluate_subsets(test_frame, labels[test_mask], predictions),
        "classification_report": classification_report(
            labels[test_mask],
            predictions,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels[test_mask], predictions).tolist(),
        "fit_seconds": fit_seconds,
        "inference_ms_per_recording": float(inference_seconds / test_mask.sum() * 1_000),
        "recordings": int(test_mask.sum()),
        "speakers": len(test_speakers),
    }

    prediction_frame = test_frame[
        ["dataset", "speaker_id", "original_filename", "audio_path", "emotion"]
    ].copy()
    prediction_frame["predicted_emotion"] = encoder.inverse_transform(predictions)
    for class_index, class_name in enumerate(classes):
        prediction_frame[f"score_{class_name}"] = probabilities[:, class_index]
    prediction_frame.to_csv(args.report_dir / "final_test_predictions.csv", index=False)

    selection_report["status"] = "final_test_evaluated"
    selection_report["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    selection_report["development_fit_splits"] = ["train", "validation"]
    selection_report["final_model_path"] = model_path.relative_to(ROOT).as_posix()
    selection_report["final_model_size_bytes"] = model_path.stat().st_size
    selection_report["test_evaluation"] = test_metrics
    args.selection_report.write_text(
        json.dumps(selection_report, indent=2), encoding="utf-8"
    )
    (args.report_dir / "final_test_evaluation.json").write_text(
        json.dumps(
            {
                "generated_at_utc": selection_report["finalized_at_utc"],
                "embedding_model": checkpoint["model_name"],
                "embedding_pooling": checkpoint.get("pooling", "mean"),
                "classifier": "logistic_regression",
                "logistic_c": logistic_c,
                "classes": classes,
                "development_fit_splits": ["train", "validation"],
                "test_evaluation": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "embedding_model": checkpoint["model_name"],
                "embedding_pooling": checkpoint.get("pooling", "mean"),
                "selected_validation_model": "logistic_regression",
                "logistic_c": logistic_c,
                "model_path": model_path.relative_to(ROOT).as_posix(),
                "classes": classes,
                "test_evaluated": True,
                "deployed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Final test: accuracy={test_metrics['accuracy']:.4f}, "
        f"macro F1={test_metrics['macro_f1']:.4f}, "
        f"weighted F1={test_metrics['weighted_f1']:.4f}"
    )
    print(f"Saved final model to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
