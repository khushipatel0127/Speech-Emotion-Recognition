"""Train and compare leakage-safe Phase 2 handcrafted-feature baselines.

This stage performs grouped cross-validation on the frozen training speakers,
then evaluates each candidate once on the validation speakers. It deliberately
does not score or fit on the final test split. A separate finalization stage is
required after model selection.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from handcrafted_features import FEATURE_CONFIG, FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "reports" / "phase2" / "dataset_manifest.csv"
FEATURES_PATH = ROOT / "models" / "phase2" / "cache" / "handcrafted_features.csv"
FEATURE_METADATA_PATH = (
    ROOT / "models" / "phase2" / "cache" / "handcrafted_features_metadata.json"
)
OUTPUT_DIR = ROOT / "models" / "phase2" / "handcrafted_baselines"
REPORT_DIR = ROOT / "reports" / "phase2" / "handcrafted_baselines"
RANDOM_SEED = 42

CANDIDATES = (
    "logistic_regression",
    "calibrated_rbf_svm",
    "hist_gradient_boosting",
    "extra_trees",
)


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    if not FEATURES_PATH.is_file() or not FEATURE_METADATA_PATH.is_file():
        raise FileNotFoundError("Run extract_features.py to completion before training.")
    cache_metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    if not cache_metadata.get("complete"):
        raise RuntimeError(
            "The handcrafted feature cache is incomplete: "
            f"{cache_metadata.get('completed_recordings')}/"
            f"{cache_metadata.get('eligible_recordings')}"
        )

    manifest = pd.read_csv(MANIFEST_PATH, dtype={"model_eligible": str})
    manifest = manifest[manifest["model_eligible"].str.lower() == "true"].copy()
    features = pd.read_csv(FEATURES_PATH)
    if features["audio_path"].duplicated().any():
        raise ValueError("Duplicate audio paths exist in the feature cache.")
    frame = manifest.merge(features, on="audio_path", how="left", validate="one_to_one")
    if frame[list(FEATURE_NAMES)].isna().any().any():
        raise ValueError("The feature cache is missing values for eligible manifest rows.")
    if len(frame) != 8_491:
        raise ValueError(f"Expected 8,491 model-eligible rows, found {len(frame)}")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["emotion"])
    matrix = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    groups = frame["speaker_id"].to_numpy()
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Non-finite feature values were found.")
    return frame, matrix, labels, groups, encoder


def build_model(
    name: str, labels: np.ndarray, groups: np.ndarray
) -> object:
    if name == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=RANDOM_SEED,
            ),
        )
    if name == "calibrated_rbf_svm":
        base = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                cache_size=1_024,
            ),
        )
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        inner_splits = list(inner.split(np.zeros(len(labels)), labels, groups))
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=inner_splits,
            ensemble=False,
        )
    if name == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_SEED,
            ),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        )
    raise KeyError(f"Unknown candidate: {name}")


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def evaluate_subsets(
    frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict[str, float]]:
    output = {}
    for dataset in sorted(frame["dataset"].unique()):
        mask = frame["dataset"].to_numpy() == dataset
        output[str(dataset)] = metric_summary(y_true[mask], y_pred[mask])
    return output


def cross_validate_candidate(
    name: str,
    frame: pd.DataFrame,
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
) -> tuple[dict[str, object], np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.full(labels.shape, -1, dtype=int)
    fold_rows = []
    total_fit_seconds = 0.0
    total_inference_seconds = 0.0

    for fold, (fit_indices, score_indices) in enumerate(
        splitter.split(matrix, labels, groups), start=1
    ):
        fit_groups = groups[fit_indices]
        score_groups = groups[score_indices]
        if set(fit_groups) & set(score_groups):
            raise RuntimeError("Speaker leakage detected in a grouped CV fold")
        model = build_model(name, labels[fit_indices], fit_groups)
        started = time.perf_counter()
        model.fit(matrix[fit_indices], labels[fit_indices])
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fold_predictions = model.predict(matrix[score_indices])
        inference_seconds = time.perf_counter() - started
        predictions[score_indices] = fold_predictions
        total_fit_seconds += fit_seconds
        total_inference_seconds += inference_seconds
        fold_rows.append(
            {
                "fold": fold,
                "fit_speakers": len(set(fit_groups)),
                "score_speakers": len(set(score_groups)),
                "fit_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
                **metric_summary(labels[score_indices], fold_predictions),
            }
        )
        print(
            f"  {name} fold {fold}: macro F1="
            f"{fold_rows[-1]['macro_f1']:.4f}, accuracy={fold_rows[-1]['accuracy']:.4f}",
            flush=True,
        )

    if np.any(predictions < 0):
        raise RuntimeError("Grouped CV did not generate every out-of-fold prediction")
    summary: dict[str, object] = {
        **metric_summary(labels, predictions),
        "fit_seconds": total_fit_seconds,
        "inference_seconds": total_inference_seconds,
        "inference_ms_per_recording": 1_000 * total_inference_seconds / len(labels),
        "folds": fold_rows,
        "per_dataset": evaluate_subsets(frame, labels, predictions),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    return summary, predictions


def validation_evaluation(
    name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    train_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    train_groups: np.ndarray,
    classes: list[str],
) -> tuple[object, dict[str, object]]:
    model = build_model(name, train_labels, train_groups)
    started = time.perf_counter()
    model.fit(train_matrix, train_labels)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(validation_matrix)
    probabilities = model.predict_proba(validation_matrix)
    inference_seconds = time.perf_counter() - started
    if probabilities.shape != (len(validation_labels), len(classes)):
        raise ValueError(f"Unexpected probability shape for {name}: {probabilities.shape}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"Probabilities do not sum to one for {name}")

    result: dict[str, object] = {
        **metric_summary(validation_labels, predictions),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_recording": 1_000 * inference_seconds / len(validation_labels),
        "per_dataset": evaluate_subsets(
            validation_frame, validation_labels, predictions
        ),
        "classification_report": classification_report(
            validation_labels,
            predictions,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(validation_labels, predictions).tolist(),
    }
    return model, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATES,
        default=list(CANDIDATES),
    )
    args = parser.parse_args()
    frame, matrix, labels, groups, encoder = load_data()
    classes = [str(value) for value in encoder.classes_]

    train_mask = frame["data_split"].to_numpy() == "train"
    validation_mask = frame["data_split"].to_numpy() == "validation"
    test_mask = frame["data_split"].to_numpy() == "test"
    if set(groups[train_mask]) & set(groups[validation_mask]):
        raise RuntimeError("Train/validation speaker leakage detected")
    if set(groups[train_mask]) & set(groups[test_mask]):
        raise RuntimeError("Train/test speaker leakage detected")

    train_frame = frame.loc[train_mask].reset_index(drop=True)
    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    train_matrix = matrix[train_mask]
    validation_matrix = matrix[validation_mask]
    train_labels = labels[train_mask]
    validation_labels = labels[validation_mask]
    train_groups = groups[train_mask]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    comparison_rows = []

    for name in args.candidates:
        print(f"\nGrouped cross-validation: {name}", flush=True)
        cv_result, _ = cross_validate_candidate(
            name,
            train_frame,
            train_matrix,
            train_labels,
            train_groups,
            classes,
        )
        print(f"Frozen validation evaluation: {name}", flush=True)
        model, validation_result = validation_evaluation(
            name,
            train_frame,
            validation_frame,
            train_matrix,
            validation_matrix,
            train_labels,
            validation_labels,
            train_groups,
            classes,
        )
        model_path = OUTPUT_DIR / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        model_size = model_path.stat().st_size
        results[name] = {
            "cross_validation": cv_result,
            "validation": validation_result,
            "model_path": model_path.relative_to(ROOT).as_posix(),
            "model_size_bytes": model_size,
        }
        comparison_rows.append(
            {
                "model": name,
                "cv_accuracy": cv_result["accuracy"],
                "cv_macro_f1": cv_result["macro_f1"],
                "cv_weighted_f1": cv_result["weighted_f1"],
                "validation_accuracy": validation_result["accuracy"],
                "validation_macro_f1": validation_result["macro_f1"],
                "validation_weighted_f1": validation_result["weighted_f1"],
                "validation_inference_ms_per_recording": validation_result[
                    "inference_ms_per_recording"
                ],
                "model_size_bytes": model_size,
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_macro_f1", "cv_macro_f1"], ascending=False
    )
    winner = str(comparison.iloc[0]["model"])
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    joblib.dump(encoder, OUTPUT_DIR / "label_encoder.joblib", compress=3)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validation_complete_test_untouched",
        "random_seed": RANDOM_SEED,
        "classes": classes,
        "feature_config": FEATURE_CONFIG,
        "training_recordings": int(train_mask.sum()),
        "validation_recordings": int(validation_mask.sum()),
        "reserved_test_recordings": int(test_mask.sum()),
        "training_speakers": int(len(set(groups[train_mask]))),
        "validation_speakers": int(len(set(groups[validation_mask]))),
        "reserved_test_speakers": int(len(set(groups[test_mask]))),
        "selection_rule": "highest frozen-validation macro F1; grouped-CV macro F1 breaks ties",
        "selected_validation_model": winner,
        "results": results,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "test_evaluation": None,
    }
    (REPORT_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "selection.json").write_text(
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
    print(f"\nSelected validation model: {winner}")
    print("The final test split remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
