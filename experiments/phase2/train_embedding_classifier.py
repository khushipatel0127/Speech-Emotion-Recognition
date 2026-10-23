"""Compare lightweight classifiers on cached frozen WavLM embeddings."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from .train_baselines import (
        MANIFEST_PATH,
        RANDOM_SEED,
        ROOT,
        metric_summary,
    )
except ImportError:
    from train_baselines import MANIFEST_PATH, RANDOM_SEED, ROOT, metric_summary


EMBEDDING_DIR = ROOT / "models" / "phase2" / "wavlm_embeddings"
OUTPUT_DIR = ROOT / "models" / "phase2" / "wavlm_classifiers"
REPORT_DIR = ROOT / "reports" / "phase2" / "wavlm_classifiers"


def load_embeddings() -> tuple[list[str], np.ndarray, dict[str, object]]:
    checkpoint = json.loads((EMBEDDING_DIR / "checkpoint.json").read_text(encoding="utf-8"))
    if not checkpoint.get("complete"):
        raise RuntimeError("WavLM embedding extraction is incomplete")
    paths: list[str] = []
    matrices = []
    for shard in sorted(EMBEDDING_DIR.glob("shard_*.npz")):
        data = np.load(shard, allow_pickle=False)
        paths.extend(str(value) for value in data["audio_paths"])
        matrices.append(data["embeddings"].astype(np.float32))
    matrix = np.concatenate(matrices)
    if len(paths) != len(set(paths)) or len(paths) != matrix.shape[0]:
        raise ValueError("Duplicate or mismatched WavLM embedding rows")
    return paths, matrix, checkpoint


def build_embedding_model(name: str, labels: np.ndarray, groups: np.ndarray):
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
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    inner_splits = list(inner.split(np.zeros(len(labels)), labels, groups))
    if name == "linear_svm":
        base = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED),
        )
    elif name == "rbf_svm":
        base = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(C=10.0, kernel="rbf", class_weight="balanced", cache_size=2_048),
        )
    else:
        raise KeyError(name)
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=inner_splits,
        ensemble=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-rbf", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(MANIFEST_PATH, dtype={"model_eligible": str})
    frame = frame[frame["model_eligible"].str.lower() == "true"].reset_index(drop=True)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["emotion"])
    groups = frame["speaker_id"].to_numpy()
    paths, embedding_matrix, checkpoint = load_embeddings()
    index_by_path = {path: index for index, path in enumerate(paths)}
    indices = np.asarray([index_by_path[path] for path in frame["audio_path"]])
    matrix = embedding_matrix[indices]
    classes = [str(value) for value in encoder.classes_]
    train_mask = frame["data_split"].to_numpy() == "train"
    validation_mask = frame["data_split"].to_numpy() == "validation"

    candidates = ["logistic_regression", "linear_svm"]
    if args.include_rbf:
        candidates.append("rbf_svm")
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_indices = np.flatnonzero(train_mask)
    for name in candidates:
        oof_predictions = np.full(train_indices.shape, -1, dtype=int)
        fold_metrics = []
        started = time.perf_counter()
        for fold, (fit_local, score_local) in enumerate(
            outer.split(matrix[train_mask], labels[train_mask], groups[train_mask]), start=1
        ):
            model = build_embedding_model(
                name, labels[train_indices[fit_local]], groups[train_indices[fit_local]]
            )
            model.fit(matrix[train_indices[fit_local]], labels[train_indices[fit_local]])
            predictions = model.predict(matrix[train_indices[score_local]])
            oof_predictions[score_local] = predictions
            fold_metrics.append(
                {"fold": fold, **metric_summary(labels[train_indices[score_local]], predictions)}
            )
        cv_seconds = time.perf_counter() - started

        model = build_embedding_model(name, labels[train_mask], groups[train_mask])
        model.fit(matrix[train_mask], labels[train_mask])
        validation_predictions = model.predict(matrix[validation_mask])
        probabilities = model.predict_proba(matrix[validation_mask])
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"Invalid probabilities from {name}")
        model_path = OUTPUT_DIR / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        results[name] = {
            "cross_validation": {
                **metric_summary(labels[train_mask], oof_predictions),
                "folds": fold_metrics,
                "elapsed_seconds": cv_seconds,
            },
            "validation": {
                **metric_summary(labels[validation_mask], validation_predictions),
                "classification_report": classification_report(
                    labels[validation_mask],
                    validation_predictions,
                    target_names=classes,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    labels[validation_mask], validation_predictions
                ).tolist(),
            },
            "model_path": model_path.relative_to(ROOT).as_posix(),
            "model_size_bytes": model_path.stat().st_size,
        }
        print(
            f"{name}: CV macro F1={results[name]['cross_validation']['macro_f1']:.4f}, "
            f"validation macro F1={results[name]['validation']['macro_f1']:.4f}"
        )

    winner = max(
        results,
        key=lambda name: (
            results[name]["validation"]["macro_f1"],
            results[name]["cross_validation"]["macro_f1"],
        ),
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validation_complete_test_untouched",
        "embedding_model": checkpoint["model_name"],
        "embedding_size": checkpoint["embedding_size"],
        "classes": classes,
        "selected_validation_model": winner,
        "results": results,
        "test_evaluation": None,
    }
    (REPORT_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "selection.json").write_text(
        json.dumps(
            {
                "embedding_model": checkpoint["model_name"],
                "selected_validation_model": winner,
                "model_path": results[winner]["model_path"],
                "test_evaluated": False,
                "deployed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Selected WavLM validation model: {winner}")
    print("The final test split remains untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
