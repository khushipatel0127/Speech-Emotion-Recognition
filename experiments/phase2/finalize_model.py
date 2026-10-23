"""Select across completed approaches and evaluate the reserved test exactly once.

This command refuses to run until both handcrafted and WavLM validation reports
exist. It also refuses to overwrite an existing final-test report. Move or archive
the versioned output deliberately before any scientifically justified new run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

try:
    from .train_baselines import (
        MANIFEST_PATH,
        RANDOM_SEED,
        ROOT,
        build_model,
        evaluate_subsets,
        load_data,
        metric_summary,
    )
    from .train_embedding_classifier import build_embedding_model, load_embeddings
except ImportError:
    from train_baselines import (
        MANIFEST_PATH,
        RANDOM_SEED,
        ROOT,
        build_model,
        evaluate_subsets,
        load_data,
        metric_summary,
    )
    from train_embedding_classifier import build_embedding_model, load_embeddings


HANDCRAFTED_REPORT = (
    ROOT / "reports" / "phase2" / "handcrafted_baselines" / "evaluation.json"
)
WAVLM_REPORT = ROOT / "reports" / "phase2" / "wavlm_classifiers" / "evaluation.json"
FINAL_REPORT = ROOT / "reports" / "phase2" / "final_test_evaluation.json"
FINAL_DIR = ROOT / "models" / "phase2" / "v1"
MODEL_VERSION = "speechsense-phase2-v1"


def selected_score(report: dict[str, object]) -> tuple[str, float]:
    name = str(report["selected_validation_model"])
    score = float(report["results"][name]["validation"]["macro_f1"])
    return name, score


def main() -> int:
    if FINAL_REPORT.exists() or (FINAL_DIR / "model_metadata.json").exists():
        raise RuntimeError(
            "A Phase 2 final-test result already exists. Refusing to evaluate the "
            "reserved test again."
        )
    if not HANDCRAFTED_REPORT.is_file() or not WAVLM_REPORT.is_file():
        raise RuntimeError(
            "Both handcrafted and WavLM validation reports are required before "
            "final model selection."
        )

    handcrafted = json.loads(HANDCRAFTED_REPORT.read_text(encoding="utf-8"))
    wavlm = json.loads(WAVLM_REPORT.read_text(encoding="utf-8"))
    handcrafted_name, handcrafted_score = selected_score(handcrafted)
    wavlm_name, wavlm_score = selected_score(wavlm)
    approach = "wavlm" if wavlm_score > handcrafted_score else "handcrafted"
    selected_name = wavlm_name if approach == "wavlm" else handcrafted_name

    if approach == "handcrafted":
        frame, matrix, labels, groups, encoder = load_data()
        model_builder = build_model
        extractor = {
            "type": "handcrafted_84",
            "sample_rate": 16_000,
            "module": "experiments.phase2.handcrafted_features",
        }
    else:
        frame = pd.read_csv(MANIFEST_PATH, dtype={"model_eligible": str})
        frame = frame[frame["model_eligible"].str.lower() == "true"].reset_index(drop=True)
        encoder = LabelEncoder()
        labels = encoder.fit_transform(frame["emotion"])
        groups = frame["speaker_id"].to_numpy()
        paths, raw_embeddings, checkpoint = load_embeddings()
        index_by_path = {path: index for index, path in enumerate(paths)}
        matrix = raw_embeddings[
            np.asarray([index_by_path[path] for path in frame["audio_path"]])
        ]
        model_builder = build_embedding_model
        extractor = {
            "type": "wavlm_mean_pool",
            "sample_rate": 16_000,
            "model_name": checkpoint["model_name"],
            "embedding_size": checkpoint["embedding_size"],
        }

    development_mask = frame["data_split"].isin(["train", "validation"]).to_numpy()
    test_mask = (frame["data_split"] == "test").to_numpy()
    if set(groups[development_mask]) & set(groups[test_mask]):
        raise RuntimeError("Speaker leakage detected before final test evaluation")

    model = model_builder(
        selected_name, labels[development_mask], groups[development_mask]
    )
    fit_started = time.perf_counter()
    model.fit(matrix[development_mask], labels[development_mask])
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    predictions = model.predict(matrix[test_mask])
    probabilities = model.predict_proba(matrix[test_mask])
    inference_seconds = time.perf_counter() - inference_started
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Final model probabilities do not sum to one")

    classes = [str(value) for value in encoder.classes_]
    final_metrics = {
        **metric_summary(labels[test_mask], predictions),
        "per_dataset": evaluate_subsets(
            frame.loc[test_mask].reset_index(drop=True),
            labels[test_mask],
            predictions,
        ),
        "classification_report": classification_report(
            labels[test_mask],
            predictions,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels[test_mask], predictions).tolist(),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_recording": 1_000 * inference_seconds / int(test_mask.sum()),
    }

    FINAL_DIR.mkdir(parents=True, exist_ok=False)
    model_path = FINAL_DIR / "model.joblib"
    encoder_path = FINAL_DIR / "label_encoder.joblib"
    joblib.dump(model, model_path, compress=3)
    joblib.dump(encoder, encoder_path, compress=3)
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approach": approach,
        "classifier": selected_name,
        "classes": classes,
        "extractor": extractor,
        "random_seed": RANDOM_SEED,
        "development_recordings": int(development_mask.sum()),
        "development_speakers": len(set(groups[development_mask])),
        "test_recordings": int(test_mask.sum()),
        "test_speakers": len(set(groups[test_mask])),
        "validation_selection": {
            "handcrafted_model": handcrafted_name,
            "handcrafted_macro_f1": handcrafted_score,
            "wavlm_model": wavlm_name,
            "wavlm_macro_f1": wavlm_score,
        },
        "final_test_metrics": final_metrics,
        "model_size_bytes": model_path.stat().st_size,
        "deployment_status": "not_deployed",
        "rollback_model": "models/final_model",
    }
    (FINAL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Final test complete for {approach}/{selected_name}: "
        f"accuracy={final_metrics['accuracy']:.4f}, "
        f"macro F1={final_metrics['macro_f1']:.4f}"
    )
    print("The final model is saved but not deployed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
