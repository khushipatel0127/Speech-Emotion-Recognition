"""Export a fitted sklearn embedding pipeline as a portable NumPy classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    imputer = pipeline.named_steps["simpleimputer"]
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    arrays = {
        "imputer_statistics": np.asarray(imputer.statistics_, dtype=np.float64),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "coef": np.asarray(classifier.coef_, dtype=np.float64),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float64),
        "classes": np.asarray(classifier.classes_, dtype=np.int64),
    }
    feature_count = arrays["coef"].shape[1]
    if any(arrays[name].shape != (feature_count,) for name in (
        "imputer_statistics", "scaler_mean", "scaler_scale"
    )):
        raise ValueError("Pipeline preprocessing dimensions do not match coefficients")

    rng = np.random.default_rng(42)
    sample = rng.normal(size=(8, feature_count)).astype(np.float32)
    sklearn_probabilities = pipeline.predict_proba(sample)
    imputed = np.where(np.isnan(sample), arrays["imputer_statistics"], sample)
    standardized = (imputed - arrays["scaler_mean"]) / arrays["scaler_scale"]
    portable_probabilities = softmax(
        standardized @ arrays["coef"].T + arrays["intercept"]
    )
    max_error = float(np.max(np.abs(sklearn_probabilities - portable_probabilities)))
    if max_error > 1e-6:
        raise ValueError(f"Portable probability verification failed: {max_error}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "format": "speechsense_numpy_logistic_v1",
                "source_model": args.model.name,
                "feature_count": feature_count,
                "class_count": int(arrays["classes"].size),
                "verification_max_probability_error": max_error,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {args.output} ({args.output.stat().st_size} bytes)")
    print(f"Maximum probability difference: {max_error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
