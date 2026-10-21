
from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "notebooks" / "advanced_features.csv"

OUTPUT = ROOT / "models" / "final_model"
OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    f"feature_{i}" for i in range(1, 85)
]

SAMPLE_RATE = 16000

EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "filename",
        "actor_id",
        "emotion"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Required metadata columns are missing."
        )

    missing = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames detected.")

    if len(df) != 1440:
        raise ValueError(
            f"Expected 1440 recordings, found {len(df)}."
        )

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    if np.isinf(X).any():
        raise ValueError(
            "Infinite feature values detected."
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["emotion"])

    groups = df["actor_id"].to_numpy()

    if len(np.unique(groups)) != 24:
        raise ValueError(
            "Expected recordings from 24 actors."
        )

    return df, X, y, groups, encoder


# --------------------------------------------------
# 3. Create models
# --------------------------------------------------

def create_svm():

    base_svm = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        )
    )

    # Probability calibration replaces the deprecated
    # SVC(probability=True).
    #
    # This calibration procedure differs from Step 12,
    # so the saved ensemble is a deployment candidate,
    # not an exact reproduction of that experiment.

    return CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=3,
        ensemble=False
    )


def create_hgb():

    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )
    )


# --------------------------------------------------
# 4. Ensemble prediction helper
# --------------------------------------------------

def predict_ensemble_proba(
    svm,
    hgb,
    X
):

    if not np.array_equal(
        svm.classes_,
        hgb.classes_
    ):
        raise ValueError(
            "SVM and HGB class orders do not match."
        )

    svm_proba = svm.predict_proba(X)
    hgb_proba = hgb.predict_proba(X)

    ensemble_proba = (
        0.5 * svm_proba
        + 0.5 * hgb_proba
    )

    return ensemble_proba


# --------------------------------------------------
# 5. Main training
# --------------------------------------------------

def main():

    df, X, y, groups, encoder = load_data()

    print("Dataset loaded successfully.")
    print("Total recordings:", len(df))
    print("Selected features:", X.shape[1])
    print("Emotions:", encoder.classes_.tolist())

    # --------------------------------------------------
    # 6. Recreate development/test split
    # --------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    development_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    development_actors = set(groups[development_idx])
    test_actors = set(groups[test_idx])

    assert development_actors.isdisjoint(test_actors)

    X_train = X[development_idx]
    y_train = y[development_idx]

    print("\n========== TRAINING DATA ==========")
    print("Training recordings:", len(X_train))
    print("Excluded recordings:", len(test_idx))
    print(
        "Training actors:",
        sorted(int(a) for a in development_actors)
    )
    print(
        "Excluded actors:",
        sorted(int(a) for a in test_actors)
    )

    # --------------------------------------------------
    # 7. Train SVM
    # --------------------------------------------------

    print("\n========== TRAINING SVM ==========")

    svm = create_svm()

    svm.fit(X_train, y_train)

    print("Calibrated SVM trained successfully.")

    # --------------------------------------------------
    # 8. Train HistGradientBoosting
    # --------------------------------------------------

    print("\n========== TRAINING HISTGRADIENTBOOSTING ==========")

    hgb = create_hgb()

    hgb.fit(X_train, y_train)

    print("HistGradientBoosting trained successfully.")

    # --------------------------------------------------
    # 9. Verify prediction pipeline
    # --------------------------------------------------

    print("\n========== MODEL VERIFICATION ==========")

    sample_X = X_train[:5]

    probabilities = predict_ensemble_proba(
        svm,
        hgb,
        sample_X
    )

    if probabilities.shape != (
        len(sample_X),
        len(encoder.classes_)
    ):
        raise ValueError(
            "Unexpected probability output shape."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5
    ):
        raise ValueError(
            "Ensemble probabilities do not sum to 1."
        )

    predicted_ids = svm.classes_[
        np.argmax(probabilities, axis=1)
    ]

    predicted_emotions = encoder.inverse_transform(
        predicted_ids
    )

    print("Sample predictions:", predicted_emotions.tolist())
    print("Prediction pipeline verified.")

    # --------------------------------------------------
    # 10. Save trained models
    # --------------------------------------------------

    svm_path = OUTPUT / "svm_model.joblib"
    hgb_path = OUTPUT / "hgb_model.joblib"
    encoder_path = OUTPUT / "label_encoder.joblib"

    joblib.dump(svm, svm_path)
    joblib.dump(hgb, hgb_path)
    joblib.dump(encoder, encoder_path)

    print("\n========== MODELS SAVED ==========")
    print(svm_path)
    print(hgb_path)
    print(encoder_path)

    # --------------------------------------------------
    # 11. Save model configuration
    # --------------------------------------------------

    metadata = {
        "project": "Speech Emotion Recognition",
        "model_name": "SVM_HGB_Soft_Voting",
        "training_date_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "RAVDESS",
        "total_recordings": len(df),
        "training_recordings": len(development_idx),
        "excluded_recordings": len(test_idx),
        "training_actors": sorted(
            int(a) for a in development_actors
        ),
        "excluded_actors": sorted(
            int(a) for a in test_actors
        ),
        "feature_count": 84,
        "feature_columns": FEATURE_COLUMNS,
        "feature_groups": {
            "MFCC_mean": 40,
            "MFCC_std": 40,
            "RMS_mean_std": 2,
            "ZCR_mean_std": 2
        },
        "audio_settings": {
            "sample_rate": SAMPLE_RATE,
            "mono": True,
            "n_mfcc": 40
        },
        "emotion_classes": encoder.classes_.tolist(),
        "emotion_codes": EMOTION_CODES,
        "ensemble_weights": {
            "SVM": 0.5,
            "HistGradientBoosting": 0.5
        },
        "svm_calibration": {
            "method": "sigmoid",
            "cv": 3,
            "ensemble": False
        },
        "evaluation_note": (
            "Deployment candidate trained on development "
            "actors only. Probability calibration differs "
            "from the Step 12 experiment. No independent "
            "evaluation was performed in this script. "
            "Previously excluded actors were inspected "
            "in earlier experiments and should not be "
            "described as a pristine test set."
        )
    }

    metadata_path = OUTPUT / "model_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("Metadata saved:", metadata_path)

    # --------------------------------------------------
    # 12. Verify saved files can be loaded
    # --------------------------------------------------

    print("\n========== VERIFY SAVED MODELS ==========")

    loaded_svm = joblib.load(svm_path)
    loaded_hgb = joblib.load(hgb_path)
    loaded_encoder = joblib.load(encoder_path)

    loaded_proba = predict_ensemble_proba(
        loaded_svm,
        loaded_hgb,
        sample_X
    )

    loaded_ids = loaded_svm.classes_[
        np.argmax(loaded_proba, axis=1)
    ]

    loaded_predictions = loaded_encoder.inverse_transform(
        loaded_ids
    )

    if not np.array_equal(
        predicted_emotions,
        loaded_predictions
    ):
        raise ValueError(
            "Saved model predictions do not match."
        )

    print("Saved models loaded successfully.")
    print("Prediction consistency verified.")

    print("\n========== TRAINING COMPLETE ==========")
    print("Model: SVM + HistGradientBoosting")
    print("Features: MFCC + RMS + ZCR")
    print("Training recordings:", len(X_train))
    print("Saved to: models/final_model/")
    print("Excluded recordings were NOT evaluated.")


if __name__ == "__main__":
    main()
