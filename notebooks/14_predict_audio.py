
from pathlib import Path
import argparse
import json
import sys

import joblib
import librosa
import numpy as np


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "final_model"

SVM_PATH = MODEL_DIR / "svm_model.joblib"
HGB_PATH = MODEL_DIR / "hgb_model.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# --------------------------------------------------
# 2. Load saved models
# --------------------------------------------------

def load_models():

    required_files = [
        SVM_PATH,
        HGB_PATH,
        ENCODER_PATH,
        METADATA_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {path}"
            )

    # Load only model files you created and trust.
    svm = joblib.load(SVM_PATH)
    hgb = joblib.load(HGB_PATH)
    encoder = joblib.load(ENCODER_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    return svm, hgb, encoder, metadata


# --------------------------------------------------
# 3. Extract 84 audio features
# --------------------------------------------------

def extract_features(audio_path, metadata):

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if audio_path.suffix.lower() != ".wav":
        raise ValueError(
            "Please provide a WAV audio file."
        )

    sample_rate = int(
        metadata["audio_settings"]["sample_rate"]
    )

    n_mfcc = int(
        metadata["audio_settings"]["n_mfcc"]
    )

    # Match training:
    # librosa.load(path, sr=16000, mono=True)
    y, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True
    )

    if len(y) == 0:
        raise ValueError(
            "The audio file is empty."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "The audio contains invalid values."
        )

    duration = len(y) / sr

    if duration < 0.5:
        raise ValueError(
            "Audio is too short. Please use at least "
            "0.5 seconds of speech."
        )

    # Reject nearly silent recordings.
    # This is a simple safety check, not a
    # full speech-activity detector.
    rms_amplitude = np.sqrt(
        np.mean(np.square(y, dtype=np.float64))
    )

    if rms_amplitude < 1e-5:
        raise ValueError(
            "The recording is nearly silent. "
            "Please provide audible speech."
        )

    # --------------------------------------------------
    # MFCC: 40 means + 40 standard deviations
    # --------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # --------------------------------------------------
    # RMS: mean + standard deviation
    # --------------------------------------------------

    rms = librosa.feature.rms(y=y)

    rms_mean = np.mean(rms, axis=1)
    rms_std = np.std(rms, axis=1)

    # --------------------------------------------------
    # ZCR: mean + standard deviation
    # --------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(y)

    zcr_mean = np.mean(zcr, axis=1)
    zcr_std = np.std(zcr, axis=1)

    # --------------------------------------------------
    # Combine features in EXACT training order
    # --------------------------------------------------

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        rms_mean,
        rms_std,
        zcr_mean,
        zcr_std
    ]).astype(np.float64)

    expected_count = int(
        metadata["feature_count"]
    )

    if len(features) != expected_count:
        raise ValueError(
            f"Feature mismatch: expected "
            f"{expected_count}, got {len(features)}."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Feature extraction produced invalid values."
        )

    # scikit-learn expects shape:
    # (number_of_recordings, number_of_features)
    features = features.reshape(1, -1)

    return features, duration


# --------------------------------------------------
# 4. Predict emotion
# --------------------------------------------------

def predict_emotion(audio_path):

    svm, hgb, encoder, metadata = load_models()

    features, duration = extract_features(
        audio_path,
        metadata
    )

    # Check model class order.
    if not np.array_equal(
        svm.classes_,
        hgb.classes_
    ):
        raise ValueError(
            "SVM and HGB class orders do not match."
        )

    # SVM + HistGradientBoosting probabilities
    svm_proba = svm.predict_proba(features)
    hgb_proba = hgb.predict_proba(features)

    # 50/50 soft voting
    probabilities = (
        0.5 * svm_proba
        + 0.5 * hgb_proba
    )[0]

    predicted_class_id = svm.classes_[
        np.argmax(probabilities)
    ]

    predicted_emotion = encoder.inverse_transform(
        [predicted_class_id]
    )[0]

    # Create probability breakdown
    class_emotions = encoder.inverse_transform(
        svm.classes_
    )

    emotion_scores = {
        str(emotion): float(probability)
        for emotion, probability in zip(
            class_emotions,
            probabilities
        )
    }

    emotion_scores = dict(
        sorted(
            emotion_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "audio_file": str(Path(audio_path).resolve()),
        "duration_seconds": round(duration, 2),
        "predicted_emotion": str(predicted_emotion),
        "emotion_scores": emotion_scores
    }


# --------------------------------------------------
# 5. Command-line interface
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict speech emotion from a WAV "
            "audio file using the saved ensemble."
        )
    )

    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to a WAV audio file"
    )

    args = parser.parse_args()

    try:
        result = predict_emotion(args.audio_path)

    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    print("\n========== SPEECH EMOTION PREDICTION ==========")

    print("Audio file:", result["audio_file"])
    print(
        "Duration:",
        result["duration_seconds"],
        "seconds"
    )

    print(
        "\nPredicted emotion:",
        result["predicted_emotion"].upper()
    )

    print("\n========== EMOTION SCORES ==========")

    for emotion, score in result[
        "emotion_scores"
    ].items():

        print(
            f"{emotion:<12}: {score * 100:6.2f}%"
        )

    print(
        "\nNote: These are model probability estimates, "
        "not guaranteed confidence or accuracy."
    )

    print("\nPrediction completed successfully.")


if __name__ == "__main__":
    main()
