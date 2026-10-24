from pathlib import Path
import json
import joblib
import librosa
import numpy as np

# Root path: project root is parent of backend/
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "final_model"

SVM_PATH = MODEL_DIR / "svm_model.joblib"
HGB_PATH = MODEL_DIR / "hgb_model.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

_loaded_models = None

def get_models():
    """Load models once and cache in memory."""
    global _loaded_models
    if _loaded_models is not None:
        return _loaded_models

    required_files = [SVM_PATH, HGB_PATH, ENCODER_PATH, METADATA_PATH]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")

    svm = joblib.load(SVM_PATH)
    hgb = joblib.load(HGB_PATH)
    encoder = joblib.load(ENCODER_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    _loaded_models = {
        "svm": svm,
        "hgb": hgb,
        "encoder": encoder,
        "metadata": metadata,
    }
    return _loaded_models

def extract_features(audio_path, metadata):
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() != ".wav":
        raise ValueError("Please provide a WAV audio file.")

    sample_rate = int(metadata["audio_settings"]["sample_rate"])
    n_mfcc = int(metadata["audio_settings"]["n_mfcc"])

    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    if len(y) == 0:
        raise ValueError("The audio file is empty.")

    if not np.all(np.isfinite(y)):
        raise ValueError("The audio contains invalid values.")

    duration = len(y) / sr

    if duration < 0.5:
        raise ValueError("Audio is too short. Please provide at least 0.5 seconds of speech.")

    rms_amplitude = np.sqrt(np.mean(np.square(y, dtype=np.float64)))
    if rms_amplitude < 1e-5:
        raise ValueError("The recording is nearly silent. Please provide audible speech.")

    # MFCC: 40 means + 40 standard deviations
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # RMS: mean + standard deviation
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms, axis=1)
    rms_std = np.std(rms, axis=1)

    # ZCR: mean + standard deviation
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr, axis=1)
    zcr_std = np.std(zcr, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        rms_mean,
        rms_std,
        zcr_mean,
        zcr_std,
    ]).astype(np.float64)

    expected_count = int(metadata["feature_count"])
    if len(features) != expected_count:
        raise ValueError(f"Feature mismatch: expected {expected_count}, got {len(features)}.")

    if not np.all(np.isfinite(features)):
        raise ValueError("Feature extraction produced invalid values.")

    features = features.reshape(1, -1)
    return features, duration

def predict_from_path(audio_path):
    models = get_models()
    svm = models["svm"]
    hgb = models["hgb"]
    encoder = models["encoder"]
    metadata = models["metadata"]

    features, duration = extract_features(audio_path, metadata)

    if not np.array_equal(svm.classes_, hgb.classes_):
        raise ValueError("SVM and HGB class orders do not match.")

    svm_proba = svm.predict_proba(features)
    hgb_proba = hgb.predict_proba(features)

    probabilities = (0.5 * svm_proba + 0.5 * hgb_proba)[0]

    predicted_class_id = svm.classes_[np.argmax(probabilities)]
    predicted_emotion = encoder.inverse_transform([predicted_class_id])[0]

    class_emotions = encoder.inverse_transform(svm.classes_)
    emotion_scores = {
        str(emotion): float(probability)
        for emotion, probability in zip(class_emotions, probabilities)
    }

    emotion_scores = dict(
        sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
    )
    ranked_scores = list(emotion_scores.values())
    top_score = ranked_scores[0]
    score_margin = top_score - ranked_scores[1] if len(ranked_scores) > 1 else top_score
    is_uncertain = top_score < 0.50 or score_margin < 0.10
    model_version = metadata.get("model_version", "historical-ravdess-8class-v1")

    return {
        "duration_seconds": round(duration, 2),
        "predicted_emotion": str(predicted_emotion),
        "emotion_scores": emotion_scores,
        "model_version": str(model_version),
        "top_score": float(top_score),
        "score_margin": float(score_margin),
        "is_uncertain": bool(is_uncertain),
        "uncertainty_note": (
            "The top model score is low or close to the next score. Interpret this result cautiously."
            if is_uncertain
            else None
        ),
    }


# The versioned implementation keeps the historical artifacts as a rollback
# option and adds the validated six-class WavLM deployment path.
from predictor_v2 import extract_features, get_model_info, get_models, predict_from_path  # noqa: E402,F401
