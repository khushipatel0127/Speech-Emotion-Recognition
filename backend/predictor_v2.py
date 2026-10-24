"""Versioned inference for the historical and Phase 2 SpeechSense models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import librosa
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_PATH = ROOT / "models" / "deployment.json"
HISTORICAL_MODEL_DIR = ROOT / "models" / "final_model"

_loaded_models = None


def _load_deployment() -> dict[str, object]:
    if not DEPLOYMENT_PATH.is_file():
        return {"active_model": "historical"}
    deployment = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    if deployment.get("active_model") not in {"historical", "phase2_wavlm"}:
        raise ValueError("models/deployment.json has an unsupported active_model")
    return deployment


def _load_historical_models(deployment: dict[str, object]) -> dict[str, object]:
    paths = {
        "svm": HISTORICAL_MODEL_DIR / "svm_model.joblib",
        "hgb": HISTORICAL_MODEL_DIR / "hgb_model.joblib",
        "encoder": HISTORICAL_MODEL_DIR / "label_encoder.joblib",
        "metadata": HISTORICAL_MODEL_DIR / "model_metadata.json",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return {
        "kind": "historical",
        "svm": joblib.load(paths["svm"]),
        "hgb": joblib.load(paths["hgb"]),
        "encoder": joblib.load(paths["encoder"]),
        "metadata": metadata,
        "deployment": deployment,
    }


def _load_wavlm_models(deployment: dict[str, object]) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel
    except ImportError as exc:
        raise RuntimeError(
            "The Phase 2 model requires PyTorch and Transformers. Install "
            "backend/requirements-wavlm.txt, then restart the backend."
        ) from exc

    config = deployment.get("phase2_wavlm")
    if not isinstance(config, dict):
        raise ValueError("Deployment configuration is missing phase2_wavlm settings")
    model_path = ROOT / str(config["classifier_path"])
    selection_path = ROOT / str(config["selection_path"])
    for path in (model_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 2 model file not found: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("test_evaluated"):
        raise RuntimeError("Refusing to load a Phase 2 artifact without final-test metadata")

    backbone_name = str(config.get("backbone", selection["embedding_model"]))
    device_request = str(config.get("device", "cpu"))
    if device_request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA deployment was requested but CUDA is unavailable")
    device = "cuda" if device_request == "cuda" else "cpu"
    local_files_only = bool(config.get("local_files_only", True))
    processor = AutoFeatureExtractor.from_pretrained(
        backbone_name, local_files_only=local_files_only
    )
    backbone = WavLMModel.from_pretrained(
        backbone_name, local_files_only=local_files_only
    ).to(device)
    backbone.eval()
    with np.load(model_path, allow_pickle=False) as artifact:
        classifier = {name: artifact[name] for name in artifact.files}
    expected_classes = [str(value) for value in config["classes"]]
    required_arrays = {
        "imputer_statistics",
        "scaler_mean",
        "scaler_scale",
        "coef",
        "intercept",
        "classes",
    }
    if set(classifier) != required_arrays:
        raise ValueError("Phase 2 portable classifier has an invalid array set")
    feature_count = int(classifier["coef"].shape[1])
    if feature_count != int(backbone.config.hidden_size):
        raise ValueError("Phase 2 classifier and WavLM embedding dimensions do not match")
    if any(
        classifier[name].shape != (feature_count,)
        for name in ("imputer_statistics", "scaler_mean", "scaler_scale")
    ):
        raise ValueError("Phase 2 portable preprocessing dimensions are invalid")
    if classifier["coef"].shape[0] != len(expected_classes):
        raise ValueError("Phase 2 classifier output count does not match configured classes")
    if classifier["intercept"].shape != (len(expected_classes),):
        raise ValueError("Phase 2 classifier intercept dimensions are invalid")
    if classifier["classes"].tolist() != list(range(len(expected_classes))):
        raise ValueError("Phase 2 classifier probability order is invalid")
    if not all(np.all(np.isfinite(value)) for value in classifier.values()):
        raise ValueError("Phase 2 portable classifier contains invalid values")
    if np.any(classifier["scaler_scale"] == 0):
        raise ValueError("Phase 2 portable classifier contains a zero scaling factor")
    return {
        "kind": "phase2_wavlm",
        "classifier": classifier,
        "processor": processor,
        "backbone": backbone,
        "torch": torch,
        "device": device,
        "metadata": {
            "model_version": str(config["model_version"]),
            "backbone": backbone_name,
            "classes": expected_classes,
            "sample_rate": 16_000,
            "pooling": str(config.get("pooling", "mean")),
            "max_duration_seconds": float(config.get("max_duration_seconds", 20.0)),
            "local_files_only": local_files_only,
        },
        "deployment": deployment,
    }


def get_models():
    """Load the configured model once and cache it in memory."""
    global _loaded_models
    if _loaded_models is not None:
        return _loaded_models
    deployment = _load_deployment()
    if deployment.get("active_model") == "phase2_wavlm":
        _loaded_models = _load_wavlm_models(deployment)
    else:
        _loaded_models = _load_historical_models(deployment)
    return _loaded_models


def get_model_info() -> dict[str, object]:
    models = get_models()
    if models["kind"] == "phase2_wavlm":
        metadata = models["metadata"]
        return {
            "kind": models["kind"],
            "model_version": metadata["model_version"],
            "classes": metadata["classes"],
            "backbone": metadata["backbone"],
            "device": models["device"],
        }
    encoder = models["encoder"]
    metadata = models["metadata"]
    return {
        "kind": models["kind"],
        "model_version": metadata.get("model_version", "historical-ravdess-8class-v1"),
        "classes": [str(value) for value in encoder.classes_],
        "backbone": None,
        "device": "cpu",
    }


def _load_valid_audio(audio_path: Path, sample_rate: int) -> tuple[np.ndarray, float]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("Please provide a WAV audio file.")
    audio, actual_sample_rate = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError("The audio file is empty.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("The audio contains invalid values.")
    duration = len(audio) / actual_sample_rate
    if duration < 0.5:
        raise ValueError("Audio is too short. Please provide at least 0.5 seconds of speech.")
    rms_amplitude = np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
    if rms_amplitude < 1e-5:
        raise ValueError("The recording is nearly silent. Please provide audible speech.")
    return audio, duration


def extract_features(audio_path, metadata):
    """Extract the historical 84-feature vector."""
    sample_rate = int(metadata["audio_settings"]["sample_rate"])
    n_mfcc = int(metadata["audio_settings"]["n_mfcc"])
    audio, duration = _load_valid_audio(Path(audio_path), sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(audio)
    features = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(rms, axis=1),
            np.std(rms, axis=1),
            np.mean(zcr, axis=1),
            np.std(zcr, axis=1),
        ]
    ).astype(np.float64)
    expected_count = int(metadata["feature_count"])
    if len(features) != expected_count:
        raise ValueError(f"Feature mismatch: expected {expected_count}, got {len(features)}.")
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature extraction produced invalid values.")
    return features.reshape(1, -1), duration


def _uncertainty(scores: dict[str, float]) -> tuple[float, float, bool]:
    ranked_scores = sorted(scores.values(), reverse=True)
    top_score = ranked_scores[0]
    score_margin = top_score - ranked_scores[1] if len(ranked_scores) > 1 else top_score
    return top_score, score_margin, top_score < 0.50 or score_margin < 0.10


def _format_prediction(
    duration: float,
    predicted_emotion: str,
    emotion_scores: dict[str, float],
    model_version: str,
) -> dict[str, object]:
    ordered_scores = dict(
        sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
    )
    top_score, score_margin, is_uncertain = _uncertainty(ordered_scores)
    return {
        "duration_seconds": round(duration, 2),
        "predicted_emotion": predicted_emotion,
        "emotion_scores": ordered_scores,
        "model_version": model_version,
        "top_score": float(top_score),
        "score_margin": float(score_margin),
        "is_uncertain": bool(is_uncertain),
        "uncertainty_note": (
            "The top model score is low or close to the next score. Interpret this result cautiously."
            if is_uncertain
            else None
        ),
    }


def _historical_prediction(audio_path: Path, models: dict[str, object]) -> dict[str, object]:
    svm = models["svm"]
    hgb = models["hgb"]
    encoder = models["encoder"]
    metadata = models["metadata"]
    features, duration = extract_features(audio_path, metadata)
    if not np.array_equal(svm.classes_, hgb.classes_):
        raise ValueError("SVM and HGB class orders do not match.")
    probabilities = (0.5 * svm.predict_proba(features) + 0.5 * hgb.predict_proba(features))[0]
    class_ids = svm.classes_
    predicted_class_id = class_ids[np.argmax(probabilities)]
    emotion_scores = {
        str(emotion): float(probability)
        for emotion, probability in zip(encoder.inverse_transform(class_ids), probabilities)
    }
    return _format_prediction(
        duration,
        str(encoder.inverse_transform([predicted_class_id])[0]),
        emotion_scores,
        str(metadata.get("model_version", "historical-ravdess-8class-v1")),
    )


def _wavlm_prediction(audio_path: Path, models: dict[str, object]) -> dict[str, object]:
    metadata = models["metadata"]
    audio, duration = _load_valid_audio(audio_path, int(metadata["sample_rate"]))
    if duration > metadata["max_duration_seconds"]:
        raise ValueError(
            f"Audio is too long for this model. Use a speech clip of at most "
            f"{metadata['max_duration_seconds']:.0f} seconds."
        )
    processor = models["processor"]
    backbone = models["backbone"]
    torch = models["torch"]
    inputs = processor(
        audio,
        sampling_rate=int(metadata["sample_rate"]),
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(models["device"])
    attention_mask = inputs.attention_mask.to(models["device"])
    with torch.inference_mode():
        hidden = backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        ).last_hidden_state
        feature_mask = backbone._get_feature_vector_attention_mask(
            hidden.shape[1], attention_mask
        ).unsqueeze(-1)
        embedding = (hidden * feature_mask).sum(dim=1) / feature_mask.sum(dim=1).clamp(min=1)
    matrix = embedding.detach().cpu().numpy().astype(np.float32)
    classifier = models["classifier"]
    imputed = np.where(
        np.isnan(matrix), classifier["imputer_statistics"], matrix
    ).astype(np.float64, copy=False)
    standardized = (imputed - classifier["scaler_mean"]) / classifier["scaler_scale"]
    logits = standardized @ classifier["coef"].T + classifier["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    probabilities = (exponentials / exponentials.sum(axis=1, keepdims=True))[0]
    class_ids = classifier["classes"].astype(int, copy=False)
    predicted_class_id = int(class_ids[int(np.argmax(probabilities))])
    classes = metadata["classes"]
    emotion_scores = {
        str(classes[int(class_id)]): float(probability)
        for class_id, probability in zip(class_ids, probabilities)
    }
    return _format_prediction(
        duration,
        str(classes[predicted_class_id]),
        emotion_scores,
        str(metadata["model_version"]),
    )


def predict_from_path(audio_path):
    models = get_models()
    path = Path(audio_path)
    if models["kind"] == "phase2_wavlm":
        return _wavlm_prediction(path, models)
    return _historical_prediction(path, models)
