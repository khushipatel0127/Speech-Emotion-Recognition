"""Shared 84-feature extraction used by Phase 2 training and inference."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 16_000
N_MFCC = 40
FEATURE_NAMES = (
    tuple(f"mfcc_mean_{index:02d}" for index in range(1, N_MFCC + 1))
    + tuple(f"mfcc_std_{index:02d}" for index in range(1, N_MFCC + 1))
    + ("rms_mean", "rms_std", "zcr_mean", "zcr_std")
)

FEATURE_CONFIG = {
    "sample_rate": SAMPLE_RATE,
    "mono": True,
    "n_mfcc": N_MFCC,
    "silence_handling": "none; preserve the complete recording",
    "loudness_normalization": False,
    "feature_order": list(FEATURE_NAMES),
}


def extract_handcrafted_features(audio_path: str | Path, gain: float = 1.0) -> np.ndarray:
    """Extract MFCC, RMS, and ZCR summary features from one recording."""
    if gain <= 0:
        raise ValueError("gain must be positive")
    audio, sample_rate = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError("audio is empty")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains non-finite samples")
    if gain != 1.0:
        audio = np.clip(audio * gain, -1.0, 1.0)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    rms = librosa.feature.rms(y=audio)
    zcr = librosa.feature.zero_crossing_rate(audio)
    features = np.concatenate(
        (
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            [float(np.mean(rms)), float(np.std(rms))],
            [float(np.mean(zcr)), float(np.std(zcr))],
        )
    ).astype(np.float64)
    if features.shape != (84,) or not np.all(np.isfinite(features)):
        raise ValueError("feature extraction did not produce 84 finite values")
    return features
