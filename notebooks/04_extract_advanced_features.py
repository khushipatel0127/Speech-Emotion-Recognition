
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

DATASET_DIR = Path("dataset")
OUTPUT_FILE = Path("notebooks/advanced_features.csv")

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def mean_std(values):
    """Return mean and standard deviation as a fixed-size vector."""
    return np.concatenate([
        np.mean(values, axis=1),
        np.std(values, axis=1),
    ])


def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = mean_std(mfcc)

    # RMS energy
    rms = librosa.feature.rms(y=audio)
    rms_features = mean_std(rms)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_features = mean_std(zcr)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_features = mean_std(contrast)

    # Pitch estimate using probabilistic YIN
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    voiced_pitch = f0[np.isfinite(f0)]
    if len(voiced_pitch) == 0:
        pitch_features = np.array([0.0, 0.0])
    else:
        pitch_features = np.array([
            np.mean(voiced_pitch),
            np.std(voiced_pitch)
        ])

    return np.concatenate([
        features,
        rms_features,
        zcr_features,
        contrast_features,
        pitch_features
    ])


def main():
    audio_files = sorted(DATASET_DIR.rglob("*.wav"))

    if not audio_files:
        print("No WAV files found inside dataset/.")
        return

    rows = []
    skipped = 0

    for index, path in enumerate(audio_files, start=1):
        parts = path.stem.split("-")

        if len(parts) != 7 or parts[2] not in EMOTIONS:
            skipped += 1
            continue

        try:
            values = extract_features(path)

            row = {
                f"feature_{i + 1}": value
                for i, value in enumerate(values)
            }

            row["emotion"] = EMOTIONS[parts[2]]
            row["actor_id"] = int(parts[6])
            row["filename"] = path.name
            rows.append(row)

        except Exception as error:
            print(f"Error processing {path.name}: {error}")
            skipped += 1

        if index % 100 == 0:
            print(f"Processed {index}/{len(audio_files)} files")

    if not rows:
        print("No features extracted.")
        return

    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    feature_columns = [
        column for column in df.columns
        if column.startswith("feature_")
    ]

    print("\nAdvanced feature extraction completed!")
    print("Successful recordings:", len(df))
    print("Skipped recordings:", skipped)
    print("Features per recording:", len(feature_columns))
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().sort_index())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
