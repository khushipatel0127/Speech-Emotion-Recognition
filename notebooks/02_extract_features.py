
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import librosa

DATASET_DIR = Path("dataset")
OUTPUT_FILE = Path("notebooks/features.csv")

# RAVDESS emotion codes
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


def extract_mfcc(audio_path):
    """Extract a fixed-length MFCC feature vector."""
    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Mean and standard deviation across time
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return features


def main():
    audio_files = sorted(DATASET_DIR.rglob("*.wav"))

    if not audio_files:
        print("No WAV files found in dataset/.")
        return

    rows = []
    skipped = 0

    for index, audio_path in enumerate(audio_files, start=1):
        # Standard RAVDESS filename:
        # modality-channel-emotion-intensity-statement-repetition-actor.wav
        parts = audio_path.stem.split("-")

        if len(parts) != 7 or parts[2] not in EMOTIONS:
            print("Skipping unexpected filename:", audio_path.name)
            skipped += 1
            continue

        emotion_code = parts[2]
        actor_id = int(parts[6])

        try:
            features = extract_mfcc(audio_path)

            row = {
                f"mfcc_{i + 1}": value
                for i, value in enumerate(features)
            }

            row["emotion"] = EMOTIONS[emotion_code]
            row["actor_id"] = actor_id
            row["filename"] = audio_path.name

            rows.append(row)

        except Exception as error:
            print(f"Could not process {audio_path.name}: {error}")
            skipped += 1

        if index % 100 == 0:
            print(f"Processed {index}/{len(audio_files)} files")

    if not rows:
        print("No features extracted. Check the dataset and filenames.")
        return

    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature extraction completed!")
    print("Successfully processed:", len(df))
    print("Skipped:", skipped)
    print("Features per recording:", 80)
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().sort_index())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
