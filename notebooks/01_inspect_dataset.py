
from pathlib import Path
import librosa

# Project dataset directory
DATASET_DIR = Path("dataset")

# Find all WAV audio files
audio_files = list(DATASET_DIR.rglob("*.wav"))

print("Total WAV files found:", len(audio_files))

if audio_files:
    # Inspect the first audio file
    audio_path = audio_files[0]

    print("\nExample file:", audio_path.name)

    # Load audio as a mono signal
    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    print("Sample rate:", sample_rate, "Hz")
    print("Duration:", round(len(audio) / sample_rate, 2), "seconds")
    print("Audio loaded successfully!")
else:
    print("No WAV files found.")
    print("Check whether the dataset is extracted inside the dataset folder.")
