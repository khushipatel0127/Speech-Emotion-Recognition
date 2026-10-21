
from pathlib import Path
from collections import Counter
import csv
import io

import pyarrow.parquet as pq
import soundfile as sf

DATA_DIR = Path("dataset/CREMA-D/data")
OUTPUT_DIR = Path("dataset/CREMA-D/wav")
MANIFEST_PATH = Path("dataset/CREMA-D/metadata.csv")

EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

counts = Counter()
total = 0
skipped = 0

parquet_files = sorted(DATA_DIR.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"No Parquet files found in {DATA_DIR}"
    )

with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_path",
            "source_file",
            "speaker_id",
            "emotion",
            "dataset",
            "sample_rate",
        ],
    )
    writer.writeheader()

    for parquet_path in parquet_files:
        print(f"\nProcessing: {parquet_path.name}", flush=True)

        parquet = pq.ParquetFile(parquet_path)

        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                emotion = EMOTIONS.get(row["emotion_code"])

                if emotion is None:
                    skipped += 1
                    continue

                speaker_id = str(row["actor_id"])
                filename = Path(row["source_file"]).name

                speaker_dir = OUTPUT_DIR / speaker_id
                speaker_dir.mkdir(parents=True, exist_ok=True)

                output_path = speaker_dir / filename

                if not output_path.exists():
                    audio_bytes = row["audio"]["bytes"]

                    if not audio_bytes:
                        raise ValueError(
                            f"Missing audio bytes for {filename}"
                        )

                    audio, sr = sf.read(
                        io.BytesIO(audio_bytes),
                        dtype="float32",
                    )

                    if sr != 16000:
                        raise ValueError(
                            f"Unexpected sample rate {sr}: {filename}"
                        )

                    sf.write(
                        str(output_path),
                        audio,
                        sr,
                        subtype="PCM_16",
                    )

                writer.writerow({
                    "file_path": output_path.as_posix(),
                    "source_file": filename,
                    "speaker_id": f"CREMAD_{speaker_id}",
                    "emotion": emotion,
                    "dataset": "CREMA-D",
                    "sample_rate": 16000,
                })

                counts[emotion] += 1
                total += 1

        print(f"Completed: {parquet_path.name}", flush=True)

print("\n=== CREMA-D EXTRACTION COMPLETE ===")
print(f"Total recordings: {total}")
print(f"Skipped recordings: {skipped}")
print(f"Metadata: {MANIFEST_PATH}")

for emotion, count in sorted(counts.items()):
    print(f"{emotion}: {count}")
