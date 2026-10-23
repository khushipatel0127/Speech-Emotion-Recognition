"""Extract resumable frozen WavLM embeddings in small, disk-backed shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "reports" / "phase2" / "dataset_manifest.csv"
DEFAULT_OUTPUT = ROOT / "models" / "phase2" / "wavlm_embeddings"
DEFAULT_MODEL = "microsoft/wavlm-base-plus"


def load_dependencies():
    try:
        import torch
        import transformers
        from transformers import AutoFeatureExtractor, WavLMModel
    except ImportError as exc:
        raise RuntimeError(
            "WavLM extraction requires torch, transformers, and tqdm. "
            "Use the supplied Google Colab notebook or install the optional dependencies."
        ) from exc
    return torch, transformers, AutoFeatureExtractor, WavLMModel


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["model_eligible"].lower() == "true"]
    return sorted(rows, key=lambda row: row["audio_path"])


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz audio, found {sample_rate}: {path}")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio: {path}")
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError(f"Empty or invalid audio: {path}")
    return audio


def completed_shards(output_dir: Path) -> set[int]:
    completed = set()
    for path in output_dir.glob("shard_*.npz"):
        try:
            completed.add(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")

    torch, transformers, AutoFeatureExtractor, WavLMModel = load_dependencies()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    rows = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("model_name") != args.model_name:
            raise RuntimeError("Existing shards were created by a different WavLM model")

    print(f"Loading {args.model_name} on {device}...", flush=True)
    processor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = WavLMModel.from_pretrained(args.model_name).to(device)
    model.eval()

    completed = completed_shards(args.output_dir)
    shard_count = (len(rows) + args.shard_size - 1) // args.shard_size
    started = time.perf_counter()
    for shard_index in range(shard_count):
        if shard_index in completed:
            continue
        start = shard_index * args.shard_size
        shard_rows = rows[start : start + args.shard_size]
        embeddings = []
        audio_paths = []
        for row in tqdm(shard_rows, desc=f"Shard {shard_index + 1}/{shard_count}"):
            audio = load_audio(ROOT / row["audio_path"])
            inputs = processor(
                audio,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(device)
            with torch.inference_mode():
                hidden = model(input_values=input_values).last_hidden_state
                embedding = hidden.mean(dim=1).squeeze(0)
            embeddings.append(embedding.detach().cpu().numpy().astype(np.float32))
            audio_paths.append(row["audio_path"])

        final_path = args.output_dir / f"shard_{shard_index:04d}.npz"
        temporary_path = args.output_dir / f"shard_{shard_index:04d}.tmp.npz"
        np.savez_compressed(
            temporary_path,
            audio_paths=np.asarray(audio_paths),
            embeddings=np.stack(embeddings),
        )
        os.replace(temporary_path, final_path)
        completed.add(shard_index)
        checkpoint = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": args.model_name,
            "device": device,
            "recordings": len(rows),
            "embedding_size": int(embeddings[0].shape[0]),
            "shard_size": args.shard_size,
            "completed_shards": len(completed),
            "total_shards": shard_count,
            "complete": len(completed) == shard_count,
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        print(f"Saved {final_path.name}", flush=True)

    print(f"Embedding extraction complete: {len(rows)} recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
