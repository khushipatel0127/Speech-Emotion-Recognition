"""Extract resumable frozen WavLM embeddings in small, disk-backed shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    torch, transformers, AutoFeatureExtractor, WavLMModel = load_dependencies()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    rows = load_manifest(args.manifest)
    manifest_hash = file_sha256(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.json"
    checkpoint: dict[str, object] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("model_name") != args.model_name:
            raise RuntimeError("Existing shards were created by a different WavLM model")
        if checkpoint.get("manifest_sha256") not in (None, manifest_hash):
            raise RuntimeError("The manifest changed after WavLM extraction started")
        if checkpoint.get("shard_size") != args.shard_size:
            raise RuntimeError("Existing shards use a different --shard-size")

    print(f"Loading {args.model_name} on {device}...", flush=True)
    processor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = WavLMModel.from_pretrained(args.model_name).to(device)
    model.eval()

    completed = completed_shards(args.output_dir)
    shard_count = (len(rows) + args.shard_size - 1) // args.shard_size
    started = time.perf_counter()
    embedding_size = None
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        shard_rows = rows[start : start + args.shard_size]
        expected_paths = [row["audio_path"] for row in shard_rows]
        final_path = args.output_dir / f"shard_{shard_index:04d}.npz"
        if shard_index in completed:
            with np.load(final_path, allow_pickle=False) as existing:
                existing_paths = [str(value) for value in existing["audio_paths"]]
                existing_embeddings = existing["embeddings"]
                if existing_paths != expected_paths or existing_embeddings.ndim != 2:
                    raise RuntimeError(f"Existing shard is incompatible: {final_path}")
                embedding_size = int(existing_embeddings.shape[1])
            continue
        embeddings = []
        audio_paths = []
        batch_starts = range(0, len(shard_rows), args.batch_size)
        for batch_start in tqdm(
            batch_starts,
            total=math.ceil(len(shard_rows) / args.batch_size),
            desc=f"Shard {shard_index + 1}/{shard_count}",
        ):
            batch_rows = shard_rows[batch_start : batch_start + args.batch_size]
            audios = [load_audio(ROOT / row["audio_path"]) for row in batch_rows]
            inputs = processor(
                audios,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            with torch.inference_mode():
                hidden = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).last_hidden_state
                feature_mask = model._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                ).unsqueeze(-1)
                summed = (hidden * feature_mask).sum(dim=1)
                counts = feature_mask.sum(dim=1).clamp(min=1)
                batch_embeddings = summed / counts
            embeddings.extend(
                batch_embeddings.detach().cpu().numpy().astype(np.float32)
            )
            audio_paths.extend(row["audio_path"] for row in batch_rows)

        temporary_path = args.output_dir / f"shard_{shard_index:04d}.tmp.npz"
        np.savez_compressed(
            temporary_path,
            audio_paths=np.asarray(audio_paths),
            embeddings=np.stack(embeddings),
        )
        os.replace(temporary_path, final_path)
        completed.add(shard_index)
        embedding_size = int(embeddings[0].shape[0])
        checkpoint = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": args.model_name,
            "manifest_sha256": manifest_hash,
            "device": device,
            "recordings": len(rows),
            "embedding_size": embedding_size,
            "shard_size": args.shard_size,
            "batch_size": args.batch_size,
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

    if len(completed) != shard_count or embedding_size is None:
        raise RuntimeError("WavLM extraction ended with incomplete shards")
    checkpoint.update(
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest_hash,
            "device": device,
            "recordings": len(rows),
            "embedding_size": embedding_size,
            "shard_size": args.shard_size,
            "batch_size": args.batch_size,
            "completed_shards": len(completed),
            "total_shards": shard_count,
            "complete": True,
            "elapsed_seconds_this_run": time.perf_counter() - started,
        }
    )
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    print(f"Embedding extraction complete: {len(rows)} recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
