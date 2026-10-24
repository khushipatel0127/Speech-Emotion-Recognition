"""Run local recordings through the exact SpeechSense production pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from audio_converter import convert_to_mono_wav  # noqa: E402
from predictor import get_model_info, predict_from_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, nargs="+")
    args = parser.parse_args()

    print(json.dumps({"model": get_model_info()}, indent=2), flush=True)
    failures = 0
    with tempfile.TemporaryDirectory(prefix="speechsense_predictions_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, source in enumerate(args.recordings):
            try:
                converted = temp_root / f"recording_{index:03d}.wav"
                convert_to_mono_wav(source, converted, max_duration_seconds=20.0)
                result = predict_from_path(converted)
                print(
                    json.dumps(
                        {"source": str(source), "prediction": result}, indent=2
                    ),
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                print(
                    json.dumps({"source": str(source), "error": str(exc)}, indent=2),
                    flush=True,
                )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
