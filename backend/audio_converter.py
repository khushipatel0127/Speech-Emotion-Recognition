import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Tuple

_FFMPEG_PATH: Optional[str] = None

def get_ffmpeg_path() -> str:
    """Find a valid FFmpeg executable (system PATH or bundled imageio-ffmpeg)."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH and os.path.isfile(_FFMPEG_PATH):
        return _FFMPEG_PATH

    # 1. Try system PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        try:
            res = subprocess.run([sys_ffmpeg, "-version"], capture_output=True, timeout=5)
            if res.returncode == 0:
                _FFMPEG_PATH = sys_ffmpeg
                return _FFMPEG_PATH
        except Exception:
            pass

    # 2. Try imageio_ffmpeg
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            _FFMPEG_PATH = bundled
            return _FFMPEG_PATH
    except Exception:
        pass

    raise RuntimeError(
        "FFmpeg executable not found. Please install FFmpeg or install imageio-ffmpeg via pip."
    )

def convert_to_mono_wav(
    input_path: Path,
    output_wav_path: Path,
    max_duration_seconds: float = 180.0
) -> Tuple[float, int]:
    """
    Extract audio stream and convert to 16,000 Hz, 1-channel (mono), 16-bit PCM WAV.
    Rejects files without audio streams, corrupted files, and recordings exceeding max duration.
    
    Returns:
        (duration_seconds, file_size_bytes)
    """
    input_path = Path(input_path)
    output_wav_path = Path(output_wav_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Source file not found: {input_path}")

    ffmpeg = get_ffmpeg_path()

    # ffmpeg command:
    # -y: overwrite
    # -i <input>: input file (mp4, m4a, mp3, ogg, webm, wav, etc.)
    # -vn: disable video recording (strip video stream if mp4/webm)
    # -ac 1: mono (1 channel)
    # -ar 16000: 16 kHz sample rate
    # -c:a pcm_s16le: 16-bit uncompressed PCM WAV
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_wav_path)
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45
        )
    except subprocess.TimeoutExpired:
        raise ValueError("Audio conversion timed out. The file may be too large or corrupted.")
    except Exception as e:
        raise RuntimeError(f"Failed to execute FFmpeg: {str(e)}")

    if proc.returncode != 0:
        stderr = proc.stderr or ""
        stderr_lower = stderr.lower()

        if "does not contain any stream" in stderr_lower or "output file does not contain any stream" in stderr_lower or "matches no streams" in stderr_lower:
            raise ValueError("The uploaded file does not contain an audio stream.")
        elif "invalid data found" in stderr_lower or "moov atom not found" in stderr_lower:
            raise ValueError("The audio/video file appears corrupted or unreadable.")
        else:
            # Extract last descriptive line if possible
            lines = [line.strip() for line in stderr.splitlines() if line.strip()]
            error_msg = lines[-1] if lines else "Unknown FFmpeg conversion error"
            raise ValueError(f"Audio extraction failed: {error_msg}")

    if not output_wav_path.exists() or output_wav_path.stat().st_size <= 44:
        raise ValueError("Audio conversion produced an empty or invalid audio stream.")

    # Check duration using wave module or soundfile for precision
    import soundfile as sf
    try:
        info = sf.info(str(output_wav_path))
        duration = info.duration
    except Exception:
        # Fallback to file size calculation for 16kHz 16-bit mono: (file_size - 44) / (16000 * 2)
        raw_bytes = max(0, output_wav_path.stat().st_size - 44)
        duration = raw_bytes / 32000.0

    if duration < 0.5:
        raise ValueError(f"Recording is too short ({duration:.2f}s). Please provide at least 0.5 seconds of speech.")

    if duration > max_duration_seconds:
        raise ValueError(
            f"Recording is too long ({duration:.1f}s). Please provide audio under {int(max_duration_seconds)} seconds."
        )

    return round(duration, 2), output_wav_path.stat().st_size
