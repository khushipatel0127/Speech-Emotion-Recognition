from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import time
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from database import (
    init_db,
    save_prediction,
    get_predictions,
    get_prediction_by_id,
    delete_all_predictions,
)
from predictor import get_models, predict_from_path
from audio_converter import convert_to_mono_wav, get_ffmpeg_path

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.environ.get("SPEECHSENSE_RUNTIME_DIR", str(BASE_DIR))).resolve()
AUDIO_DIR = RUNTIME_DIR / "audio"
UPLOADS_DIR = RUNTIME_DIR / "uploads"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".webm", ".aac", ".wma", ".flac"
}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

app = FastAPI(
    title="SpeechSense API",
    description="Speech Emotion Recognition Backend using Ensemble ML & FFmpeg Audio Conversion",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()
    try:
        get_models()
        print("Models successfully preloaded.")
    except Exception as e:
        print(f"Warning: Failed to preload models: {e}")

    try:
        ff_path = get_ffmpeg_path()
        print(f"FFmpeg ready: {ff_path}")
    except Exception as e:
        print(f"Warning: FFmpeg check failed: {e}")

@app.get("/api/health")
def health_check():
    try:
        models = get_models()
        models_loaded = models is not None and "svm" in models and "hgb" in models
    except Exception:
        models_loaded = False

    try:
        ff_path = get_ffmpeg_path()
        ffmpeg_ok = bool(ff_path and os.path.exists(ff_path))
    except Exception:
        ffmpeg_ok = False
        ff_path = None

    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "ffmpeg_available": ffmpeg_ok,
        "ffmpeg_path": ff_path,
        "supported_formats": sorted(list(SUPPORTED_EXTENSIONS)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...)):
    processing_started = time.perf_counter()
    raw_filename = file.filename or "recording.webm"
    ext = Path(raw_filename).suffix.lower()

    # If no extension or unrecognized, allow browser recording types
    if not ext:
        content_type = (file.content_type or "").lower()
        if "webm" in content_type:
            ext = ".webm"
        elif "wav" in content_type:
            ext = ".wav"
        elif "mp4" in content_type:
            ext = ".mp4"
        elif "ogg" in content_type:
            ext = ".ogg"
        else:
            ext = ".audio"

    if ext not in SUPPORTED_EXTENSIONS and ext != ".audio":
        allowed_str = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported formats: {allowed_str.upper()}.",
        )

    # Read binary content
    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded audio file is empty (0 bytes).",
        )

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 25 MB limit.",
        )

    prediction_id = str(uuid.uuid4())
    temp_raw_path = UPLOADS_DIR / f"{prediction_id}_raw{ext}"
    temp_converted_wav = UPLOADS_DIR / f"{prediction_id}_converted.wav"
    permanent_audio_path = AUDIO_DIR / f"{prediction_id}.wav"

    try:
        # Save raw uploaded recording
        with open(temp_raw_path, "wb") as f:
            f.write(content)

        # Convert to 16kHz mono PCM WAV using FFmpeg
        try:
            duration_sec, converted_size = convert_to_mono_wav(
                input_path=temp_raw_path,
                output_wav_path=temp_converted_wav,
                max_duration_seconds=180.0
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Audio processing error: {str(ex)}")

        # Run real ML prediction on converted 16kHz mono WAV
        try:
            result = predict_from_path(temp_converted_wav)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(ex)}")

        # Move/copy converted WAV to permanent storage for in-browser playback
        shutil.copy2(temp_converted_wav, permanent_audio_path)

        created_at = datetime.now(timezone.utc).isoformat()
        rel_audio_path = f"audio/{prediction_id}.wav"
        processing_time_ms = round((time.perf_counter() - processing_started) * 1000, 2)

        # Save to SQLite
        save_prediction(
            prediction_id=prediction_id,
            filename=raw_filename,
            predicted_emotion=result["predicted_emotion"],
            probabilities=result["emotion_scores"],
            duration_seconds=result["duration_seconds"],
            file_size_bytes=file_size,
            audio_path=rel_audio_path,
            created_at=created_at,
            model_version=result["model_version"],
            processing_time_ms=processing_time_ms,
            is_uncertain=result["is_uncertain"],
        )

        return {
            "id": prediction_id,
            "filename": raw_filename,
            "predicted_emotion": result["predicted_emotion"],
            "probabilities": result["emotion_scores"],
            "emotion_scores": result["emotion_scores"],
            "duration_seconds": result["duration_seconds"],
            "file_size_bytes": file_size,
            "model_version": result["model_version"],
            "processing_time_ms": processing_time_ms,
            "is_uncertain": result["is_uncertain"],
            "uncertainty_note": result["uncertainty_note"],
            "created_at": created_at,
            "audio_url": f"/audio/{prediction_id}",
        }

    finally:
        # Clean up temporary conversion files
        if temp_raw_path.exists():
            try:
                temp_raw_path.unlink()
            except Exception:
                pass
        if temp_converted_wav.exists():
            try:
                temp_converted_wav.unlink()
            except Exception:
                pass

@app.get("/api/history")
def get_history(
    search: str = Query(None),
    emotion: str = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(15, ge=1, le=100),
):
    items, total = get_predictions(
        search=search,
        emotion=emotion,
        page=page,
        per_page=per_page,
    )
    pages = math.ceil(total / per_page) if total > 0 else 1

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }

@app.get("/api/history/{prediction_id}")
def get_history_detail(prediction_id: str):
    record = get_prediction_by_id(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    record["audio_url"] = f"/audio/{prediction_id}"
    return record

@app.delete("/api/history")
def clear_history():
    deleted_count = delete_all_predictions()

    # Clear stored audio files
    for item in AUDIO_DIR.glob("*.wav"):
        try:
            item.unlink()
        except Exception:
            pass

    return {
        "status": "success",
        "deleted": deleted_count,
        "message": f"Successfully deleted {deleted_count} prediction records.",
    }

@app.get("/api/report/{prediction_id}")
def download_report(prediction_id: str):
    record = get_prediction_by_id(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    probs = record.get("probabilities", {})
    if isinstance(probs, str):
        import json
        try:
            probs = json.loads(probs)
        except Exception:
            probs = {}

    report_lines = [
        "============================================================",
        "             SPEECHSENSE — ANALYSIS REPORT                  ",
        "          Speech Emotion Recognition Ensemble               ",
        "============================================================",
        f"Prediction ID    : {record['id']}",
        f"Filename         : {record['filename']}",
        f"Analysis Date    : {record['created_at']}",
        f"Audio Duration   : {record['duration_seconds']} seconds",
        f"Audio File Size  : {record['file_size_bytes']} bytes",
        f"Model Version    : {record.get('model_version') or 'historical-ravdess-8class-v1'}",
        f"Processing Time  : {record.get('processing_time_ms') or 'Unavailable'} ms",
        f"Low-Certainty    : {'Yes' if record.get('is_uncertain') else 'No'}",
        "------------------------------------------------------------",
        f"PREDICTED EMOTION: {record['predicted_emotion'].upper()}",
        "------------------------------------------------------------",
        "PROBABILITY BREAKDOWN (SVM + HistGradientBoosting Soft Voting):",
    ]

    for em, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {em.capitalize():<12} : {prob * 100:6.2f}%")

    report_lines.extend([
        "------------------------------------------------------------",
        "NOTE: Predictions are statistical estimates derived from",
        "the RAVDESS dataset using MFCC, RMS, and ZCR acoustic",
        "features. Results should not be used as clinical or legal proof",
        "of emotional state.",
        "============================================================",
    ])

    report_text = "\n".join(report_lines)
    filename = f"speechsense_report_{record['id'][:8]}.txt"

    return Response(
        content=report_text,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )

def _serve_audio(prediction_id: str):
    audio_path = AUDIO_DIR / f"{prediction_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=f"{prediction_id}.wav",
    )

@app.get("/audio/{prediction_id}")
def serve_audio(prediction_id: str):
    return _serve_audio(prediction_id)

@app.get("/api/audio/{prediction_id}")
def serve_api_audio(prediction_id: str):
    return _serve_audio(prediction_id)
