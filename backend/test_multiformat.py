import atexit
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TEST_RUNTIME = Path(tempfile.mkdtemp(prefix="speechsense-format-test-"))
atexit.register(shutil.rmtree, TEST_RUNTIME, ignore_errors=True)
os.environ["SPEECHSENSE_DB_PATH"] = str(TEST_RUNTIME / "speechsense.db")
os.environ["SPEECHSENSE_RUNTIME_DIR"] = str(TEST_RUNTIME)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from starlette.testclient import TestClient
from main import app
from audio_converter import get_ffmpeg_path
from database import init_db

init_db()
client = TestClient(app)

def test_multiformat_pipeline():
    print("========================================================")
    print(" TESTING MULTI-FORMAT AUDIO PIPELINE (WAV/MP3/M4A/MP4)  ")
    print("========================================================")

    ffmpeg = get_ffmpeg_path()
    print(f"Using FFmpeg: {ffmpeg}")

    root = Path(__file__).resolve().parents[1]
    src_wav = root / "dataset" / "Actor_01" / "03-01-01-01-01-01-01.wav"
    assert src_wav.exists(), f"Source audio not found: {src_wav}"

    tmp_dir = TEST_RUNTIME / "generated-inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    test_mp3 = tmp_dir / "test_whatsapp_voice.mp3"
    test_m4a = tmp_dir / "test_iphone_recording.m4a"
    test_mp4_with_audio = tmp_dir / "test_whatsapp_video.mp4"
    test_mp4_no_audio = tmp_dir / "test_silent_video_no_audio.mp4"
    test_corrupt = tmp_dir / "test_corrupt.m4a"

    # Generate realistic multi-format recordings from real RAVDESS audio
    # 1. MP3
    subprocess.run([ffmpeg, "-y", "-i", str(src_wav), "-b:a", "128k", str(test_mp3)], check=True, capture_output=True)
    # 2. M4A
    subprocess.run([ffmpeg, "-y", "-i", str(src_wav), "-c:a", "aac", "-b:a", "128k", str(test_m4a)], check=True, capture_output=True)
    # 3. MP4 with video & audio (WhatsApp video note style)
    subprocess.run([
        ffmpeg, "-y",
        "-i", str(src_wav),
        "-f", "lavfi", "-i", "color=c=black:s=160x120:d=3",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", str(test_mp4_with_audio)
    ], check=True, capture_output=True)
    # 4. MP4 with NO audio stream
    subprocess.run([
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:d=2",
        "-c:v", "libx264", "-an",
        str(test_mp4_no_audio)
    ], check=True, capture_output=True)
    # 5. Corrupt file
    with open(test_corrupt, "wb") as f:
        f.write(b"NOT_A_VALID_AUDIO_FILE_DATA_CORRUPT")

    try:
        # Test 1: Real WAV
        print("\n--- 1. Testing WAV upload ---")
        with open(src_wav, "rb") as f:
            res = client.post("/api/predict", files={"file": (src_wav.name, f.read(), "audio/wav")})
        assert res.status_code == 200, res.text
        wav_data = res.json()
        print(f" WAV OK: emotion={wav_data['predicted_emotion']}, conf={max(wav_data['probabilities'].values())*100:.1f}%")

        # Test 2: Real MP3 (voice note)
        print("\n--- 2. Testing MP3 upload (WhatsApp voice style) ---")
        with open(test_mp3, "rb") as f:
            res = client.post("/api/predict", files={"file": (test_mp3.name, f.read(), "audio/mpeg")})
        assert res.status_code == 200, res.text
        mp3_data = res.json()
        print(f" MP3 OK: emotion={mp3_data['predicted_emotion']}, filename={mp3_data['filename']}")
        assert mp3_data["predicted_emotion"] == wav_data["predicted_emotion"]

        # Test 3: Real M4A (iPhone voice memo)
        print("\n--- 3. Testing M4A upload (iPhone voice memo) ---")
        with open(test_m4a, "rb") as f:
            res = client.post("/api/predict", files={"file": (test_m4a.name, f.read(), "audio/mp4")})
        assert res.status_code == 200, res.text
        m4a_data = res.json()
        print(f" M4A OK: emotion={m4a_data['predicted_emotion']}, filename={m4a_data['filename']}")

        # Test 4: Real MP4 with Audio (WhatsApp video note)
        print("\n--- 4. Testing MP4 upload with audio stream ---")
        with open(test_mp4_with_audio, "rb") as f:
            res = client.post("/api/predict", files={"file": (test_mp4_with_audio.name, f.read(), "video/mp4")})
        assert res.status_code == 200, res.text
        mp4_data = res.json()
        print(f" MP4 with audio OK: emotion={mp4_data['predicted_emotion']}, filename={mp4_data['filename']}")

        # Test 5: Playback of converted audio
        print("\n--- 5. Testing converted audio playback streaming ---")
        aud_res = client.get(f"/audio/{mp4_data['id']}")
        assert aud_res.status_code == 200
        assert aud_res.headers["content-type"] == "audio/wav"
        print(f" Audio streaming OK: {len(aud_res.content)} bytes WAV served")

        # Test 6: Rejection of MP4 without audio stream
        print("\n--- 6. Testing rejection of MP4 with NO audio stream ---")
        with open(test_mp4_no_audio, "rb") as f:
            res = client.post("/api/predict", files={"file": (test_mp4_no_audio.name, f.read(), "video/mp4")})
        assert res.status_code == 400
        print(f" Correctly rejected MP4 with no audio: {res.json()['detail']}")

        # Test 7: Rejection of corrupt file
        print("\n--- 7. Testing rejection of corrupted file ---")
        with open(test_corrupt, "rb") as f:
            res = client.post("/api/predict", files={"file": (test_corrupt.name, f.read(), "audio/mp4")})
        assert res.status_code == 400
        print(f" Correctly rejected corrupt audio: {res.json()['detail']}")

        # Test 8: Browser WebM recording simulation (no extension, blob content-type)
        print("\n--- 8. Testing browser WebM recording upload ---")
        with open(test_m4a, "rb") as f:  # use audio bytes with webm mime
            res = client.post("/api/predict", files={"file": ("blob", f.read(), "audio/webm;codecs=opus")})
        assert res.status_code == 200
        webm_data = res.json()
        print(f" WebM recording OK: emotion={webm_data['predicted_emotion']}, ID={webm_data['id']}")

        print("\n========================================================")
        print(" ALL MULTI-FORMAT AUDIO PIPELINE TESTS PASSED 100%!     ")
        print("========================================================")

    finally:
        # Cleanup test files
        for p in [test_mp3, test_m4a, test_mp4_with_audio, test_mp4_no_audio, test_corrupt]:
            if p.exists():
                try: p.unlink()
                except Exception: pass

if __name__ == "__main__":
    test_multiformat_pipeline()
