import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

TEST_RUNTIME = Path(tempfile.mkdtemp(prefix="speechsense-e2e-test-"))
atexit.register(shutil.rmtree, TEST_RUNTIME, ignore_errors=True)
os.environ["SPEECHSENSE_DB_PATH"] = str(TEST_RUNTIME / "speechsense.db")
os.environ["SPEECHSENSE_RUNTIME_DIR"] = str(TEST_RUNTIME)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from starlette.testclient import TestClient
from main import app
from database import init_db

init_db()
client = TestClient(app)

def test_multiple_ravdess_files():
    print("==================================================")
    print(" RUNNING END-TO-END MULTI-EMOTION PREDICTION TEST ")
    print("==================================================")

    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset" / "Actor_01"

    test_samples = [
        ("03-01-01-01-01-01-01.wav", "Neutral/Calm baseline speech"),
        ("03-01-05-01-01-01-01.wav", "Angry expression speech"),
        ("03-01-03-01-01-01-01.wav", "Happy expression speech"),
    ]

    for filename, description in test_samples:
        file_path = dataset / filename
        assert file_path.exists(), f"Missing audio file: {file_path}"

        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        response = client.post(
            "/api/predict",
            files={"file": (filename, audio_bytes, "audio/wav")}
        )

        assert response.status_code == 200, f"Failed prediction for {filename}: {response.text}"
        data = response.json()

        print(f"\nRecording: {filename} ({description})")
        print(f"Predicted Emotion : {data['predicted_emotion'].upper()}")
        print(f"Duration          : {data['duration_seconds']}s")
        print(f"Top 3 Probabilities:")
        sorted_scores = sorted(data["probabilities"].items(), key=lambda x: x[1], reverse=True)
        for em, score in sorted_scores[:3]:
            print(f"  - {em:<10}: {score * 100:5.2f}%")

        # Verify record exists in history
        pred_id = data["id"]
        h_res = client.get(f"/api/history/{pred_id}")
        assert h_res.status_code == 200

        # Verify report generates
        rep_res = client.get(f"/api/report/{pred_id}")
        assert rep_res.status_code == 200
        assert data["predicted_emotion"].upper() in rep_res.text

        # Verify audio stream
        aud_res = client.get(f"/audio/{pred_id}")
        assert aud_res.status_code == 200
        assert len(aud_res.content) == len(audio_bytes)

    print("\n--------------------------------------------------")
    # Verify history list contains all 3 entries
    list_res = client.get("/api/history")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 3
    print(f"Total history entries in SQLite: {list_res.json()['total']}")
    print(" ALL END-TO-END MULTI-AUDIO TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_multiple_ravdess_files()
