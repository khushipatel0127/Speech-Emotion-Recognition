import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

TEST_RUNTIME = Path(tempfile.mkdtemp(prefix="speechsense-api-test-"))
atexit.register(shutil.rmtree, TEST_RUNTIME, ignore_errors=True)
os.environ["SPEECHSENSE_DB_PATH"] = str(TEST_RUNTIME / "speechsense.db")
os.environ["SPEECHSENSE_RUNTIME_DIR"] = str(TEST_RUNTIME)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from starlette.testclient import TestClient
from main import app
from database import init_db

init_db()
client = TestClient(app)

def test_api_workflow():
    print("--- 1. Testing GET /api/health ---")
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    print(" Health check OK!")

    print("\n--- 2. Testing POST /api/predict with real WAV ---")
    test_wav = Path(__file__).resolve().parents[1] / "dataset" / "Actor_01" / "03-01-01-01-01-01-01.wav"
    with open(test_wav, "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/predict",
        files={"file": ("test_sample.wav", file_bytes, "audio/wav")}
    )
    assert res.status_code == 200, f"Predict failed: {res.text}"
    p_data = res.json()
    pred_id = p_data["id"]
    assert p_data["predicted_emotion"] == "calm"
    assert "calm" in p_data["probabilities"]
    assert p_data["duration_seconds"] > 0
    assert p_data["model_version"] == "historical-ravdess-8class-v1"
    assert p_data["processing_time_ms"] > 0
    assert isinstance(p_data["is_uncertain"], bool)
    print(f" Prediction OK! ID: {pred_id}, Emotion: {p_data['predicted_emotion']}")

    print("\n--- 3. Testing GET /api/stats ---")
    res = client.get("/api/stats")
    assert res.status_code == 200
    stats = res.json()
    assert stats["total"] == 1
    assert stats["emotion_distribution"]["calm"] == 1
    assert stats["average_duration_seconds"] > 0
    assert stats["recent_count"] == 1
    print(" SQLite dashboard statistics OK!")

    print("\n--- 4. Testing GET /api/history ---")
    res = client.get("/api/history")
    assert res.status_code == 200
    h_data = res.json()
    assert h_data["total"] >= 1
    assert any(item["id"] == pred_id for item in h_data["items"])
    print(f" History OK! Total records: {h_data['total']}")

    print("\n--- 5. Testing GET /api/history/{id} ---")
    res = client.get(f"/api/history/{pred_id}")
    assert res.status_code == 200
    d_data = res.json()
    assert d_data["id"] == pred_id
    assert d_data["predicted_emotion"] == "calm"
    print(" History detail OK!")

    print("\n--- 6. Testing GET /api/report/{id} ---")
    res = client.get(f"/api/report/{pred_id}")
    assert res.status_code == 200
    assert "attachment" in res.headers.get("content-disposition", "")
    assert "SPEECHSENSE" in res.text
    assert "CALM" in res.text
    print(" Report generation OK!")

    print("\n--- 7. Testing GET /audio/{id} ---")
    res = client.get(f"/audio/{pred_id}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert len(res.content) == len(file_bytes)
    print(" Audio playback streaming OK!")

    print("\n--- 8. Testing Validation & Error Handling ---")
    # Non-wav
    res = client.post("/api/predict", files={"file": ("test.txt", b"hello world", "text/plain")})
    assert res.status_code == 400
    # Empty wav
    res = client.post("/api/predict", files={"file": ("empty.wav", b"", "audio/wav")})
    assert res.status_code == 400
    print(" Error handling & input validation OK!")

    print("\n--- 9. Testing DELETE /api/history/{id} ---")
    res = client.delete(f"/api/history/{pred_id}")
    assert res.status_code == 200
    assert res.json()["id"] == pred_id
    assert client.get(f"/api/history/{pred_id}").status_code == 404
    assert client.get(f"/audio/{pred_id}").status_code == 404
    print(" Individual history and audio deletion OK!")

    # Create one more entry to verify bulk deletion independently.
    res = client.post(
        "/api/predict",
        files={"file": ("bulk_delete_sample.wav", file_bytes, "audio/wav")},
    )
    assert res.status_code == 200

    print("\n--- 10. Testing DELETE /api/history ---")
    res = client.delete("/api/history")
    assert res.status_code == 200
    del_data = res.json()
    assert del_data["deleted"] >= 1

    res = client.get("/api/history")
    assert res.json()["total"] == 0
    print(" Delete history OK!")

if __name__ == "__main__":
    test_api_workflow()
    print("\n ALL FASTAPI ENDPOINT TESTS PASSED!")
