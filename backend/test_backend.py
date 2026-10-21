import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from database import (
    init_db,
    save_prediction,
    get_predictions,
    get_prediction_by_id,
    delete_all_predictions,
)
from predictor import predict_from_path, get_models

def test_database():
    print("--- Testing SQLite Database ---")
    init_db()
    delete_all_predictions()

    save_prediction(
        prediction_id="test-id-1",
        filename="test.wav",
        predicted_emotion="happy",
        probabilities={"happy": 0.85, "sad": 0.15},
        duration_seconds=2.5,
        file_size_bytes=10240,
        audio_path="audio/test-id-1.wav",
        created_at="2026-10-20T10:00:00Z"
    )

    items, total = get_predictions()
    assert total == 1, f"Expected total 1, got {total}"
    assert items[0]["predicted_emotion"] == "happy"
    assert items[0]["id"] == "test-id-1"
    assert isinstance(items[0]["probabilities"], dict)

    record = get_prediction_by_id("test-id-1")
    assert record is not None
    assert record["filename"] == "test.wav"

    # Search filter
    s_items, s_total = get_predictions(search="test")
    assert s_total == 1
    s_items2, s_total2 = get_predictions(search="nonexistent")
    assert s_total2 == 0

    # Emotion filter
    e_items, e_total = get_predictions(emotion="happy")
    assert e_total == 1
    e_items2, e_total2 = get_predictions(emotion="sad")
    assert e_total2 == 0

    # Clean up
    deleted = delete_all_predictions()
    assert deleted == 1
    _, after_total = get_predictions()
    assert after_total == 0

    print(" Database tests PASSED!")

def test_prediction_accuracy():
    print("\n--- Testing Model Prediction on Real RAVDESS Audio ---")
    test_wav = Path(__file__).resolve().parents[1] / "dataset" / "Actor_01" / "03-01-01-01-01-01-01.wav"
    assert test_wav.exists(), f"Sample audio not found at {test_wav}"

    result = predict_from_path(test_wav)
    print(f"File: {test_wav.name}")
    print(f"Predicted emotion: {result['predicted_emotion']}")
    print(f"Duration: {result['duration_seconds']}s")
    print("Scores:")
    for em, score in result["emotion_scores"].items():
        print(f"  {em:<12}: {score * 100:.2f}%")

    # From 14_predict_audio.py:
    # calm: 38.20%, neutral: 29.85%, sad: 20.55%
    expected_top = "calm"
    assert result["predicted_emotion"] == expected_top, f"Expected {expected_top}, got {result['predicted_emotion']}"
    assert abs(result["emotion_scores"]["calm"] - 0.3820) < 0.005, f"Score mismatch: {result['emotion_scores']['calm']}"
    assert abs(result["emotion_scores"]["neutral"] - 0.2985) < 0.005, f"Score mismatch: {result['emotion_scores']['neutral']}"
    print(" Model prediction matches 14_predict_audio.py EXACTLY!")

if __name__ == "__main__":
    test_database()
    test_prediction_accuracy()
    print("\n ALL BACKEND CORE TESTS PASSED SUCCESSFULLY!")
