import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import joblib
import librosa
import soundfile as sf

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "final_model"
DATASET_DIR = ROOT / "dataset"

def run_diagnostics():
    print("=" * 70)
    print("       SPEECHSENSE ML PREDICTION PIPELINE DIAGNOSTICS       ")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Inspect Model Architecture, Classes, and Label Encoder
    # ---------------------------------------------------------
    print("\n[DIAGNOSTIC 1] Model Classes & Label Encoder Alignment")
    svm = joblib.load(MODEL_DIR / "svm_model.joblib")
    hgb = joblib.load(MODEL_DIR / "hgb_model.joblib")
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    print(f"Encoder classes: {encoder.classes_.tolist()}")
    print(f"SVM classes_   : {svm.classes_.tolist()}")
    print(f"HGB classes_   : {hgb.classes_.tolist()}")
    assert np.array_equal(svm.classes_, hgb.classes_), "SVM and HGB class ordering mismatch!"
    
    # Check if svm.classes_ indices match encoder
    class_names_svm = encoder.inverse_transform(svm.classes_)
    print(f"Mapped SVM classes to names: {class_names_svm.tolist()}")
    assert list(class_names_svm) == list(encoder.classes_), "Classes do not match encoder ordering!"
    print(" Label encoder and model class ordering are 100% verified and aligned.")

    # ---------------------------------------------------------
    # 2. Compare Features: 04 vs 14 vs backend predictor
    # ---------------------------------------------------------
    print("\n[DIAGNOSTIC 2] Feature Extraction Numerical Verification")
    sample_wav = DATASET_DIR / "Actor_01" / "03-01-01-01-01-01-01.wav"

    # From notebooks/14_predict_audio.py
    import sys
    sys.path.insert(0, str(ROOT / "notebooks"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict_audio", str(ROOT / "notebooks" / "14_predict_audio.py"))
    mod_14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_14)

    feat_14, dur_14 = mod_14.extract_features(sample_wav, metadata)

    # From backend/predictor.py
    sys.path.insert(0, str(ROOT / "backend"))
    from predictor import extract_features as backend_extract

    feat_backend, dur_backend = backend_extract(sample_wav, metadata)

    max_diff = np.max(np.abs(feat_14 - feat_backend))
    print(f"Max absolute difference between 14_predict_audio and backend: {max_diff}")
    assert max_diff == 0.0, f"Feature mismatch! Diff: {max_diff}"
    print(" Feature extraction in backend matches 14_predict_audio.py EXACTLY (0.000 diff).")

    # ---------------------------------------------------------
    # 3. Verify FFmpeg MP4 / WebM conversion fidelity
    # ---------------------------------------------------------
    print("\n[DIAGNOSTIC 3] FFmpeg Audio Conversion vs Original WAV")
    from audio_converter import get_ffmpeg_path, convert_to_mono_wav
    ffmpeg = get_ffmpeg_path()
    
    tmp_mp4 = ROOT / "backend" / "uploads" / "diag_test.mp4"
    tmp_wav = ROOT / "backend" / "uploads" / "diag_test_converted.wav"
    
    # Encode sample_wav to AAC in MP4
    subprocess.run([
        ffmpeg, "-y", "-i", str(sample_wav),
        "-c:a", "aac", "-b:a", "128k",
        "-f", "mp4", str(tmp_mp4)
    ], check=True, capture_output=True)

    # Convert back to 16kHz mono WAV using our pipeline
    dur_conv, sz_conv = convert_to_mono_wav(tmp_mp4, tmp_wav)
    feat_conv, _ = backend_extract(tmp_wav, metadata)

    # Check librosa sample rate and channels
    y_orig, sr_orig = librosa.load(str(sample_wav), sr=16000, mono=True)
    y_conv, sr_conv = librosa.load(str(tmp_wav), sr=16000, mono=True)

    print(f"Original WAV: sr={sr_orig}, len={len(y_orig)}, rms={np.sqrt(np.mean(y_orig**2)):.5f}")
    print(f"Converted   : sr={sr_conv}, len={len(y_conv)}, rms={np.sqrt(np.mean(y_conv**2)):.5f}")

    # Compare features between original lossless WAV and AAC-compressed MP4
    feat_diff = np.mean(np.abs(feat_14 - feat_conv))
    print(f"Mean feature difference due to AAC lossy codec compression: {feat_diff:.4f}")

    # Compare model predictions on original vs MP4
    res_orig = mod_14.predict_emotion(sample_wav)
    res_mp4 = mod_14.predict_emotion(tmp_wav)
    print(f"Prediction on original WAV: {res_orig['predicted_emotion']} (calm={res_orig['emotion_scores']['calm']*100:.2f}%)")
    print(f"Prediction on MP4 convert: {res_mp4['predicted_emotion']} (calm={res_mp4['emotion_scores']['calm']*100:.2f}%)")

    # Cleanup
    for p in [tmp_mp4, tmp_wav]:
        if p.exists(): p.unlink()

    # ---------------------------------------------------------
    # 4. Excluded Actor Evaluation: Happy vs Angry Analysis
    # ---------------------------------------------------------
    print("\n[DIAGNOSTIC 4] Excluded Actors Evaluation (Unseen Actors)")
    excluded_actors = metadata["excluded_actors"]
    print(f"Excluded actors from training: {excluded_actors}")

    # Test ALL happy recordings from excluded actors
    happy_files = []
    angry_files = []
    all_excluded_files = []

    for actor in excluded_actors:
        actor_dir = DATASET_DIR / f"Actor_{actor:02d}"
        if not actor_dir.exists():
            continue
        for wav_path in actor_dir.glob("*.wav"):
            parts = wav_path.stem.split("-")
            em_code = parts[2]
            emotion = metadata["emotion_codes"].get(em_code)
            all_excluded_files.append((wav_path, emotion, actor))
            if emotion == "happy":
                happy_files.append((wav_path, actor))
            elif emotion == "angry":
                angry_files.append((wav_path, actor))

    print(f"Found {len(happy_files)} HAPPY recordings and {len(angry_files)} ANGRY recordings from excluded actors.")

    happy_predictions = []
    for path, actor in happy_files:
        pred = mod_14.predict_emotion(path)
        happy_predictions.append({
            "filename": path.name,
            "actor": actor,
            "predicted": pred["predicted_emotion"],
            "happy_prob": pred["emotion_scores"]["happy"],
            "angry_prob": pred["emotion_scores"]["angry"],
            "top_prob": max(pred["emotion_scores"].values()),
        })

    df_happy = pd.DataFrame(happy_predictions)
    print("\n--- HAPPY Recordings Prediction Breakdown (Excluded Actors) ---")
    print(df_happy["predicted"].value_counts())
    happy_correct = (df_happy["predicted"] == "happy").sum()
    happy_as_angry = (df_happy["predicted"] == "angry").sum()
    print(f"Happy Correctly Classified: {happy_correct}/{len(df_happy)} ({happy_correct/len(df_happy)*100:.1f}%)")
    print(f"Happy Misclassified as ANGRY: {happy_as_angry}/{len(df_happy)} ({happy_as_angry/len(df_happy)*100:.1f}%)")

    # Sample misclassified happy files
    print("\nSample Happy recordings misclassified:")
    print(df_happy[df_happy["predicted"] != "happy"][["filename", "actor", "predicted", "happy_prob", "angry_prob"]].head(8))

    # ---------------------------------------------------------
    # 5. Acoustic Feature Inspection: Happy vs Angry in Training Data
    # ---------------------------------------------------------
    print("\n[DIAGNOSTIC 5] Acoustic Feature Comparison: Happy vs Angry")
    df_adv = pd.read_csv(ROOT / "notebooks" / "advanced_features.csv")
    
    happy_df = df_adv[df_adv["emotion"] == "happy"]
    angry_df = df_adv[df_adv["emotion"] == "angry"]
    calm_df = df_adv[df_adv["emotion"] == "calm"]

    # Feature 81 is RMS mean, Feature 82 is RMS std, Feature 83 is ZCR mean, Feature 84 is ZCR std
    print(f"RMS Mean  - Angry: {angry_df['feature_81'].mean():.4f} +/- {angry_df['feature_81'].std():.4f}")
    print(f"RMS Mean  - Happy: {happy_df['feature_81'].mean():.4f} +/- {happy_df['feature_81'].std():.4f}")
    print(f"RMS Mean  - Calm : {calm_df['feature_81'].mean():.4f} +/- {calm_df['feature_81'].std():.4f}")

    print(f"ZCR Mean  - Angry: {angry_df['feature_83'].mean():.4f} +/- {angry_df['feature_83'].std():.4f}")
    print(f"ZCR Mean  - Happy: {happy_df['feature_83'].mean():.4f} +/- {happy_df['feature_83'].std():.4f}")
    print(f"ZCR Mean  - Calm : {calm_df['feature_83'].mean():.4f} +/- {calm_df['feature_83'].std():.4f}")

    # MFCC 1 (overall loudness / energy)
    print(f"MFCC 1 (Energy) - Angry: {angry_df['feature_1'].mean():.2f}")
    print(f"MFCC 1 (Energy) - Happy: {happy_df['feature_1'].mean():.2f}")
    print(f"MFCC 1 (Energy) - Calm : {calm_df['feature_1'].mean():.2f}")

    # ---------------------------------------------------------
    # 6. Check Volume / Gain Sensitivity (Simulating Mic Gain)
    # ---------------------------------------------------------
    print("\n[DIAGNOSTIC 6] Sensitivity to Volume / Microphone Gain")
    # Take a happy recording and test predictions at different volume scalings
    sample_happy = happy_files[0][0]
    y_h, sr_h = librosa.load(str(sample_happy), sr=16000, mono=True)
    
    print(f"Testing sample happy file: {sample_happy.name}")
    gains = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for g in gains:
        scaled_y = y_h * g
        # Extract features manually with gain
        n_mfcc = 40
        mfcc = librosa.feature.mfcc(y=scaled_y, sr=sr_h, n_mfcc=n_mfcc)
        rms = librosa.feature.rms(y=scaled_y)
        zcr = librosa.feature.zero_crossing_rate(scaled_y)
        feats = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(rms, axis=1),
            np.std(rms, axis=1),
            np.mean(zcr, axis=1),
            np.std(zcr, axis=1)
        ]).reshape(1, -1)

        svm_p = svm.predict_proba(feats)[0]
        hgb_p = hgb.predict_proba(feats)[0]
        p = 0.5 * svm_p + 0.5 * hgb_p
        top_idx = svm.classes_[np.argmax(p)]
        top_em = encoder.inverse_transform([top_idx])[0]
        happy_score = p[list(encoder.classes_).index("happy")]
        angry_score = p[list(encoder.classes_).index("angry")]
        print(f"  Gain {g:4.2f}x (RMS {np.sqrt(np.mean(scaled_y**2)):.4f}): Top = {top_em:<10} | Happy: {happy_score*100:5.1f}% | Angry: {angry_score*100:5.1f}%")

if __name__ == "__main__":
    run_diagnostics()
