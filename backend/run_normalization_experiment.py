import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "models" / "normalization_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_RMS = 0.010
MAX_PEAK = 0.95
MIN_RMS_THRESHOLD = 1e-5

EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def normalize_audio(y: np.ndarray, target_rms: float = TARGET_RMS, max_peak: float = MAX_PEAK) -> np.ndarray:
    """
    Applies audio loudness normalization with safeguards against silence amplification and digital clipping.
    """
    if len(y) == 0:
        return y

    rms = np.sqrt(np.mean(np.square(y, dtype=np.float64)))

    # Safeguard 1: Silence threshold
    if rms < MIN_RMS_THRESHOLD:
        return y

    # Calculate gain to reach target RMS
    scale = target_rms / rms

    # Limit maximum boost factor to 10x (20 dB) to prevent amplifying faint background noise
    scale = min(scale, 10.0)
    y_norm = y * scale

    # Safeguard 2: Peak clipping protection
    peak = np.max(np.abs(y_norm))
    if peak > max_peak:
        y_norm = y_norm * (max_peak / peak)

    return y_norm.astype(np.float32)

def extract_84_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extracts identical 84 acoustic features:
    40 MFCC mean + 40 MFCC std + RMS mean + RMS std + ZCR mean + ZCR std
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms, axis=1)
    rms_std = np.std(rms, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr, axis=1)
    zcr_std = np.std(zcr, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        rms_mean,
        rms_std,
        zcr_mean,
        zcr_std
    ]).astype(np.float64)

    return features

def extract_all_normalized_features(audio_files, baseline_df):
    """
    Extracts features for all audio files with loudness normalization applied.
    """
    norm_features_file = OUTPUT_DIR / "normalized_features.csv"
    if norm_features_file.exists():
        print(f"Loading existing normalized features from {norm_features_file}...")
        df_norm = pd.read_csv(norm_features_file)
        if len(df_norm) == len(baseline_df):
            return df_norm

    print("Extracting 84 features from normalized audio for all 1,440 recordings...")
    t0 = time.time()
    
    # Pre-warm JIT
    y_dummy = np.zeros(16000, dtype=np.float32)
    librosa.feature.mfcc(y=y_dummy, sr=16000, n_mfcc=40)

    rows = []
    for idx, path in enumerate(audio_files):
        parts = path.stem.split("-")
        if len(parts) != 7 or parts[2] not in EMOTION_CODES:
            continue

        y, sr = librosa.load(str(path), sr=16000, mono=True)
        y_norm = normalize_audio(y)
        feats = extract_84_features(y_norm, sr=sr)

        row = {f"feature_{i+1}": val for i, val in enumerate(feats)}
        row["filename"] = path.name
        row["emotion"] = EMOTION_CODES[parts[2]]
        row["actor_id"] = int(parts[6])
        rows.append(row)

        if (idx + 1) % 300 == 0:
            print(f"  Processed {idx + 1}/{len(audio_files)} files ({time.time()-t0:.1f}s)...")

    df_norm = pd.DataFrame(rows)
    # Align rows to match baseline_df
    df_norm = df_norm.set_index("filename").loc[baseline_df["filename"]].reset_index()
    df_norm.to_csv(norm_features_file, index=False)
    print(f"Extracted and saved {len(df_norm)} normalized recordings to {norm_features_file} in {time.time()-t0:.1f}s.")
    return df_norm

def create_svm():
    base_svm = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        )
    )
    return CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=3,
        ensemble=False
    )

def create_hgb():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )
    )

def predict_ensemble(svm, hgb, X):
    p_svm = svm.predict_proba(X)
    p_hgb = hgb.predict_proba(X)
    p = 0.5 * p_svm + 0.5 * p_hgb
    return p

def main():
    print("=" * 70)
    print("   EXPERIMENT: LOUDNESS NORMALIZATION & VOLUME AUGMENTATION IN SER   ")
    print("=" * 70)

    # 1. Load baseline features
    baseline_file = ROOT / "notebooks" / "advanced_features.csv"
    assert baseline_file.exists(), f"Baseline features file missing: {baseline_file}"
    df_base = pd.read_csv(baseline_file)
    feature_cols = [f"feature_{i}" for i in range(1, 85)]

    audio_files = sorted(DATASET_DIR.rglob("*.wav"))
    assert len(audio_files) == 1440, f"Expected 1440 WAV files, found {len(audio_files)}"

    # 2. Extract normalized features
    df_norm = extract_all_normalized_features(audio_files, df_base)

    # 3. Align datasets and label encoding
    encoder = LabelEncoder()
    y_all = encoder.fit_transform(df_base["emotion"].to_numpy())
    groups_all = df_base["actor_id"].to_numpy()
    emotions = list(encoder.classes_)

    # 4. Same train / test split as original Step 13
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    dev_idx, test_idx = next(splitter.split(df_base, y_all, groups=groups_all))

    dev_actors = sorted(set(groups_all[dev_idx]))
    test_actors = sorted(set(groups_all[test_idx]))

    print("\n--- Data Partitioning ---")
    print(f"Total recordings: {len(df_base)}")
    print(f"Development recordings ({len(dev_actors)} actors): {len(dev_idx)}")
    print(f"Development actors: {dev_actors}")
    print(f"Excluded test recordings ({len(test_actors)} actors): {len(test_idx)}")
    print(f"Excluded test actors: {test_actors}")
    assert set(dev_actors).isdisjoint(set(test_actors))

    X_base = df_base[feature_cols].to_numpy(dtype=float)
    X_norm = df_norm[feature_cols].to_numpy(dtype=float)

    # Split development and test
    X_base_dev, y_dev, groups_dev = X_base[dev_idx], y_all[dev_idx], groups_all[dev_idx]
    X_base_test, y_test = X_base[test_idx], y_all[test_idx]

    X_norm_dev = X_norm[dev_idx]
    X_norm_test = X_norm[test_idx]

    # Map filename to file path for fast lookup during gain augmentation
    file_map = {f.name: f for f in audio_files}
    dev_filenames = df_base["filename"].iloc[dev_idx].tolist()

    # Pre-extract volume-augmented training features (gains: 0.5x, 0.75x, 1.5x, 2.0x)
    aug_cache_file = OUTPUT_DIR / "augmented_dev_features.joblib"
    if aug_cache_file.exists():
        print(f"\nLoading cached volume-augmented training features from {aug_cache_file}...")
        aug_data = joblib.load(aug_cache_file)
    else:
        print("\nPre-extracting volume-augmented features for development set (gains: 0.5x, 0.75x, 1.5x, 2.0x)...")
        t0 = time.time()
        aug_data = {}
        gain_levels = [0.5, 0.75, 1.5, 2.0]
        for g in gain_levels:
            aug_data[g] = np.zeros_like(X_base_dev)
            for i, fname in enumerate(dev_filenames):
                p = file_map[fname]
                y_raw, sr = librosa.load(str(p), sr=16000, mono=True)
                y_g = y_raw * g
                aug_data[g][i] = extract_84_features(y_g, sr=sr)
        joblib.dump(aug_data, aug_cache_file)
        print(f"Pre-extracted and cached gain augmentations in {time.time()-t0:.1f}s.")

    # -------------------------------------------------------------------------
    # 5. Speaker-Separated 5-Fold Cross-Validation on Development Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 1: 5-FOLD SPEAKER-SEPARATED CROSS-VALIDATION (DEVELOPMENT SET) ")
    print("=" * 70)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(np.zeros(len(dev_idx)), y_dev, groups_dev))

    cv_results = []
    models_to_test = ["Baseline", "Loudness_Normalized", "Volume_Augmented"]

    oof_preds = {m: np.zeros((len(dev_idx), len(emotions))) for m in models_to_test}

    for model_type in models_to_test:
        print(f"\n>>> Running 5-Fold CV for: {model_type} <<<")
        fold_accs = []
        fold_f1s = []

        for fold_num, (tr_idx, val_idx) in enumerate(folds, 1):
            val_actors = sorted(set(groups_dev[val_idx]))

            if model_type == "Baseline":
                X_tr, y_tr = X_base_dev[tr_idx], y_dev[tr_idx]
                X_val, y_val = X_base_dev[val_idx], y_dev[val_idx]
            elif model_type == "Loudness_Normalized":
                X_tr, y_tr = X_norm_dev[tr_idx], y_dev[tr_idx]
                X_val, y_val = X_norm_dev[val_idx], y_dev[val_idx]
            elif model_type == "Volume_Augmented":
                # Training fold has original + augmented variants; Validation fold is clean original!
                aug_X_list = [X_base_dev[tr_idx]]
                aug_y_list = [y_dev[tr_idx]]
                # Add 0.5x and 2.0x augmentations to training set
                for g in [0.5, 2.0]:
                    aug_X_list.append(aug_data[g][tr_idx])
                    aug_y_list.append(y_dev[tr_idx])
                X_tr = np.vstack(aug_X_list)
                y_tr = np.concatenate(aug_y_list)
                X_val, y_val = X_base_dev[val_idx], y_dev[val_idx]

            svm = create_svm()
            hgb = create_hgb()

            svm.fit(X_tr, y_tr)
            hgb.fit(X_tr, y_tr)

            p_val = predict_ensemble(svm, hgb, X_val)
            oof_preds[model_type][val_idx] = p_val

            pred_val = np.argmax(p_val, axis=1)
            acc = accuracy_score(y_val, pred_val)
            f1 = f1_score(y_val, pred_val, average="macro")

            fold_accs.append(acc)
            fold_f1s.append(f1)
            print(f"  Fold {fold_num} (Val actors {val_actors}): Accuracy = {acc*100:5.2f}%, Macro F1 = {f1*100:5.2f}%")

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)

        cv_results.append({
            "model_type": model_type,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1
        })
        print(f"  => {model_type} Mean CV: Accuracy = {mean_acc*100:.2f}% ± {std_acc*100:.2f}%, Macro F1 = {mean_f1*100:.2f}% ± {std_f1*100:.2f}%")

    df_cv_summary = pd.DataFrame(cv_results)
    df_cv_summary.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False)
    print("\n--- Cross-Validation Summary Table ---")
    print(df_cv_summary.to_string(index=False))

    # Per-emotion recall from Out-of-Fold predictions
    recall_rows = []
    for model_type in models_to_test:
        preds = np.argmax(oof_preds[model_type], axis=1)
        recalls = recall_score(y_dev, preds, average=None)
        for em, rec in zip(emotions, recalls):
            recall_rows.append({
                "model_type": model_type,
                "emotion": em,
                "recall": rec
            })

    df_recall = pd.DataFrame(recall_rows)
    df_recall.to_csv(OUTPUT_DIR / "oof_per_emotion_recall.csv", index=False)

    df_recall_pivot = df_recall.pivot(index="emotion", columns="model_type", values="recall")
    print("\n--- Out-of-Fold Per-Emotion Recall Comparison ---")
    print((df_recall_pivot * 100).round(2).to_string())

    # -------------------------------------------------------------------------
    # 6. Train Full Development Models for Independent & Sensitivity Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2: FULL DEVELOPMENT TRAINING & EVALUATION ON EXCLUDED ACTORS ")
    print("=" * 70)

    trained_models = {}

    # Model A: Baseline
    print("Training Full Baseline Model...")
    svm_base = create_svm().fit(X_base_dev, y_dev)
    hgb_base = create_hgb().fit(X_base_dev, y_dev)
    trained_models["Baseline"] = (svm_base, hgb_base)

    # Model B: Loudness Normalized
    print("Training Full Loudness Normalized Model...")
    svm_norm = create_svm().fit(X_norm_dev, y_dev)
    hgb_norm = create_hgb().fit(X_norm_dev, y_dev)
    trained_models["Loudness_Normalized"] = (svm_norm, hgb_norm)

    # Model C: Volume Augmented
    print("Training Full Volume Augmented Model...")
    aug_X_dev_list = [X_base_dev]
    aug_y_dev_list = [y_dev]
    for g in [0.5, 2.0]:
        aug_X_dev_list.append(aug_data[g])
        aug_y_dev_list.append(y_dev)
    X_aug_dev = np.vstack(aug_X_dev_list)
    y_aug_dev = np.concatenate(aug_y_dev_list)

    svm_aug = create_svm().fit(X_aug_dev, y_aug_dev)
    hgb_aug = create_hgb().fit(X_aug_dev, y_aug_dev)
    trained_models["Volume_Augmented"] = (svm_aug, hgb_aug)

    # Evaluate on Unseen Excluded Actors at Baseline (1.0x)
    test_eval_results = []
    test_preds_baseline = {}

    for model_name, (svm_m, hgb_m) in trained_models.items():
        if model_name == "Loudness_Normalized":
            X_test_eval = X_norm_test
        else:
            X_test_eval = X_base_test

        p_test = predict_ensemble(svm_m, hgb_m, X_test_eval)
        pred_test = np.argmax(p_test, axis=1)
        test_preds_baseline[model_name] = pred_test

        acc = accuracy_score(y_test, pred_test)
        f1 = f1_score(y_test, pred_test, average="macro")
        rec = recall_score(y_test, pred_test, average=None)

        em_rec_dict = {f"recall_{em}": round(r*100, 2) for em, r in zip(emotions, rec)}
        res_row = {
            "model": model_name,
            "test_accuracy": round(acc * 100, 2),
            "test_macro_f1": round(f1 * 100, 2),
            **em_rec_dict
        }
        test_eval_results.append(res_row)

    df_test_eval = pd.DataFrame(test_eval_results)
    df_test_eval.to_csv(OUTPUT_DIR / "excluded_actors_test_results.csv", index=False)
    print("\n--- Independent Evaluation on Excluded Actors (Actors 1, 9, 12, 17, 19) at 1.0x ---")
    print(df_test_eval[["model", "test_accuracy", "test_macro_f1", "recall_happy", "recall_angry", "recall_calm"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # 7. Volume Sensitivity Benchmark on Excluded Actors
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 3: VOLUME SENSITIVITY BENCHMARK (0.5x, 1.0x, 2.0x, 3.0x GAIN) ")
    print("=" * 70)

    test_filenames = df_base["filename"].iloc[test_idx].tolist()
    gain_levels = [0.5, 1.0, 2.0, 3.0]
    sensitivity_results = []

    # Extract test features at each gain level
    test_features_by_gain = {}
    print("Extracting test audio features across gain levels (0.5x, 1.0x, 2.0x, 3.0x)...")
    t0 = time.time()
    for g in gain_levels:
        raw_feats = np.zeros((len(test_idx), 84))
        norm_feats = np.zeros((len(test_idx), 84))
        for i, fname in enumerate(test_filenames):
            p = file_map[fname]
            y_raw, sr = librosa.load(str(p), sr=16000, mono=True)
            y_scaled = y_raw * g
            raw_feats[i] = extract_84_features(y_scaled, sr=sr)
            # For normalized model, the normalized audio is extracted
            y_norm = normalize_audio(y_scaled)
            norm_feats[i] = extract_84_features(y_norm, sr=sr)
        test_features_by_gain[g] = {
            "raw": raw_feats,
            "norm": norm_feats
        }
    print(f"Extracted all gain level test features in {time.time()-t0:.1f}s.")

    for g in gain_levels:
        print(f"\n--- Testing Gain Level: {g:.2f}x ({'+' if g >= 1.0 else ''}{20*np.log10(g):.1f} dB) ---")
        for model_name, (svm_m, hgb_m) in trained_models.items():
            if model_name == "Loudness_Normalized":
                X_eval = test_features_by_gain[g]["norm"]
            else:
                X_eval = test_features_by_gain[g]["raw"]

            p = predict_ensemble(svm_m, hgb_m, X_eval)
            preds = np.argmax(p, axis=1)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            rec = recall_score(y_test, preds, average=None)
            rec_dict = {em: r for em, r in zip(emotions, rec)}

            # Measure agreement with the model's own 1.0x predictions
            agreement = np.mean(preds == test_preds_baseline[model_name])

            # Specifically check how many Happy recordings were classified as Angry
            happy_idx = np.where(y_test == emotions.index("happy"))[0]
            happy_preds = preds[happy_idx]
            happy_as_angry = np.sum(happy_preds == emotions.index("angry"))
            happy_as_angry_pct = (happy_as_angry / len(happy_idx)) * 100

            sensitivity_results.append({
                "gain": g,
                "gain_db": round(20 * np.log10(g), 1),
                "model": model_name,
                "accuracy": round(acc * 100, 2),
                "macro_f1": round(f1 * 100, 2),
                "agreement_with_1x": round(agreement * 100, 2),
                "happy_recall": round(rec_dict["happy"] * 100, 2),
                "angry_recall": round(rec_dict["angry"] * 100, 2),
                "happy_misclassified_as_angry_count": int(happy_as_angry),
                "happy_misclassified_as_angry_pct": round(happy_as_angry_pct, 2)
            })

            print(f"  {model_name:<20}: Acc = {acc*100:5.2f}%, F1 = {f1*100:5.2f}%, Agreement = {agreement*100:5.1f}%, Happy Recall = {rec_dict['happy']*100:5.1f}%, Happy->Angry = {happy_as_angry}/{len(happy_idx)} ({happy_as_angry_pct:.1f}%)")

    df_sens = pd.DataFrame(sensitivity_results)
    df_sens.to_csv(OUTPUT_DIR / "volume_sensitivity_results.csv", index=False)

    # Save models in experiment folder
    joblib.dump(trained_models["Baseline"][0], OUTPUT_DIR / "baseline_svm.joblib")
    joblib.dump(trained_models["Baseline"][1], OUTPUT_DIR / "baseline_hgb.joblib")
    joblib.dump(trained_models["Loudness_Normalized"][0], OUTPUT_DIR / "normalized_svm.joblib")
    joblib.dump(trained_models["Loudness_Normalized"][1], OUTPUT_DIR / "normalized_hgb.joblib")
    joblib.dump(trained_models["Volume_Augmented"][0], OUTPUT_DIR / "augmented_svm.joblib")
    joblib.dump(trained_models["Volume_Augmented"][1], OUTPUT_DIR / "augmented_hgb.joblib")
    joblib.dump(encoder, OUTPUT_DIR / "label_encoder.joblib")

    # Save full JSON summary
    summary_report = {
        "experiment_title": "SpeechSense Loudness Normalization & Volume Augmentation Experiment",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "RAVDESS (1440 audio recordings, 24 actors)",
        "features": "84 acoustic features (40 MFCC mean, 40 MFCC std, 2 RMS, 2 ZCR)",
        "development_actors": dev_actors,
        "excluded_test_actors": test_actors,
        "cross_validation_5fold": cv_results,
        "independent_test_evaluation": test_eval_results,
        "volume_sensitivity_evaluation": sensitivity_results
    }
    with open(OUTPUT_DIR / "experiment_report.json", "w") as f:
        json.dump(summary_report, f, indent=4)

    print("\n" + "=" * 70)
    print(" EXPERIMENT COMPLETE! ALL RESULTS SAVED TO models/normalization_experiment/")
    print("=" * 70)

if __name__ == "__main__":
    main()
