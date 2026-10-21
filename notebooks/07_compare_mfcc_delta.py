
from pathlib import Path
import json

import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
OUTPUT = ROOT / "models" / "delta_experiment"

OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
N_MFCC = 40
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Locate audio recordings
# --------------------------------------------------

def locate_audio(row, wav_by_name):
    filename = Path(str(row["filename"])).name
    matches = wav_by_name.get(filename, [])

    if len(matches) != 1:
        raise ValueError(
            f"Expected one WAV for {filename}, "
            f"found {len(matches)}. Check dataset paths."
        )

    return matches[0]


# --------------------------------------------------
# 3. Extract MFCC, delta and delta-delta features
# --------------------------------------------------

def extract_features(audio_path):
    signal, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if signal.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def statistics(matrix):
        return np.concatenate([
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1)
        ])

    mfcc_features = statistics(mfcc)
    delta_features = statistics(delta)
    delta2_features = statistics(delta2)

    original = mfcc_features

    with_delta = np.concatenate([
        mfcc_features,
        delta_features
    ])

    with_delta2 = np.concatenate([
        mfcc_features,
        delta_features,
        delta2_features
    ])

    return original, with_delta, with_delta2


# --------------------------------------------------
# 4. Load dataset information
# --------------------------------------------------

def main():

    data_file = NOTEBOOKS / "features.csv"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Missing file: {data_file}"
        )

    df = pd.read_csv(data_file)

    required = {"filename", "actor_id", "emotion"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required - set(df.columns)}"
        )

    if len(df) != 1440:
        raise ValueError(
            f"Expected 1440 recordings, found {len(df)}"
        )

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found.")

    print("Total recordings:", len(df))

    wav_by_name = {}

    for path in ROOT.rglob("*.wav"):
        if ".venv" not in path.parts:
            wav_by_name.setdefault(
                path.name, []
            ).append(path)

    print("Locating audio recordings...")

    feature_sets = {
        "mfcc": [],
        "mfcc_delta": [],
        "mfcc_delta_delta2": []
    }

    audio_paths = []

    # --------------------------------------------------
    # 5. Extract all feature representations
    # --------------------------------------------------

    for i, row in df.iterrows():

        audio_path = locate_audio(row, wav_by_name)

        audio_paths.append(
            str(audio_path.relative_to(ROOT))
        )

        extracted = extract_features(audio_path)

        for feature_name, values in zip(
            feature_sets.keys(), extracted
        ):
            feature_sets[feature_name].append(values)

        if len(audio_paths) % 100 == 0:
            print(
                f"Processed {len(audio_paths)}/{len(df)}",
                flush=True
            )

    matrices = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in feature_sets.items()
    }

    expected_counts = {
        "mfcc": 80,
        "mfcc_delta": 160,
        "mfcc_delta_delta2": 240
    }

    for name, X in matrices.items():

        if X.shape != (len(df), expected_counts[name]):
            raise ValueError(
                f"Unexpected shape for {name}: {X.shape}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Invalid feature values in {name}"
            )

    print("\nFeature extraction completed.")

    for name, X in matrices.items():
        print(name, ":", X.shape)

    # --------------------------------------------------
    # 6. Encode labels
    # --------------------------------------------------

    y_text = (
        df["emotion"]
        .astype(str)
        .str.lower()
        .to_numpy()
    )

    groups = df["actor_id"].to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    print("\nEmotion classes:")
    print(list(encoder.classes_))

    # --------------------------------------------------
    # 7. Preserve original actor-based split
    # --------------------------------------------------

    split1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_val_idx, test_idx = next(
        split1.split(
            matrices["mfcc"],
            y,
            groups=groups
        )
    )

    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    train_rel, val_rel = next(
        split2.split(
            matrices["mfcc"][train_val_idx],
            y[train_val_idx],
            groups=groups[train_val_idx]
        )
    )

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    train_actors = set(groups[train_idx])
    val_actors = set(groups[val_idx])
    test_actors = set(groups[test_idx])

    assert train_actors.isdisjoint(val_actors)
    assert train_actors.isdisjoint(test_actors)
    assert val_actors.isdisjoint(test_actors)

    print("\nTraining:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Reserved test:", len(test_idx))
    print("Actor separation verified.")

    # Save split information
    split_names = np.full(
        len(df),
        "test_reserved",
        dtype=object
    )

    split_names[train_idx] = "train"
    split_names[val_idx] = "validation"

    pd.DataFrame({
        "filename": df["filename"],
        "actor_id": groups,
        "emotion": y_text,
        "split": split_names,
        "audio_path": audio_paths
    }).to_csv(
        OUTPUT / "split_manifest.csv",
        index=False
    )

    np.savez_compressed(
        OUTPUT / "feature_matrices.npz",
        **matrices
    )

    # --------------------------------------------------
    # 8. Train and compare models
    # --------------------------------------------------

    results = []
    best = None

    class_ids = np.arange(len(encoder.classes_))

    for feature_name, X in matrices.items():

        models = {
            "SVM": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    class_weight="balanced"
                )
            ),

            "Random Forest": make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        }

        for model_name, model in models.items():

            print(
                f"\nTraining {feature_name} + {model_name}"
            )

            model.fit(
                X[train_idx],
                y[train_idx]
            )

            val_pred = model.predict(X[val_idx])

            val_accuracy = accuracy_score(
                y[val_idx],
                val_pred
            )

            val_macro_f1 = f1_score(
                y[val_idx],
                val_pred,
                average="macro",
                labels=class_ids,
                zero_division=0
            )

            print(
                f"Validation accuracy: {val_accuracy:.4f}"
            )
            print(
                f"Validation macro F1: {val_macro_f1:.4f}"
            )

            results.append({
                "feature_set": feature_name,
                "feature_count": X.shape[1],
                "model": model_name,
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": val_macro_f1
            })

            score = (val_macro_f1, val_accuracy)

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "model": model,
                    "feature_set": feature_name,
                    "model_name": model_name,
                    "predictions": val_pred
                }

    # --------------------------------------------------
    # 9. Save model comparison
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "validation_macro_f1",
            "validation_accuracy"
        ],
        ascending=False
    )

    results_df.to_csv(
        OUTPUT / "validation_comparison.csv",
        index=False
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\n========== SELECTED MODEL ==========")
    print("Feature set:", best["feature_set"])
    print("Classifier:", best["model_name"])
    print("Validation macro F1:", best["score"][0])
    print("Validation accuracy:", best["score"][1])

    # --------------------------------------------------
    # 10. Classification report
    # --------------------------------------------------

    report = classification_report(
        y[val_idx],
        best["predictions"],
        labels=class_ids,
        target_names=encoder.classes_,
        zero_division=0
    )

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)

    report_dict = classification_report(
        y[val_idx],
        best["predictions"],
        labels=class_ids,
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        OUTPUT / "selected_validation_classification_report.csv"
    )

    # --------------------------------------------------
    # 11. Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y[val_idx],
        best["predictions"],
        labels=class_ids
    )

    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    ).to_csv(
        OUTPUT / "selected_validation_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    ).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "selected_validation_confusion_matrix.png",
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------
    # 12. Save selected model and metadata
    # --------------------------------------------------

    joblib.dump(
        best["model"],
        OUTPUT / "selected_validation_model.joblib"
    )

    joblib.dump(
        encoder,
        OUTPUT / "label_encoder.joblib"
    )

    metadata = {
        "feature_set": best["feature_set"],
        "feature_count": expected_counts[best["feature_set"]],
        "model_name": best["model_name"],
        "validation_macro_f1": float(best["score"][0]),
        "validation_accuracy": float(best["score"][1]),
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": N_MFCC,
        "feature_order": (
            "MFCC means, MFCC std, "
            "delta means, delta std, "
            "delta-delta means, delta-delta std"
        ),
        "split_seed": RANDOM_STATE,
        "selection_metric": "validation macro F1, then accuracy",
        "test_status": "reserved, not evaluated",
        "emotion_classes": encoder.classes_.tolist()
    }

    with open(
        OUTPUT / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print("Results saved in models/delta_experiment/")
    print("Reserved test set was NOT evaluated.")


if __name__ == "__main__":
    main()
