
from pathlib import Path
import json

import joblib
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
OUTPUT = ROOT / "models" / "error_analysis"

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

DATASETS = {
    "MFCC": {
        "file": NOTEBOOKS / "features.csv",
        "prefix": "mfcc_"
    },
    "Advanced": {
        "file": NOTEBOOKS / "advanced_features.csv",
        "prefix": "feature_"
    }
}


# --------------------------------------------------
# 2. Load and verify datasets
# --------------------------------------------------

datasets = {}

for name, config in DATASETS.items():

    df = pd.read_csv(config["file"])

    required = {"filename", "actor_id", "emotion"}

    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {name}")

    if df["filename"].duplicated().any():
        raise ValueError(f"Duplicate filenames in {name}")

    feature_cols = [
        col for col in df.columns
        if col.startswith(config["prefix"])
    ]

    if not feature_cols:
        raise ValueError(f"No features found in {name}")

    datasets[name] = {
        "df": df,
        "features": feature_cols
    }

base = datasets["MFCC"]["df"].copy()

if len(base) != 1440:
    raise ValueError("Expected 1440 recordings")

# Align all datasets by filename.
# This prevents comparing different recordings accidentally.
base_files = base["filename"].tolist()

for name in datasets:

    df = datasets[name]["df"]

    if set(df["filename"]) != set(base_files):
        raise ValueError(
            f"Filename mismatch in {name}"
        )

    df = df.set_index("filename").loc[base_files]
    df = df.reset_index()

    if not np.array_equal(
        df["actor_id"].to_numpy(),
        base["actor_id"].to_numpy()
    ):
        raise ValueError(f"Actor mismatch in {name}")

    if not np.array_equal(
        df["emotion"].to_numpy(),
        base["emotion"].to_numpy()
    ):
        raise ValueError(f"Emotion mismatch in {name}")

    datasets[name]["df"] = df

print("Dataset alignment verified.")
print("Recordings:", len(base))


# --------------------------------------------------
# 3. Actor-based split
# --------------------------------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(
    base["emotion"].to_numpy()
)

groups = base["actor_id"].to_numpy()

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(
    split1.split(base, y, groups=groups)
)

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

train_rel, val_rel = next(
    split2.split(
        base.iloc[train_val_idx],
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

print("Training:", len(train_idx))
print("Validation:", len(val_idx))
print("Reserved test:", len(test_idx))
print("Actor separation verified.")


# --------------------------------------------------
# 4. Train comparable SVM models
# --------------------------------------------------

all_predictions = {}
summary = []
class_ids = np.arange(len(encoder.classes_))

for name, data in datasets.items():

    print(f"\n========== {name} ==========")

    df = data["df"]
    feature_cols = data["features"]

    X = df[feature_cols].to_numpy(dtype=float)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        )
    )

    model.fit(X[train_idx], y[train_idx])

    predictions = model.predict(X[val_idx])

    all_predictions[name] = predictions

    accuracy = accuracy_score(
        y[val_idx], predictions
    )

    macro_f1 = f1_score(
        y[val_idx],
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0
    )

    print("Validation accuracy:", round(accuracy, 4))
    print("Validation macro F1:", round(macro_f1, 4))

    summary.append({
        "model": name,
        "feature_count": len(feature_cols),
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1
    })

    # --------------------------------------------------
    # 5. Classification report
    # --------------------------------------------------

    report = classification_report(
        y[val_idx],
        predictions,
        labels=class_ids,
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        OUTPUT / f"{name.lower()}_classification_report.csv"
    )

    print("\nPer-class metrics:")
    print(report_df.round(3).to_string())

    # --------------------------------------------------
    # 6. Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y[val_idx],
        predictions,
        labels=class_ids
    )

    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    ).to_csv(
        OUTPUT / f"{name.lower()}_confusion_matrix.csv"
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

    ax.set_title(f"{name} - Validation Confusion Matrix")
    fig.tight_layout()

    fig.savefig(
        OUTPUT / f"{name.lower()}_confusion_matrix.png",
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------
    # 7. Prediction distribution
    # --------------------------------------------------

    predicted_counts = pd.Series(
        encoder.inverse_transform(predictions)
    ).value_counts()

    actual_counts = pd.Series(
        encoder.inverse_transform(y[val_idx])
    ).value_counts()

    distribution = pd.DataFrame({
        "actual": actual_counts,
        "predicted": predicted_counts
    }).fillna(0).astype(int)

    distribution.to_csv(
        OUTPUT / f"{name.lower()}_prediction_distribution.csv"
    )

    # --------------------------------------------------
    # 8. Per-actor accuracy
    # --------------------------------------------------

    actor_rows = []

    for actor in sorted(val_actors):

        mask = groups[val_idx] == actor

        actor_rows.append({
            "actor_id": int(actor),
            "recordings": int(mask.sum()),
            "accuracy": accuracy_score(
                y[val_idx][mask],
                predictions[mask]
            ),
            "macro_f1": f1_score(
                y[val_idx][mask],
                predictions[mask],
                labels=class_ids,
                average="macro",
                zero_division=0
            )
        })

    actor_df = pd.DataFrame(actor_rows)

    actor_df.to_csv(
        OUTPUT / f"{name.lower()}_actor_performance.csv",
        index=False
    )

    print("\nActor performance:")
    print(actor_df.to_string(index=False))


# --------------------------------------------------
# 9. Compare predictions on identical recordings
# --------------------------------------------------

comparison = pd.DataFrame({
    "filename": base.iloc[val_idx]["filename"].to_numpy(),
    "actor_id": groups[val_idx],
    "actual": encoder.inverse_transform(y[val_idx])
})

for name, predictions in all_predictions.items():

    comparison[f"{name}_prediction"] = (
        encoder.inverse_transform(predictions)
    )

    comparison[f"{name}_correct"] = (
        predictions == y[val_idx]
    )

comparison["both_wrong"] = (
    ~comparison["MFCC_correct"]
    & ~comparison["Advanced_correct"]
)

comparison["advanced_only_correct"] = (
    ~comparison["MFCC_correct"]
    & comparison["Advanced_correct"]
)

comparison["mfcc_only_correct"] = (
    comparison["MFCC_correct"]
    & ~comparison["Advanced_correct"]
)

comparison.to_csv(
    OUTPUT / "recording_level_comparison.csv",
    index=False
)

print("\n========== ERROR COMPARISON ==========")

print(
    "Both models wrong:",
    int(comparison["both_wrong"].sum())
)

print(
    "Only advanced correct:",
    int(comparison["advanced_only_correct"].sum())
)

print(
    "Only MFCC correct:",
    int(comparison["mfcc_only_correct"].sum())
)


# --------------------------------------------------
# 10. Most common emotion confusions
# --------------------------------------------------

for name, predictions in all_predictions.items():

    errors = pd.DataFrame({
        "actual": encoder.inverse_transform(y[val_idx]),
        "predicted": encoder.inverse_transform(predictions)
    })

    errors = errors[
        errors["actual"] != errors["predicted"]
    ]

    confusion_pairs = (
        errors.groupby(["actual", "predicted"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    confusion_pairs.to_csv(
        OUTPUT / f"{name.lower()}_top_confusions.csv",
        index=False
    )

    print(f"\n{name} - Top confusion pairs:")
    print(confusion_pairs.head(10).to_string(index=False))


# --------------------------------------------------
# 11. Save summary and metadata
# --------------------------------------------------

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT / "model_comparison.csv",
    index=False
)

metadata = {
    "experiment": "Validation error analysis",
    "sample_rate": 16000,
    "random_state": RANDOM_STATE,
    "training_recordings": len(train_idx),
    "validation_recordings": len(val_idx),
    "reserved_test_recordings": len(test_idx),
    "selection_policy": "Validation-only diagnostics",
    "test_evaluated": False,
    "models": [
        "MFCC SVM",
        "Advanced SVM"
    ],
    "emotion_classes": encoder.classes_.tolist()
}

with open(
    OUTPUT / "metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\n========== FINAL COMPARISON ==========")
print(summary_df.to_string(index=False))

print("\nError analysis completed.")
print("Saved results to models/error_analysis/")
print("Reserved test set was NOT evaluated.")
