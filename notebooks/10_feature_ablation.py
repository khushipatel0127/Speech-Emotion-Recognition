
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "notebooks" / "advanced_features.csv"
OUTPUT = ROOT / "models" / "feature_ablation"

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# Feature positions in advanced_features.csv:
# 0:80   -> MFCC
# 80:82  -> RMS
# 82:84  -> ZCR
# 84:98  -> Spectral contrast
# 98:100 -> Pitch

FEATURE_GROUPS = {
    "MFCC": list(range(0, 80)),
    "RMS": list(range(80, 82)),
    "ZCR": list(range(82, 84)),
    "Spectral": list(range(84, 98)),
    "Pitch": list(range(98, 100))
}

COMBINATIONS = {
    "MFCC_only": ["MFCC"],
    "MFCC_RMS_ZCR": ["MFCC", "RMS", "ZCR"],
    "MFCC_Spectral": ["MFCC", "Spectral"],
    "MFCC_Pitch": ["MFCC", "Pitch"],
    "MFCC_RMS_ZCR_Spectral": [
        "MFCC", "RMS", "ZCR", "Spectral"
    ],
    "MFCC_RMS_ZCR_Pitch": [
        "MFCC", "RMS", "ZCR", "Pitch"
    ],
    "All_Advanced": [
        "MFCC", "RMS", "ZCR", "Spectral", "Pitch"
    ]
}


# --------------------------------------------------
# 2. Model definition
# --------------------------------------------------

def create_model():

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        )
    )


# --------------------------------------------------
# 3. Main experiment
# --------------------------------------------------

def main():

    df = pd.read_csv(INPUT_FILE)

    required = {"filename", "actor_id", "emotion"}

    if not required.issubset(df.columns):
        raise ValueError("Required metadata columns are missing.")

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames detected.")

    feature_cols = [
        f"feature_{i}"
        for i in range(1, 101)
    ]

    missing = [
        col for col in feature_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    X_all = df[feature_cols].to_numpy(dtype=float)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["emotion"])

    groups = df["actor_id"].to_numpy()

    class_ids = np.arange(len(encoder.classes_))

    if len(df) != 1440:
        raise ValueError(
            f"Expected 1440 recordings, found {len(df)}."
        )

    print("Dataset loaded successfully.")
    print("Recordings:", len(df))
    print("Features:", X_all.shape[1])

    # --------------------------------------------------
    # 4. Exclude previously reserved test actors
    # --------------------------------------------------

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    development_idx, test_idx = next(
        outer_split.split(X_all, y, groups=groups)
    )

    dev_actors = set(groups[development_idx])
    test_actors = set(groups[test_idx])

    assert dev_actors.isdisjoint(test_actors)

    X_dev = X_all[development_idx]
    y_dev = y[development_idx]
    groups_dev = groups[development_idx]

    print("Development recordings:", len(development_idx))
    print("Reserved test recordings:", len(test_idx))
    print("Actor separation verified.")

    # --------------------------------------------------
    # 5. Generate identical folds for all combinations
    # --------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = list(
        cv.split(X_dev, y_dev, groups_dev)
    )

    fold_results = []
    recall_results = []
    prediction_results = []

    # --------------------------------------------------
    # 6. Feature ablation
    # --------------------------------------------------

    for combo_name, group_names in COMBINATIONS.items():

        selected_indices = []

        for group_name in group_names:
            selected_indices.extend(
                FEATURE_GROUPS[group_name]
            )

        selected_indices = sorted(set(selected_indices))

        X = X_dev[:, selected_indices]

        print(f"\n========== {combo_name} ==========")
        print("Feature count:", X.shape[1])

        for fold_num, (train_idx, val_idx) in enumerate(
            folds, start=1
        ):

            train_actors = set(groups_dev[train_idx])
            val_actors = set(groups_dev[val_idx])

            assert train_actors.isdisjoint(val_actors)

            model = create_model()

            model.fit(
                X[train_idx],
                y_dev[train_idx]
            )

            predictions = model.predict(
                X[val_idx]
            )

            accuracy = accuracy_score(
                y_dev[val_idx],
                predictions
            )

            macro_f1 = f1_score(
                y_dev[val_idx],
                predictions,
                labels=class_ids,
                average="macro",
                zero_division=0
            )

            recalls = recall_score(
                y_dev[val_idx],
                predictions,
                labels=class_ids,
                average=None,
                zero_division=0
            )

            fold_results.append({
                "combination": combo_name,
                "fold": fold_num,
                "feature_count": X.shape[1],
                "validation_recordings": len(val_idx),
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

            for class_id, emotion in enumerate(
                encoder.classes_
            ):

                recall_results.append({
                    "combination": combo_name,
                    "fold": fold_num,
                    "emotion": emotion,
                    "recall": recalls[class_id],
                    "support": int(
                        np.sum(y_dev[val_idx] == class_id)
                    )
                })

            for idx, prediction in zip(
                val_idx, predictions
            ):

                original_idx = development_idx[idx]

                prediction_results.append({
                    "combination": combo_name,
                    "fold": fold_num,
                    "filename": df.iloc[original_idx]["filename"],
                    "actor_id": int(groups[original_idx]),
                    "actual": encoder.inverse_transform(
                        [y[original_idx]]
                    )[0],
                    "predicted": encoder.inverse_transform(
                        [prediction]
                    )[0],
                    "correct": bool(
                        y[original_idx] == prediction
                    )
                })

            print(
                f"Fold {fold_num}: "
                f"Accuracy={accuracy:.4f}, "
                f"Macro F1={macro_f1:.4f}"
            )

    # --------------------------------------------------
    # 7. Save fold-level results
    # --------------------------------------------------

    folds_df = pd.DataFrame(fold_results)

    folds_df.to_csv(
        OUTPUT / "fold_results.csv",
        index=False
    )

    recall_df = pd.DataFrame(recall_results)

    recall_df.to_csv(
        OUTPUT / "per_class_recall.csv",
        index=False
    )

    predictions_df = pd.DataFrame(prediction_results)

    predictions_df.to_csv(
        OUTPUT / "out_of_fold_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # 8. Cross-validation summary
    # --------------------------------------------------

    summary = (
        folds_df.groupby("combination")
        .agg(
            feature_count=("feature_count", "first"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std")
        )
        .reset_index()
        .sort_values(
            "mean_macro_f1",
            ascending=False
        )
    )

    summary.to_csv(
        OUTPUT / "ablation_summary.csv",
        index=False
    )

    print("\n========== FEATURE ABLATION SUMMARY ==========")
    print(summary.to_string(index=False))

    # --------------------------------------------------
    # 9. Recall summary
    # --------------------------------------------------

    recall_summary = (
        recall_df.groupby(
            ["combination", "emotion"]
        )
        .agg(
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            total_support=("support", "sum")
        )
        .reset_index()
    )

    recall_summary.to_csv(
        OUTPUT / "recall_summary.csv",
        index=False
    )

    print("\n========== NEUTRAL AND SAD RECALL ==========")

    important = recall_summary[
        recall_summary["emotion"].isin(
            ["neutral", "sad"]
        )
    ]

    print(
        important.sort_values(
            ["emotion", "mean_recall"],
            ascending=[True, False]
        ).to_string(index=False)
    )

    # --------------------------------------------------
    # 10. Plot comparison
    # --------------------------------------------------

    plot_df = summary.sort_values(
        "mean_macro_f1"
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(
        plot_df["combination"],
        plot_df["mean_macro_f1"],
        xerr=plot_df["std_macro_f1"],
        capsize=4
    )

    ax.set_xlabel("Mean macro F1")
    ax.set_title(
        "Feature Ablation - Actor-Grouped Cross-Validation"
    )

    ax.set_xlim(0, 1)

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "feature_ablation_comparison.png",
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------
    # 11. Confusion matrix for best combination
    # --------------------------------------------------

    best_combination = summary.iloc[0]["combination"]

    best_predictions = predictions_df[
        predictions_df["combination"] == best_combination
    ]

    cm = confusion_matrix(
        best_predictions["actual"],
        best_predictions["predicted"],
        labels=encoder.classes_
    )

    pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    ).to_csv(
        OUTPUT / "best_combination_confusion_matrix.csv"
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

    ax.set_title(
        f"Best Feature Combination: {best_combination}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "best_combination_confusion_matrix.png",
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------
    # 12. Save experiment metadata
    # --------------------------------------------------

    metadata = {
        "experiment": "Feature ablation",
        "cross_validation": "StratifiedGroupKFold",
        "folds": N_SPLITS,
        "random_state": RANDOM_STATE,
        "development_recordings": len(development_idx),
        "reserved_test_recordings": len(test_idx),
        "test_evaluated": False,
        "classifier": {
            "type": "SVM",
            "kernel": "rbf",
            "C": 10,
            "gamma": "scale",
            "class_weight": "balanced"
        },
        "feature_groups": FEATURE_GROUPS,
        "combinations": COMBINATIONS,
        "best_combination_by_mean_macro_f1": best_combination,
        "emotion_classes": encoder.classes_.tolist()
    }

    with open(
        OUTPUT / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n========== SELECTED FEATURE COMBINATION ==========")
    print("Best combination:", best_combination)
    print(
        "Mean macro F1:",
        round(float(summary.iloc[0]["mean_macro_f1"]), 4)
    )
    print(
        "Mean accuracy:",
        round(float(summary.iloc[0]["mean_accuracy"]), 4)
    )

    print("\nExperiment completed successfully.")
    print("Results saved to models/feature_ablation/")
    print("Reserved test actors were NOT evaluated.")


if __name__ == "__main__":
    main()
