
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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
OUTPUT = ROOT / "models" / "group_cross_validation"

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

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
# 2. Load and align datasets
# --------------------------------------------------

def load_datasets():

    datasets = {}

    for name, config in DATASETS.items():

        df = pd.read_csv(config["file"])

        required = {
            "filename",
            "actor_id",
            "emotion"
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing columns in {name}"
            )

        if df["filename"].duplicated().any():
            raise ValueError(
                f"Duplicate filenames in {name}"
            )

        feature_cols = [
            col for col in df.columns
            if col.startswith(config["prefix"])
        ]

        datasets[name] = {
            "df": df,
            "feature_cols": feature_cols
        }

    base = datasets["MFCC"]["df"].copy()

    if len(base) != 1440:
        raise ValueError(
            f"Expected 1440 recordings, found {len(base)}"
        )

    base_files = base["filename"].tolist()

    for name, data in datasets.items():

        df = data["df"]

        if set(df["filename"]) != set(base_files):
            raise ValueError(
                f"Filename mismatch in {name}"
            )

        # Align rows without repeatedly inserting columns
        aligned = (
            df.set_index("filename")
            .loc[base_files]
            .reset_index()
        )

        if not np.array_equal(
            aligned["actor_id"].to_numpy(),
            base["actor_id"].to_numpy()
        ):
            raise ValueError(
                f"Actor mismatch in {name}"
            )

        if not np.array_equal(
            aligned["emotion"].to_numpy(),
            base["emotion"].to_numpy()
        ):
            raise ValueError(
                f"Emotion mismatch in {name}"
            )

        datasets[name]["df"] = aligned

    return datasets, base


# --------------------------------------------------
# 3. Create original reserved-test split
# --------------------------------------------------

def create_actor_split(base, y, groups):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    development_idx, test_idx = next(
        splitter.split(base, y, groups=groups)
    )

    development_actors = set(groups[development_idx])
    test_actors = set(groups[test_idx])

    assert development_actors.isdisjoint(test_actors)

    return development_idx, test_idx


# --------------------------------------------------
# 4. Create fixed SVM pipeline
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
# 5. Main experiment
# --------------------------------------------------

def main():

    datasets, base = load_datasets()

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        base["emotion"].to_numpy()
    )

    groups = base["actor_id"].to_numpy()

    class_ids = np.arange(
        len(encoder.classes_)
    )

    development_idx, test_idx = create_actor_split(
        base, y, groups
    )

    print("Dataset alignment verified.")
    print("Total recordings:", len(base))
    print("Development recordings:", len(development_idx))
    print("Reserved test recordings:", len(test_idx))

    print(
        "Development actors:",
        sorted(set(groups[development_idx]))
    )

    print(
        "Reserved test actors:",
        sorted(set(groups[test_idx]))
    )

    # Cross-validation uses development actors only
    y_dev = y[development_idx]
    groups_dev = groups[development_idx]

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Generate folds once and reuse them for both models
    folds = list(
        cv.split(
            np.zeros(len(development_idx)),
            y_dev,
            groups_dev
        )
    )

    fold_rows = []
    recall_rows = []
    prediction_rows = []
    confusion_totals = {}

    for name, data in datasets.items():

        print(f"\n========== {name} ==========")

        X = data["df"][
            data["feature_cols"]
        ].to_numpy(dtype=float)

        X_dev = X[development_idx]

        all_true = []
        all_pred = []

        for fold_number, (train_idx, val_idx) in enumerate(
            folds,
            start=1
        ):

            train_actors = set(
                groups_dev[train_idx]
            )

            val_actors = set(
                groups_dev[val_idx]
            )

            assert train_actors.isdisjoint(val_actors)

            model = create_model()

            model.fit(
                X_dev[train_idx],
                y_dev[train_idx]
            )

            predictions = model.predict(
                X_dev[val_idx]
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

            fold_rows.append({
                "feature_set": name,
                "fold": fold_number,
                "training_recordings": len(train_idx),
                "validation_recordings": len(val_idx),
                "training_actors": len(train_actors),
                "validation_actors": len(val_actors),
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

            for emotion, recall in zip(
                encoder.classes_,
                recalls
            ):

                recall_rows.append({
                    "feature_set": name,
                    "fold": fold_number,
                    "emotion": emotion,
                    "recall": recall,
                    "support": int(
                        np.sum(
                            y_dev[val_idx] ==
                            encoder.transform([emotion])[0]
                        )
                    )
                })

            for local_idx, pred in zip(
                val_idx, predictions
            ):

                original_idx = development_idx[local_idx]

                prediction_rows.append({
                    "feature_set": name,
                    "fold": fold_number,
                    "filename": base.iloc[original_idx]["filename"],
                    "actor_id": int(groups[original_idx]),
                    "actual": encoder.inverse_transform(
                        [y[original_idx]]
                    )[0],
                    "predicted": encoder.inverse_transform(
                        [pred]
                    )[0],
                    "correct": bool(
                        y[original_idx] == pred
                    )
                })

            all_true.extend(
                y_dev[val_idx].tolist()
            )

            all_pred.extend(
                predictions.tolist()
            )

            print(
                f"Fold {fold_number}: "
                f"Accuracy={accuracy:.4f}, "
                f"Macro F1={macro_f1:.4f}, "
                f"Validation actors={sorted(val_actors)}"
            )

        cm = confusion_matrix(
            all_true,
            all_pred,
            labels=class_ids
        )

        confusion_totals[name] = cm

        pd.DataFrame(
            cm,
            index=encoder.classes_,
            columns=encoder.classes_
        ).to_csv(
            OUTPUT /
            f"{name.lower()}_cv_confusion_matrix.csv"
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
            f"{name} - Out-of-fold Confusion Matrix"
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT /
            f"{name.lower()}_cv_confusion_matrix.png",
            dpi=160
        )

        plt.close(fig)

        report = classification_report(
            all_true,
            all_pred,
            labels=class_ids,
            target_names=encoder.classes_,
            zero_division=0,
            output_dict=True
        )

        pd.DataFrame(report).transpose().to_csv(
            OUTPUT /
            f"{name.lower()}_cv_classification_report.csv"
        )

    # --------------------------------------------------
    # 6. Save fold-wise results
    # --------------------------------------------------

    folds_df = pd.DataFrame(fold_rows)

    folds_df.to_csv(
        OUTPUT / "fold_results.csv",
        index=False
    )

    recall_df = pd.DataFrame(recall_rows)

    recall_df.to_csv(
        OUTPUT / "per_class_recall.csv",
        index=False
    )

    predictions_df = pd.DataFrame(prediction_rows)

    predictions_df.to_csv(
        OUTPUT / "out_of_fold_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # 7. Summarize cross-validation results
    # --------------------------------------------------

    summary = (
        folds_df.groupby("feature_set")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std")
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT / "cv_summary.csv",
        index=False
    )

    print("\n========== CROSS-VALIDATION SUMMARY ==========")
    print(summary.to_string(index=False))

    # --------------------------------------------------
    # 8. Per-class recall summary
    # --------------------------------------------------

    recall_summary = (
        recall_df.groupby(
            ["feature_set", "emotion"]
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

    print("\n========== PER-CLASS RECALL ==========")
    print(recall_summary.to_string(index=False))

    # --------------------------------------------------
    # 9. Plot cross-validation performance
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(summary))
    width = 0.35

    ax.bar(
        x - width / 2,
        summary["mean_accuracy"],
        width,
        yerr=summary["std_accuracy"],
        capsize=5,
        label="Accuracy"
    )

    ax.bar(
        x + width / 2,
        summary["mean_macro_f1"],
        width,
        yerr=summary["std_macro_f1"],
        capsize=5,
        label="Macro F1"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["feature_set"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Actor-Grouped Cross-Validation")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "cv_comparison.png",
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------
    # 10. Save metadata
    # --------------------------------------------------

    metadata = {
        "experiment": "Actor-grouped cross-validation",
        "cross_validation": "StratifiedGroupKFold",
        "folds": N_SPLITS,
        "random_state": RANDOM_STATE,
        "development_recordings": len(development_idx),
        "reserved_test_recordings": len(test_idx),
        "test_evaluated": False,
        "model": {
            "type": "SVM",
            "kernel": "rbf",
            "C": 10,
            "gamma": "scale",
            "class_weight": "balanced"
        },
        "feature_sets": {
            name: len(data["feature_cols"])
            for name, data in datasets.items()
        },
        "emotion_classes": encoder.classes_.tolist()
    }

    with open(
        OUTPUT / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nExperiment completed successfully.")
    print("Results saved in models/group_cross_validation/")
    print("Reserved test actors were NOT evaluated.")


if __name__ == "__main__":
    main()
