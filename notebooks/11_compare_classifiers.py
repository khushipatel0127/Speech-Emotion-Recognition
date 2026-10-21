
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
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "notebooks" / "advanced_features.csv"
)

OUTPUT = (
    ROOT / "models" / "classifier_comparison"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------
# 2. Select 84 features
# --------------------------------------------------

# Advanced feature order:
# 1-80   : MFCC
# 81-82  : RMS
# 83-84  : ZCR
# 85-98  : Spectral contrast
# 99-100 : Pitch

FEATURE_COLUMNS = [
    f"feature_{i}"
    for i in range(1, 85)
]


# --------------------------------------------------
# 3. Define classifiers
# --------------------------------------------------

def get_models():

    return {
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

        "Random_Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),

        "Extra_Trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),

        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=RANDOM_STATE
            )
        )
    }


# --------------------------------------------------
# 4. Load dataset
# --------------------------------------------------

def load_data():

    df = pd.read_csv(INPUT_FILE)

    required = {
        "filename",
        "actor_id",
        "emotion"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Missing required dataset columns."
        )

    if df["filename"].duplicated().any():
        raise ValueError(
            "Duplicate filenames found."
        )

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if len(df) != 1440:
        raise ValueError(
            f"Expected 1440 recordings, found {len(df)}."
        )

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["emotion"])

    groups = df["actor_id"].to_numpy()

    return df, X, y, groups, encoder


# --------------------------------------------------
# 5. Main experiment
# --------------------------------------------------

def main():

    df, X, y, groups, encoder = load_data()

    class_ids = np.arange(
        len(encoder.classes_)
    )

    print("Dataset loaded successfully.")
    print("Total recordings:", len(df))
    print("Selected features:", X.shape[1])
    print("Emotion classes:", encoder.classes_.tolist())

    # --------------------------------------------------
    # 6. Preserve original actor-based test split
    # --------------------------------------------------

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    development_idx, test_idx = next(
        outer_split.split(X, y, groups=groups)
    )

    development_actors = set(groups[development_idx])
    test_actors = set(groups[test_idx])

    assert development_actors.isdisjoint(test_actors)

    X_dev = X[development_idx]
    y_dev = y[development_idx]
    groups_dev = groups[development_idx]

    print("\nDevelopment recordings:", len(development_idx))
    print("Excluded test recordings:", len(test_idx))
    print("Actor separation verified.")

    # --------------------------------------------------
    # 7. Generate identical actor-grouped folds
    # --------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = list(
        cv.split(
            X_dev,
            y_dev,
            groups=groups_dev
        )
    )

    fold_results = []
    recall_results = []
    prediction_results = []

    models = get_models()

    # --------------------------------------------------
    # 8. Train and evaluate each classifier
    # --------------------------------------------------

    for model_name, model in models.items():

        print(f"\n========== {model_name} ==========")

        for fold_number, (train_idx, val_idx) in enumerate(
            folds,
            start=1
        ):

            train_actors = set(groups_dev[train_idx])
            val_actors = set(groups_dev[val_idx])

            assert train_actors.isdisjoint(val_actors)

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

            fold_results.append({
                "classifier": model_name,
                "fold": fold_number,
                "feature_count": X_dev.shape[1],
                "training_recordings": len(train_idx),
                "validation_recordings": len(val_idx),
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

            for class_id, emotion in enumerate(
                encoder.classes_
            ):

                recall_results.append({
                    "classifier": model_name,
                    "fold": fold_number,
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
                    "classifier": model_name,
                    "fold": fold_number,
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
                f"Fold {fold_number}: "
                f"Accuracy={accuracy:.4f}, "
                f"Macro F1={macro_f1:.4f}, "
                f"Validation actors={sorted(val_actors)}"
            )

    # --------------------------------------------------
    # 9. Save fold-level results
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
    # 10. Calculate mean and standard deviation
    # --------------------------------------------------

    summary = (
        folds_df.groupby("classifier")
        .agg(
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
        OUTPUT / "classifier_summary.csv",
        index=False
    )

    print("\n========== CLASSIFIER COMPARISON ==========")
    print(summary.to_string(index=False))

    # --------------------------------------------------
    # 11. Per-class recall
    # --------------------------------------------------

    recall_summary = (
        recall_df.groupby(
            ["classifier", "emotion"]
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

    important = recall_summary[
        recall_summary["emotion"].isin(
            ["neutral", "sad"]
        )
    ]

    print("\n========== NEUTRAL AND SAD RECALL ==========")
    print(
        important.sort_values(
            ["emotion", "mean_recall"],
            ascending=[True, False]
        ).to_string(index=False)
    )

    # --------------------------------------------------
    # 12. Comparison chart
    # --------------------------------------------------

    plot_df = summary.sort_values(
        "mean_macro_f1"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        plot_df["classifier"],
        plot_df["mean_macro_f1"],
        xerr=plot_df["std_macro_f1"],
        capsize=4
    )

    ax.set_xlabel("Mean macro F1")
    ax.set_xlim(0, 1)
    ax.set_title(
        "Classifier Comparison - Actor-Grouped CV"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "classifier_comparison.png",
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------
    # 13. Best classifier confusion matrix
    # --------------------------------------------------

    best_classifier = summary.iloc[0]["classifier"]

    best_predictions = predictions_df[
        predictions_df["classifier"] == best_classifier
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
        OUTPUT / "best_classifier_confusion_matrix.csv"
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
        f"Best Classifier: {best_classifier}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "best_classifier_confusion_matrix.png",
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------
    # 14. Classification report
    # --------------------------------------------------

    report = classification_report(
        best_predictions["actual"],
        best_predictions["predicted"],
        labels=encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT / "best_classifier_report.csv"
    )

    # --------------------------------------------------
    # 15. Save metadata
    # --------------------------------------------------

    metadata = {
        "experiment": "Classifier comparison",
        "features": "MFCC + RMS + ZCR",
        "feature_count": 84,
        "cross_validation": "StratifiedGroupKFold",
        "folds": N_SPLITS,
        "random_state": RANDOM_STATE,
        "development_recordings": len(development_idx),
        "excluded_test_recordings": len(test_idx),
        "test_evaluated": False,
        "classifiers": list(models.keys()),
        "best_classifier_by_mean_macro_f1": best_classifier,
        "emotion_classes": encoder.classes_.tolist()
    }

    with open(
        OUTPUT / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n========== SELECTED CLASSIFIER ==========")
    print("Classifier:", best_classifier)
    print(
        "Mean macro F1:",
        round(float(summary.iloc[0]["mean_macro_f1"]), 4)
    )
    print(
        "Mean accuracy:",
        round(float(summary.iloc[0]["mean_accuracy"]), 4)
    )

    print("\nExperiment completed successfully.")
    print("Results saved in models/classifier_comparison/")
    print("Excluded test actors were NOT evaluated.")


if __name__ == "__main__":
    main()
