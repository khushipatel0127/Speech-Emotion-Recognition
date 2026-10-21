
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

import matplotlib.pyplot as plt
import joblib


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "notebooks" / "advanced_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

feature_cols = [
    col for col in df.columns
    if col.startswith("feature_")
]

X = df[feature_cols].to_numpy(dtype=float)
y_text = df["emotion"].to_numpy()
groups = df["actor_id"].to_numpy()

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

print("Total recordings:", len(df))
print("Number of features:", len(feature_cols))
print("Emotion classes:", list(encoder.classes_))


# --------------------------------------------------
# 3. Actor-based train / validation / test split
# --------------------------------------------------

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(
    split1.split(X, y, groups=groups)
)

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

train_rel, val_rel = next(
    split2.split(
        X[train_val_idx],
        y[train_val_idx],
        groups=groups[train_val_idx]
    )
)

train_idx = train_val_idx[train_rel]
val_idx = train_val_idx[val_rel]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

train_actors = set(groups[train_idx])
val_actors = set(groups[val_idx])
test_actors = set(groups[test_idx])

assert train_actors.isdisjoint(val_actors)
assert train_actors.isdisjoint(test_actors)
assert val_actors.isdisjoint(test_actors)

print("\nTraining recordings:", len(train_idx))
print("Validation recordings:", len(val_idx))
print("Test recordings:", len(test_idx))
print("Actor separation verified.")


# --------------------------------------------------
# 4. Define parameter combinations
# --------------------------------------------------

svm_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.001, 0.01, 0.1],
    "kernel": ["rbf"],
    "class_weight": ["balanced"]
}

rf_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20, 40],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
    "random_state": [RANDOM_STATE],
    "n_jobs": [-1]
}


# --------------------------------------------------
# 5. Tune SVM on validation accuracy
# --------------------------------------------------

results = []
best_model = None
best_name = None
best_params = None
best_val_accuracy = -1.0

print("\n========== SVM TUNING ==========")

for params in ParameterGrid(svm_grid):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(**params)
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(
        y_val, val_pred, average="macro", zero_division=0
    )

    results.append({
        "model": "SVM",
        "params": params,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1
    })

    print(
        f"C={params['C']}, gamma={params['gamma']} | "
        f"Accuracy={val_accuracy:.4f} | "
        f"Macro-F1={val_macro_f1:.4f}"
    )

    # Choose using validation accuracy only
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model
        best_name = "SVM"
        best_params = params.copy()


# --------------------------------------------------
# 6. Tune Random Forest on validation accuracy
# --------------------------------------------------

print("\n========== RANDOM FOREST TUNING ==========")

for params in ParameterGrid(rf_grid):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(**params)
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(
        y_val, val_pred, average="macro", zero_division=0
    )

    results.append({
        "model": "Random Forest",
        "params": params,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1
    })

    print(
        f"Trees={params['n_estimators']}, "
        f"depth={params['max_depth']}, "
        f"leaf={params['min_samples_leaf']} | "
        f"Accuracy={val_accuracy:.4f} | "
        f"Macro-F1={val_macro_f1:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model
        best_name = "Random Forest"
        best_params = params.copy()


# --------------------------------------------------
# 7. Summarize validation results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["validation_accuracy", "validation_macro_f1"],
    ascending=False
)

results_df.to_csv(
    MODEL_DIR / "tuning_results.csv",
    index=False
)

print("\n========== BEST VALIDATION MODEL ==========")
print("Model:", best_name)
print("Parameters:", best_params)
print("Validation accuracy:", round(best_val_accuracy, 4))


# --------------------------------------------------
# 8. Evaluate the selected model on test data
# --------------------------------------------------

test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_pred)
test_macro_f1 = f1_score(
    y_test, test_pred, average="macro", zero_division=0
)

print("\n========== HELD-OUT TEST RESULTS ==========")
print("Test accuracy:", round(test_accuracy, 4))
print("Test macro F1:", round(test_macro_f1, 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)


# --------------------------------------------------
# 9. Save confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=np.arange(len(encoder.classes_))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(xticks_rotation=45, cmap="Blues")
plt.tight_layout()
plt.savefig(
    MODEL_DIR / "tuned_confusion_matrix.png",
    dpi=150
)
plt.close()


# --------------------------------------------------
# 10. Save selected model and metadata
# --------------------------------------------------

joblib.dump(
    best_model,
    MODEL_DIR / "tuned_emotion_model.joblib"
)

joblib.dump(
    encoder,
    MODEL_DIR / "tuned_label_encoder.joblib"
)

metadata = {
    "feature_set": "advanced_features",
    "feature_columns": feature_cols,
    "model_name": best_name,
    "best_parameters": best_params,
    "validation_accuracy": float(best_val_accuracy),
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "sample_rate": 16000,
    "n_mfcc": 40,
    "emotion_classes": encoder.classes_.tolist()
}

with open(MODEL_DIR / "tuned_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4, default=str)

print("\nSaved:")
print("- models/tuned_emotion_model.joblib")
print("- models/tuned_label_encoder.joblib")
print("- models/tuned_metadata.json")
print("- models/tuning_results.csv")
print("- models/tuned_confusion_matrix.png")
