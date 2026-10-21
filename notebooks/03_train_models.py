from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "notebooks" / "features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

feature_cols = [
    col for col in df.columns if col.startswith("mfcc_")
]

X = df[feature_cols].to_numpy()
y_text = df["emotion"].to_numpy()
groups = df["actor_id"].to_numpy()

# Encode emotion names as numeric labels
encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

# First split: 20% test actors
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_val_idx, test_idx = next(
    split1.split(X, y, groups=groups)
)

# Second split: approximately 20% of remaining data for validation
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
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

# Verify that no actor occurs in multiple splits
train_actors = set(groups[train_idx])
val_actors = set(groups[val_idx])
test_actors = set(groups[test_idx])

assert train_actors.isdisjoint(val_actors)
assert train_actors.isdisjoint(test_actors)
assert val_actors.isdisjoint(test_actors)

print("Training recordings:", len(train_idx))
print("Validation recordings:", len(val_idx))
print("Test recordings:", len(test_idx))
print("Actor separation verified.")

models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        )
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)

    results[name] = {
        "model": model,
        "validation_accuracy": val_accuracy
    }

    print(f"{name} validation accuracy: {val_accuracy:.4f}")

# Select the model using validation accuracy only
best_name = max(
    results,
    key=lambda name: results[name]["validation_accuracy"]
)
best_model = results[best_name]["model"]

# Final evaluation on held-out test actors
test_pred = best_model.predict(X_test)

print("\nSelected model:", best_name)
print(
    "Held-out test accuracy:",
    round(accuracy_score(y_test, test_pred), 4)
)
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
plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
plt.close()

joblib.dump(best_model, MODEL_DIR / "emotion_model.joblib")
joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(
        {
            "model_name": best_name,
            "sample_rate": 16000,
            "n_mfcc": 40,
            "feature_count": len(feature_cols),
            "emotions": encoder.classes_.tolist()
        },
        f,
        indent=4
    )

print("\nSaved model and label encoder in models/")
print("Saved confusion matrix in models/confusion_matrix.png")
