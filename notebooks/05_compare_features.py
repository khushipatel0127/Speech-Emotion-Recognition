
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "MFCC only": ROOT / "notebooks" / "features.csv",
    "Advanced features": ROOT / "notebooks" / "advanced_features.csv",
}


def split_by_actor(df):
    # Use the same random seed and actor-based split for both datasets.
    X = df.filter(regex=r"^(mfcc_|feature_)").to_numpy()
    y_text = df["emotion"].to_numpy()
    groups = df["actor_id"].to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    first = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_val, test = next(first.split(X, y, groups))

    second = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=42
    )
    train_rel, val_rel = next(
        second.split(
            X[train_val],
            y[train_val],
            groups=groups[train_val]
        )
    )

    train = train_val[train_rel]
    val = train_val[val_rel]

    assert set(groups[train]).isdisjoint(set(groups[val]))
    assert set(groups[train]).isdisjoint(set(groups[test]))
    assert set(groups[val]).isdisjoint(set(groups[test]))

    return X, y, train, val, test, encoder


results = []

for dataset_name, file_path in DATASETS.items():
    print(f"\n{'=' * 55}")
    print("Feature set:", dataset_name)

    df = pd.read_csv(file_path)
    X, y, train, val, test, encoder = split_by_actor(df)

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
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    }

    best_model = None
    best_name = None
    best_val = -1

    for model_name, model in models.items():
        model.fit(X[train], y[train])
        val_pred = model.predict(X[val])
        val_acc = accuracy_score(y[val], val_pred)

        print(f"{model_name} validation accuracy: {val_acc:.4f}")

        results.append({
            "Feature set": dataset_name,
            "Model": model_name,
            "Validation accuracy": val_acc
        })

        if val_acc > best_val:
            best_val = val_acc
            best_model = model
            best_name = model_name

    print("Best model for this feature set:", best_name)
    print("Best validation accuracy:", round(best_val, 4))

    # Test is evaluated only after model selection on validation data.
    test_pred = best_model.predict(X[test])
    test_acc = accuracy_score(y[test], test_pred)

    print("Held-out test accuracy:", round(test_acc, 4))
    print("\nTest classification report:")
    print(classification_report(
        y[test],
        test_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))

print("\nValidation comparison:")
print(pd.DataFrame(results).to_string(index=False))
