import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = os.path.join("data", "diabetes.csv")
MODEL_PATH = os.path.join("model", "diabetes_model.pkl")
METRICS_PATH = os.path.join("metrics", "metrics.json")
os.makedirs("model", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    # Replace known "zero means missing" columns with NaN for imputation
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)
    return df

def make_pipeline(model, scale=False):
    numeric_features = FEATURES
    if scale:
        pre = ColumnTransformer(
            transformers=[
                ("imputer", SimpleImputer(strategy="median"), numeric_features),
                ("scaler", StandardScaler(), numeric_features)
            ],
            remainder='drop'
        )
        # Note: ColumnTransformer expects tuples; combine via pipeline
        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", model)
        ])
    else:
        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", model)
        ])
    return pipe

def main():
    df = load_and_clean()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = [
        ("LogisticRegression", make_pipeline(LogisticRegression(max_iter=200, n_jobs=None), scale=True)),
        ("RandomForest", make_pipeline(RandomForestClassifier(n_estimators=300, random_state=42), scale=False)),
        ("SVM", make_pipeline(SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42), scale=True)),
        ("DecisionTree", make_pipeline(DecisionTreeClassifier(max_depth=None, random_state=42), scale=False)),
        ("KNN", make_pipeline(KNeighborsClassifier(n_neighbors=7), scale=True)),
    ]

    results = []
    best = (None, None, -1.0, None)  # name, pipe, best_f1, metrics

    for name, pipe in candidates:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        if f1 > best[2]:
            # store classification report too
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            best = (name, pipe, f1, {
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "classification_report": report
            })

    best_name, best_pipe, best_f1, best_metrics = best
    meta = {
        "best_model": best_name,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_order": FEATURES
    }

    # Save model bundle
    bundle = {
        "pipeline": best_pipe,
        "meta": meta
    }
    joblib.dump(bundle, MODEL_PATH)

    # Save metrics
    out = {
        "meta": meta,
        "all_models": results,
        "best_model_metrics": best_metrics
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("== Training complete ==")
    print("Best model:", best_name)
    print(json.dumps(best_metrics, indent=2))

if __name__ == "__main__":
    main()
