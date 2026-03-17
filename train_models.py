"""
VACS Score – Model Training Pipeline
Trains and compares 5 ML models on corporate credit rating data,
then saves the best model and evaluation artifacts to models/.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = "corporate_rating.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MIN_CLASS_SAMPLES = 10  # classes with fewer samples are dropped

FEATURE_COLS = [
    "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    "netProfitMargin", "pretaxProfitMargin", "grossProfitMargin",
    "operatingProfitMargin", "returnOnAssets", "returnOnCapitalEmployed",
    "returnOnEquity", "assetTurnover", "fixedAssetTurnover",
    "debtEquityRatio", "debtRatio", "effectiveTaxRate",
    "freeCashFlowOperatingCashFlowRatio", "freeCashFlowPerShare",
    "cashPerShare", "companyEquityMultiplier", "ebitPerRevenue",
    "enterpriseValueMultiple", "operatingCashFlowPerShare",
    "operatingCashFlowSalesRatio", "payablesTurnover",
]
TARGET_COL = "Rating"

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Data Loading & Preprocessing ──────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)

    # Keep only needed columns
    df = df[[TARGET_COL] + FEATURE_COLS].copy()

    # Drop rows with missing target
    df.dropna(subset=[TARGET_COL], inplace=True)

    # Remove rating classes that have too few samples for stratified split
    class_counts = df[TARGET_COL].value_counts()
    valid_classes = class_counts[class_counts >= MIN_CLASS_SAMPLES].index
    df = df[df[TARGET_COL].isin(valid_classes)].reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


# ─── Model Definitions ─────────────────────────────────────────────────────────
def get_model_pipelines() -> dict:
    """Return a dict of named sklearn Pipelines (imputer → scaler → classifier)."""
    shared_pre = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    return {
        "Logistic Regression": Pipeline(shared_pre + [
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs",
                random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest": Pipeline(shared_pre + [
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_leaf=2,
                n_jobs=-1, random_state=RANDOM_STATE
            ))
        ]),
        "XGBoost": Pipeline(shared_pre + [
            ("clf", xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE
            ))
        ]),
        "SVM": Pipeline(shared_pre + [
            ("clf", SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=RANDOM_STATE
            ))
        ]),
        "Neural Network": Pipeline(shared_pre + [
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), activation="relu",
                max_iter=400, early_stopping=True, validation_fraction=0.1,
                random_state=RANDOM_STATE
            ))
        ]),
    }


# ─── Training & Evaluation ─────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, le, pipelines: dict) -> dict:
    results = {}
    for name, pipeline in pipelines.items():
        print(f"\n  [{name}] Training...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=CV_FOLDS,
            scoring="accuracy", n_jobs=-1
        )

        results[name] = {
            "pipeline": pipeline,
            "test_accuracy": round(test_acc, 4),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "y_pred": y_pred,
        }
        print(
            f"Test Acc={test_acc:.4f} | "
            f"CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

    return results


# ─── Artifact Persistence ──────────────────────────────────────────────────────
def save_artifacts(best_name, best_pipeline, le, results, y_test):
    # Best model pipeline (includes imputer + scaler)
    with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    # Label encoder
    with open(f"{MODEL_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Comparison summary + confusion matrix (no pipeline objects)
    summary = {
        name: {
            "test_accuracy": v["test_accuracy"],
            "cv_mean": v["cv_mean"],
            "cv_std": v["cv_std"],
        }
        for name, v in results.items()
    }
    y_pred_best = results[best_name]["y_pred"]
    cm = confusion_matrix(y_test, y_pred_best)
    report = classification_report(
        y_test, y_pred_best,
        target_names=le.classes_, output_dict=True
    )

    with open(f"{MODEL_DIR}/model_summary.pkl", "wb") as f:
        pickle.dump({
            "summary": summary,
            "best_model": best_name,
            "confusion_matrix": cm,
            "rating_labels": list(le.classes_),
            "classification_report": report,
        }, f)

    print(f"\nArtifacts saved -> {MODEL_DIR}/")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  VACS Score – Model Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading and preprocessing data...")
    X, y_raw = load_data(DATA_PATH)
    print(f"      Samples: {len(X):,}  |  Classes: {y_raw.nunique()}  |  Features: {len(FEATURE_COLS)}")

    # 2. Encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"      Ratings: {', '.join(le.classes_)}")

    # 3. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 4. Train models
    print("\n[2/4] Training & evaluating models...")
    pipelines = get_model_pipelines()
    results = train_and_evaluate(X_train, X_test, y_train, y_test, le, pipelines)

    # 5. Pick best model
    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"\n[3/4] Best model -> {best_name}  (Accuracy: {results[best_name]['test_accuracy']:.4f})")

    # 6. Print classification report
    print("\n[4/4] Classification Report (Best Model):")
    y_pred_best = results[best_name]["y_pred"]
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    # 7. Save
    save_artifacts(best_name, best_pipeline, le, results, y_test)
    print("\nDone! Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
