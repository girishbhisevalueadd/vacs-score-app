"""
VACS Score – Model Training Pipeline v2
ValueAdd Research and Analytics LLP

Improvements over v1:
  - SMOTE class-imbalance handling via imblearn Pipeline
  - Added LightGBM model
  - GridSearchCV hyperparameter tuning for Random Forest, XGBoost, LightGBM
  - ROC-AUC (macro, one-vs-rest) for all models
  - Feature importances for tree-based and linear models
  - Detailed timestamped logging (console + training.log)
  - Saves all models (all_models.pkl) + best model (best_model.pkl)
  - Per-model confusion matrix & classification report in model_summary.pkl
"""

import logging
import pickle
import time
import warnings
from pathlib import Path

# Ensure logs directory exists before logging is configured
Path("logs").mkdir(exist_ok=True)

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("VACS.train")

# ─── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH         = "corporate_rating.csv"
MODEL_DIR         = Path("models")
RANDOM_STATE      = 42
TEST_SIZE         = 0.20
CV_FOLDS          = 5
MIN_CLASS_SAMPLES = 10

# Models that will receive GridSearchCV tuning
TUNE_MODELS = {"Random Forest", "XGBoost", "LightGBM"}

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

FEATURE_LABELS = {
    "currentRatio":                       "Current Ratio",
    "quickRatio":                         "Quick Ratio",
    "cashRatio":                          "Cash Ratio",
    "daysOfSalesOutstanding":             "Days of Sales Outstanding",
    "netProfitMargin":                    "Net Profit Margin",
    "pretaxProfitMargin":                 "Pre-Tax Profit Margin",
    "grossProfitMargin":                  "Gross Profit Margin",
    "operatingProfitMargin":              "Operating Profit Margin",
    "returnOnAssets":                     "Return on Assets (ROA)",
    "returnOnCapitalEmployed":            "Return on Capital Employed",
    "returnOnEquity":                     "Return on Equity (ROE)",
    "assetTurnover":                      "Asset Turnover",
    "fixedAssetTurnover":                 "Fixed Asset Turnover",
    "debtEquityRatio":                    "Debt / Equity Ratio",
    "debtRatio":                          "Debt Ratio",
    "effectiveTaxRate":                   "Effective Tax Rate",
    "freeCashFlowOperatingCashFlowRatio": "FCF / Operating CF Ratio",
    "freeCashFlowPerShare":               "Free Cash Flow per Share",
    "cashPerShare":                       "Cash per Share",
    "companyEquityMultiplier":            "Equity Multiplier",
    "ebitPerRevenue":                     "EBIT per Revenue",
    "enterpriseValueMultiple":            "Enterprise Value Multiple",
    "operatingCashFlowPerShare":          "Operating CF per Share",
    "operatingCashFlowSalesRatio":        "Operating CF / Sales Ratio",
    "payablesTurnover":                   "Payables Turnover",
}

TARGET_COL = "Rating"

MODEL_DIR.mkdir(exist_ok=True)

# ─── GridSearchCV Hyperparameter Grids ──────────────────────────────────────────
# Kept intentionally lean for reasonable training time on a laptop.
# Expand grids for deeper tuning on faster hardware.
PARAM_GRIDS = {
    "Random Forest": {
        "clf__n_estimators":     [200, 400],
        "clf__max_depth":        [10, 20, None],
        "clf__min_samples_leaf": [1, 2],
    },
    "XGBoost": {
        "clf__n_estimators":     [200, 300],
        "clf__max_depth":        [4, 6],
        "clf__learning_rate":    [0.05, 0.10],
        "clf__subsample":        [0.8, 1.0],
    },
    "LightGBM": {
        "clf__n_estimators":     [200, 300],
        "clf__max_depth":        [6, 10],
        "clf__learning_rate":    [0.05, 0.10],
        "clf__num_leaves":       [31, 63],
    },
}


# ─── Data Loading & Preprocessing ───────────────────────────────────────────────
def load_data(path: str):
    """Load CSV, keep required columns, drop rare rating classes."""
    log.info(f"Loading data from: {path}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        log.error(f"Data file not found: {path}")
        raise

    log.info(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Validate required columns
    needed = [TARGET_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[needed].copy()
    before = len(df)
    df.dropna(subset=[TARGET_COL], inplace=True)
    if len(df) < before:
        log.warning(f"  Dropped {before - len(df)} rows with missing Rating")

    # Rating distribution
    class_counts = df[TARGET_COL].value_counts()
    log.info(f"  Rating distribution:\n{class_counts.to_string()}")

    # Drop classes with too few samples for stratified split
    valid_classes = class_counts[class_counts >= MIN_CLASS_SAMPLES].index
    dropped = sorted(set(class_counts.index) - set(valid_classes))
    if dropped:
        log.warning(
            f"  Dropping classes with < {MIN_CLASS_SAMPLES} samples: {dropped}"
        )
    df = df[df[TARGET_COL].isin(valid_classes)].reset_index(drop=True)

    log.info(
        f"  After filtering: {len(df):,} rows | "
        f"{df[TARGET_COL].nunique()} classes"
    )

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Log feature statistics for debugging
    log.debug(f"  Feature summary:\n{X.describe().to_string()}")

    return X, y


# ─── Model Pipelines with SMOTE ─────────────────────────────────────────────────
def build_pipelines() -> dict:
    """
    Build imblearn Pipelines: imputer → scaler → SMOTE → classifier.
    SMOTE is applied only on training folds during cross-validation,
    never on validation/test data.
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    pre = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("smote",   smote),
    ]

    return {
        "Logistic Regression": ImbPipeline(pre + [
            ("clf", LogisticRegression(
                C=1.0, max_iter=2000, solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": ImbPipeline(pre + [
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost": ImbPipeline(pre + [
            ("clf", xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "LightGBM": ImbPipeline(pre + [
            ("clf", lgb.LGBMClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.1,
                num_leaves=63, class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
            )),
        ]),
        "SVM": ImbPipeline(pre + [
            ("clf", SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Neural Network": ImbPipeline(pre + [
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), activation="relu",
                max_iter=500, early_stopping=True, validation_fraction=0.1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ─── Training, Tuning & Evaluation ──────────────────────────────────────────────
def train_and_evaluate_all(
    pipelines: dict,
    X_train, X_test,
    y_train, y_test,
    le: LabelEncoder,
) -> dict:
    """
    For each pipeline:
      1. GridSearchCV if in TUNE_MODELS, else plain fit.
      2. Evaluate on test set: accuracy, ROC-AUC, CM, classification report.
      3. Cross-validate on training set (or reuse GridSearch CV scores).
    Returns a dict keyed by model name.
    """
    results = {}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, pipeline in pipelines.items():
        log.info(f"\n{'─' * 62}")
        log.info(f"  MODEL: {name}")
        t0 = time.time()

        # ── Fit / Tune ────────────────────────────────────────────────────────
        if name in TUNE_MODELS and name in PARAM_GRIDS:
            n_combos = 1
            for vals in PARAM_GRIDS[name].values():
                n_combos *= len(vals)
            log.info(
                f"  GridSearchCV: {n_combos} combos × {CV_FOLDS} folds "
                f"= {n_combos * CV_FOLDS} fits  (n_jobs=-1)"
            )
            gs = GridSearchCV(
                pipeline,
                PARAM_GRIDS[name],
                cv=skf,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
            gs.fit(X_train, y_train)
            fitted = gs.best_estimator_
            cv_mean = round(float(gs.best_score_), 4)
            cv_std  = round(
                float(gs.cv_results_["std_test_score"][gs.best_index_]), 4
            )
            log.info(f"  Best params : {gs.best_params_}")
            log.info(f"  Best CV acc : {cv_mean:.4f} ± {cv_std:.4f}")
        else:
            log.info(f"  Fitting with default hyperparameters...")
            pipeline.fit(X_train, y_train)
            fitted = pipeline
            cv_scores = cross_val_score(
                fitted, X_train, y_train,
                cv=skf, scoring="accuracy", n_jobs=-1,
            )
            cv_mean = round(float(cv_scores.mean()), 4)
            cv_std  = round(float(cv_scores.std()),  4)
            log.info(f"  CV {CV_FOLDS}-fold: {cv_mean:.4f} ± {cv_std:.4f}")

        # ── Test-set metrics ──────────────────────────────────────────────────
        y_pred   = fitted.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        # ROC-AUC (one-vs-rest, macro average)
        roc_auc = None
        try:
            proba   = fitted.predict_proba(X_test)
            roc_auc = round(
                float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro")),
                4,
            )
        except Exception as exc:
            log.warning(f"  ROC-AUC unavailable ({exc})")

        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )

        elapsed = time.time() - t0
        log.info(f"  Test Accuracy : {test_acc:.4f}")
        log.info(f"  ROC-AUC (OVR) : {roc_auc if roc_auc else 'N/A'}")
        log.info(f"  Elapsed       : {elapsed:.1f}s")

        results[name] = {
            "pipeline":              fitted,
            "test_accuracy":         round(float(test_acc), 4),
            "cv_mean":               cv_mean,
            "cv_std":                cv_std,
            "roc_auc":               roc_auc,
            "y_pred":                y_pred,
            "confusion_matrix":      cm,
            "classification_report": report,
        }

    return results


# ─── Feature Importances ────────────────────────────────────────────────────────
def extract_feature_importances(pipeline) -> dict | None:
    """Return {feature: importance} for tree or linear classifiers."""
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return None
    if hasattr(clf, "feature_importances_"):
        return {
            feat: float(imp)
            for feat, imp in zip(FEATURE_COLS, clf.feature_importances_)
        }
    if hasattr(clf, "coef_"):
        # Logistic Regression: mean |coef| across classes
        mean_coef = np.abs(clf.coef_).mean(axis=0)
        return {
            feat: float(imp)
            for feat, imp in zip(FEATURE_COLS, mean_coef)
        }
    return None


# ─── Save Artifacts ─────────────────────────────────────────────────────────────
def save_artifacts(best_name: str, results: dict, le: LabelEncoder):
    """Persist all models and evaluation artifacts to MODEL_DIR."""
    log.info(f"\n{'─' * 62}")
    log.info(f"  Saving artifacts to: {MODEL_DIR}/")

    # Best model (backward-compat with original app.py)
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(results[best_name]["pipeline"], f)
    log.info("  ✔  best_model.pkl")

    # All model pipelines (single file)
    all_models = {name: v["pipeline"] for name, v in results.items()}
    with open(MODEL_DIR / "all_models.pkl", "wb") as f:
        pickle.dump(all_models, f)
    log.info(f"  ✔  all_models.pkl  ({len(all_models)} models)")

    # Individual model pkl files (allows app to discover models independently)
    for name, v in results.items():
        safe_name = name.replace(" ", "_")
        with open(MODEL_DIR / f"{safe_name}.pkl", "wb") as f:
            pickle.dump(v["pipeline"], f)
        log.info(f"  ✔  {safe_name}.pkl")

    # Label encoder
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    log.info("  ✔  label_encoder.pkl")

    # Compact summary (no pipeline objects) + per-model eval data
    summary = {
        name: {
            "test_accuracy": v["test_accuracy"],
            "cv_mean":       v["cv_mean"],
            "cv_std":        v["cv_std"],
            "roc_auc":       v["roc_auc"],
        }
        for name, v in results.items()
    }

    per_model_eval = {
        name: {
            "confusion_matrix":      v["confusion_matrix"],
            "classification_report": v["classification_report"],
            "feature_importances":   extract_feature_importances(v["pipeline"]),
        }
        for name, v in results.items()
    }

    best_v = results[best_name]
    with open(MODEL_DIR / "model_summary.pkl", "wb") as f:
        pickle.dump(
            {
                "best_model":            best_name,
                "summary":               summary,
                "rating_labels":         list(le.classes_),
                "per_model_eval":        per_model_eval,
                # Backward-compat keys
                "confusion_matrix":      best_v["confusion_matrix"],
                "classification_report": best_v["classification_report"],
            },
            f,
        )
    log.info("  ✔  model_summary.pkl")
    log.info(f"{'─' * 62}")


# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    log.info("=" * 62)
    log.info("  VACS Score – Model Training Pipeline v2")
    log.info("=" * 62)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    log.info("\n[1/5] Loading & preprocessing data")
    X, y_raw = load_data(DATA_PATH)

    # ── Step 2: Encode labels ─────────────────────────────────────────────────
    log.info("\n[2/5] Encoding target labels")
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    log.info(f"  Classes ({len(le.classes_)}): {', '.join(le.classes_)}")

    # ── Step 3: Train / test split ────────────────────────────────────────────
    log.info("\n[3/5] Stratified train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Log class distribution in train/test
    train_dist = pd.Series(y_train).value_counts().sort_index()
    test_dist  = pd.Series(y_test).value_counts().sort_index()
    log.info(
        "  Train class counts: "
        + "  ".join(f"{le.classes_[i]}={n}" for i, n in train_dist.items())
    )
    log.info(
        "  Test  class counts: "
        + "  ".join(f"{le.classes_[i]}={n}" for i, n in test_dist.items())
    )

    # ── Step 4: Train & evaluate ──────────────────────────────────────────────
    log.info("\n[4/5] Training & evaluating models")
    log.info(
        f"  Models  : {', '.join(build_pipelines().keys())}\n"
        f"  Tuned   : {', '.join(TUNE_MODELS)}\n"
        f"  CV folds: {CV_FOLDS}  |  SMOTE: enabled  |  n_jobs=-1"
    )
    pipelines = build_pipelines()
    results   = train_and_evaluate_all(
        pipelines, X_train, X_test, y_train, y_test, le
    )

    # ── Step 5: Leaderboard & save ────────────────────────────────────────────
    log.info("\n[5/5] Results leaderboard")
    header = f"{'Model':<22} {'Test Acc':>9} {'CV Mean':>8} {'CV Std':>7} {'ROC-AUC':>9}"
    log.info(header)
    log.info("─" * len(header))
    for name, v in sorted(
        results.items(), key=lambda kv: kv[1]["test_accuracy"], reverse=True
    ):
        roc_str = f"{v['roc_auc']:.4f}" if v["roc_auc"] else "   N/A "
        marker  = "  ← BEST" if name == (best_name := max(
            results, key=lambda k: results[k]["test_accuracy"]
        )) else ""
        log.info(
            f"  {name:<20} {v['test_accuracy']:>8.4f} "
            f"{v['cv_mean']:>8.4f} {v['cv_std']:>7.4f} "
            f"{roc_str:>8}{marker}"
        )

    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    log.info(
        f"\n  ★  BEST MODEL : {best_name}\n"
        f"     Test Acc   : {results[best_name]['test_accuracy']:.4f}\n"
        f"     CV Mean    : {results[best_name]['cv_mean']:.4f}\n"
        f"     ROC-AUC    : "
        f"{results[best_name]['roc_auc'] if results[best_name]['roc_auc'] else 'N/A'}"
    )

    # Print full classification report for best model
    log.info(f"\nClassification Report — {best_name}:")
    log.info(
        "\n"
        + classification_report(
            y_test, results[best_name]["y_pred"], target_names=le.classes_
        )
    )

    save_artifacts(best_name, results, le)

    total_min = (time.time() - t_total) / 60
    log.info(f"\nTotal training time: {total_min:.1f} min")
    log.info("Done!  →  streamlit run app.py")


if __name__ == "__main__":
    main()
