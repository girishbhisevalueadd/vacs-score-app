"""
VACS Score – Streamlit Credit Rating Prediction App v2
ValueAdd Research and Analytics LLP

New in v2:
  - Model selector dropdown (defaults to best model)
  - All 6 trained models available for comparison
  - Input validation with warnings
  - Per-model confusion matrix, classification report, feature importances
  - Fixed chart sizing (no longer overflows laptop screen)
  - Detailed error handling
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ─── App Logging ────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)

_log_handler_file    = logging.FileHandler("logs/app.log", mode="a", encoding="utf-8")
_log_handler_console = logging.StreamHandler()
_log_fmt = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log_handler_file.setFormatter(_log_fmt)
_log_handler_console.setFormatter(_log_fmt)

log = logging.getLogger("VACS.app")
if not log.handlers:
    log.addHandler(_log_handler_file)
    log.addHandler(_log_handler_console)
log.setLevel(logging.INFO)

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VACS Score",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Light Theme CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f7fa; }
    html, body, [class*="css"] { font-size: 15px; }

    .vacs-header {
        background: linear-gradient(90deg, #1a3e6e 0%, #2563a8 100%);
        padding: 1.8rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .vacs-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: 8px;
        margin: 0;
    }
    .vacs-subtitle {
        font-size: 1.1rem;
        color: #d0e4ff;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    .vacs-tagline {
        font-size: 0.95rem;
        color: #a8caff;
        margin-top: 0.2rem;
    }

    h3 { color: #1a3e6e !important; font-size: 1.25rem !important; }

    .result-card {
        background: #ffffff;
        border: 1px solid #d0dae8;
        border-left: 5px solid #2563a8;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
    }
    .result-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #5a7a9e;
        margin-bottom: 0.5rem;
    }

    .rating-badge {
        display: inline-block;
        padding: 0.5rem 2rem;
        border-radius: 6px;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 3px;
    }
    .badge-investment  { background-color: #dcfce7; color: #166534; border: 2px solid #16a34a; }
    .badge-speculative { background-color: #fff7ed; color: #9a3412; border: 2px solid #ea580c; }
    .badge-default     { background-color: #fee2e2; color: #991b1b; border: 2px solid #dc2626; }

    .confidence-text {
        font-size: 1.05rem;
        color: #374151;
        margin-top: 0.8rem;
    }
    .confidence-value {
        font-weight: 700;
        color: #2563a8;
        font-size: 1.2rem;
    }

    .input-section-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #2563a8;
        border-bottom: 2px solid #2563a8;
        padding-bottom: 4px;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Model selector card */
    .model-selector-note {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* Validation warning */
    .val-warning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400e;
        margin-bottom: 0.5rem;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }

    [data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #1a3e6e !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #5a7a9e !important; }

    .stTabs [data-baseweb="tab"]  { font-size: 1rem; font-weight: 600; color: #374151; }
    .stTabs [aria-selected="true"] { color: #2563a8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

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

FEATURE_GROUPS = {
    "Liquidity Ratios": [
        "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    ],
    "Profitability Ratios": [
        "netProfitMargin", "pretaxProfitMargin", "grossProfitMargin",
        "operatingProfitMargin", "returnOnAssets",
        "returnOnCapitalEmployed", "returnOnEquity",
    ],
    "Efficiency Ratios": [
        "assetTurnover", "fixedAssetTurnover", "effectiveTaxRate",
    ],
    "Leverage Ratios": [
        "debtEquityRatio", "debtRatio", "companyEquityMultiplier",
    ],
    "Cash Flow Ratios": [
        "freeCashFlowOperatingCashFlowRatio", "freeCashFlowPerShare",
        "cashPerShare", "operatingCashFlowPerShare", "operatingCashFlowSalesRatio",
    ],
    "Valuation Ratios": [
        "ebitPerRevenue", "enterpriseValueMultiple", "payablesTurnover",
    ],
}

INVESTMENT_GRADES  = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"}
SPECULATIVE_GRADES = {"BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C"}

DEFAULTS = {
    "currentRatio": 1.20, "quickRatio": 0.80, "cashRatio": 0.20,
    "daysOfSalesOutstanding": 45.0, "netProfitMargin": 0.05,
    "pretaxProfitMargin": 0.06, "grossProfitMargin": 0.30,
    "operatingProfitMargin": 0.10, "returnOnAssets": 0.05,
    "returnOnCapitalEmployed": 0.10, "returnOnEquity": 0.12,
    "assetTurnover": 0.80, "fixedAssetTurnover": 2.50,
    "debtEquityRatio": 1.50, "debtRatio": 0.45,
    "effectiveTaxRate": 0.22, "freeCashFlowOperatingCashFlowRatio": 0.70,
    "freeCashFlowPerShare": 2.00, "cashPerShare": 5.00,
    "companyEquityMultiplier": 2.50, "ebitPerRevenue": 0.10,
    "enterpriseValueMultiple": 10.0, "operatingCashFlowPerShare": 3.00,
    "operatingCashFlowSalesRatio": 0.10, "payablesTurnover": 8.00,
}

# Soft validation bounds: (min, max, description)
VALIDATION_BOUNDS = {
    "currentRatio":           (0,    50,    "Current Ratio (typically 0–10)"),
    "quickRatio":             (0,    50,    "Quick Ratio (typically 0–10)"),
    "cashRatio":              (0,    20,    "Cash Ratio (typically 0–5)"),
    "daysOfSalesOutstanding": (0,    500,   "DSO (typically 0–200 days)"),
    "grossProfitMargin":      (-2,   1,     "Gross Profit Margin (typically -1 to 1)"),
    "netProfitMargin":        (-5,   1,     "Net Profit Margin (typically -1 to 1)"),
    "debtRatio":              (-1,   5,     "Debt Ratio (typically 0–2)"),
    "effectiveTaxRate":       (-1,   2,     "Effective Tax Rate (typically 0–1)"),
    "enterpriseValueMultiple": (-50, 200,   "EV Multiple (typically -50 to 100)"),
}


# ─── Artifact Loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    """
    Load model artifacts from models/.

    Discovery order for model pipelines:
      1. all_models.pkl  – single dict written by train_models v2
      2. Individual <Name>.pkl files – also written by train_models v2
      3. best_model.pkl  – fallback for old single-model training runs
    Returns (all_models_dict, label_encoder, summary_dict).
    """
    required = ["label_encoder.pkl", "model_summary.pkl"]
    for fname in required:
        if not (MODEL_DIR / fname).exists():
            return None, None, None

    try:
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        with open(MODEL_DIR / "model_summary.pkl", "rb") as f:
            summary = pickle.load(f)

        all_models: dict = {}

        # Option 1 – all_models.pkl (v2 training)
        if (MODEL_DIR / "all_models.pkl").exists():
            with open(MODEL_DIR / "all_models.pkl", "rb") as f:
                all_models = pickle.load(f)
            log.info(f"Loaded all_models.pkl  ({len(all_models)} models: {', '.join(all_models)})")

        # Option 2 – individual <Model_Name>.pkl files (v2 training)
        if not all_models:
            _name_map = {
                "Logistic_Regression": "Logistic Regression",
                "Random_Forest":       "Random Forest",
                "XGBoost":             "XGBoost",
                "LightGBM":            "LightGBM",
                "SVM":                 "SVM",
                "Neural_Network":      "Neural Network",
            }
            for safe, display in _name_map.items():
                path = MODEL_DIR / f"{safe}.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        all_models[display] = pickle.load(f)
                    log.info(f"Loaded individual model file: {path.name}")
            if all_models:
                log.info(f"Discovered {len(all_models)} individual model(s): {', '.join(all_models)}")

        # Option 3 – best_model.pkl only (v1 / single-model training)
        if not all_models and (MODEL_DIR / "best_model.pkl").exists():
            with open(MODEL_DIR / "best_model.pkl", "rb") as f:
                all_models = {summary.get("best_model", "Best Model"): pickle.load(f)}
            log.warning(
                "Only best_model.pkl found. Run 'python train_models.py' "
                "to enable all model selection."
            )

        if not all_models:
            log.error("No model artifacts found in models/")
            return None, None, None

        log.info(
            f"Artifacts loaded — best: {summary.get('best_model')} | "
            f"classes: {summary.get('rating_labels')}"
        )
        return all_models, le, summary

    except Exception as exc:
        log.exception(f"Failed to load model artifacts: {exc}")
        st.error(f"Failed to load model artifacts: {exc}")
        return None, None, None


# ─── Input Validation ─────────────────────────────────────────────────────────
def validate_inputs(inputs: dict) -> list[str]:
    """Return a list of warning messages for out-of-range inputs."""
    warnings_list = []
    for feat, (lo, hi, desc) in VALIDATION_BOUNDS.items():
        val = inputs.get(feat, 0)
        if not (lo <= val <= hi):
            warnings_list.append(
                f"{desc}: entered {val:.4f} is outside the expected range [{lo}, {hi}]."
            )
    # Check for any NaN / inf
    for feat, val in inputs.items():
        if not np.isfinite(val):
            warnings_list.append(f"{FEATURE_LABELS[feat]}: value is not finite ({val}).")
    return warnings_list


# ─── Rating Badge ──────────────────────────────────────────────────────────────
def rating_badge(rating: str) -> str:
    if rating in INVESTMENT_GRADES:
        css = "badge-investment"
    elif rating in SPECULATIVE_GRADES:
        css = "badge-speculative"
    else:
        css = "badge-default"
    return f'<span class="rating-badge {css}">{rating}</span>'


# ─── Input Form ───────────────────────────────────────────────────────────────
def build_input_form() -> dict:
    inputs = {}
    for group_name, features in FEATURE_GROUPS.items():
        st.markdown(
            f'<div class="input-section-label">{group_name}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        for i, feat in enumerate(features):
            with cols[i % 4]:
                inputs[feat] = st.number_input(
                    label=FEATURE_LABELS[feat],
                    value=float(DEFAULTS.get(feat, 0.0)),
                    format="%.4f",
                    key=f"input_{feat}",
                )
    return inputs


# ─── Charts ───────────────────────────────────────────────────────────────────
def plot_model_comparison(summary: dict, best_model: str):
    names     = list(summary.keys())
    test_accs = [v["test_accuracy"] * 100 for v in summary.values()]
    cv_means  = [v["cv_mean"]       * 100 for v in summary.values()]
    cv_stds   = [v["cv_std"]        * 100 for v in summary.values()]

    x     = np.arange(len(names))
    width = 0.35

    # Fixed small size – DO NOT use use_container_width (would stretch and overflow)
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100, facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")

    bars1 = ax.bar(x - width / 2, test_accs, width,
                   label="Test Accuracy", color="#2563a8", alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width / 2, cv_means, width,
                   label=f"CV Mean ({5}-fold)", color="#60a5fa", alpha=0.85, zorder=3,
                   yerr=cv_stds, capsize=3,
                   error_kw={"color": "#64748b", "linewidth": 1.2})

    # Highlight best model
    if best_model in names:
        best_idx = names.index(best_model)
        for bar in [bars1[best_idx], bars2[best_idx]]:
            bar.set_edgecolor("#16a34a")
            bar.set_linewidth(2.5)

    ax.set_xlabel("Model", fontsize=10, color="#374151")
    ax.set_ylabel("Accuracy (%)", fontsize=10, color="#374151")
    ax.set_title("Model Accuracy Comparison", fontsize=12, fontweight="bold",
                 color="#1e293b", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, color="#374151", rotation=15, ha="right")
    ax.tick_params(colors="#374151", labelsize=9)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=7.5, color="#374151", fontweight="600")

    fig.tight_layout(pad=1.5)
    return fig


def plot_roc_auc_bar(summary: dict, best_model: str):
    """Horizontal bar chart of ROC-AUC scores."""
    names    = []
    roc_vals = []
    for name, v in summary.items():
        if v.get("roc_auc") is not None:
            names.append(name)
            roc_vals.append(v["roc_auc"])

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(4.5, max(2.2, len(names) * 0.5)), dpi=100, facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    colors = ["#16a34a" if n == best_model else "#2563a8" for n in names]
    ax.barh(names, roc_vals, color=colors, alpha=0.85)
    for i, (n, v) in enumerate(zip(names, roc_vals)):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9,
                color="#1e293b", fontweight="600")
    ax.set_xlabel("ROC-AUC (Macro OVR)", fontsize=10, color="#374151")
    ax.set_xlim(0, 1.1)
    ax.set_title("ROC-AUC Comparison", fontsize=11, fontweight="bold", color="#1e293b")
    ax.tick_params(labelsize=9, colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    fig.tight_layout(pad=1.5)
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list, model_name: str):
    # Limit to top-15 classes by frequency to keep chart legible
    if len(labels) > 15:
        top_idx = np.argsort(cm.sum(axis=1))[::-1][:15]
        top_idx.sort()
        cm     = cm[np.ix_(top_idx, top_idx)]
        labels = [labels[i] for i in top_idx]

    n = len(labels)
    # Cap size tightly so it never exceeds a typical laptop screen height
    # 4.5 inches × 100 dpi = 450 px — fits comfortably on any display
    size = min(max(3.5, n * 0.6), 4.5)

    fig, ax = plt.subplots(figsize=(size, size), dpi=100, facecolor="#ffffff")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.4, linecolor="#e2e8f0",
        annot_kws={"size": max(7, 10 - n // 3)},
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(f"Confusion Matrix  –  {model_name}",
                 fontsize=10, fontweight="bold", color="#1e293b", pad=8)
    ax.set_xlabel("Predicted Rating", fontsize=9, color="#374151")
    ax.set_ylabel("Actual Rating",    fontsize=9, color="#374151")
    tick_fs = max(7, 9 - n // 4)
    ax.tick_params(labelsize=tick_fs, colors="#374151")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_fs)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=tick_fs)
    fig.tight_layout(pad=1.5)
    return fig


def plot_top3_bar(top3_ratings, top3_probs):
    fig, ax = plt.subplots(figsize=(5, 2.6), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    colors = ["#2563a8", "#60a5fa", "#93c5fd"]
    ax.barh(top3_ratings[::-1], top3_probs[::-1], color=colors, alpha=0.9)
    for i, (r, p) in enumerate(zip(top3_ratings[::-1], top3_probs[::-1])):
        ax.text(p + 0.8, i, f"{p:.1f}%", va="center", fontsize=10,
                color="#1e293b", fontweight="600")
    ax.set_xlabel("Confidence (%)", fontsize=10, color="#374151")
    ax.set_xlim(0, 115)
    ax.tick_params(labelsize=10, colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    fig.tight_layout(pad=1.5)
    return fig


def plot_feature_importance(feat_imp: dict):
    """Plot top-15 features by importance."""
    items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    feats, imps = zip(*items)
    labels = [FEATURE_LABELS.get(f, f) for f in feats]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100, facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.barh(labels[::-1], list(imps)[::-1], color="#2563a8", alpha=0.85)
    ax.set_xlabel("Importance Score", fontsize=10, color="#374151")
    ax.set_title("Top 15 Feature Importances", fontsize=11,
                 fontweight="bold", color="#1e293b")
    ax.tick_params(labelsize=9, colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    fig.tight_layout(pad=1.5)
    return fig


# ─── App Layout ────────────────────────────────────────────────────────────────
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="vacs-header">
            <p class="vacs-title">VACS SCORE</p>
            <p class="vacs-subtitle">ValueAdd Research and Analytics LLP</p>
            <p class="vacs-tagline">Corporate Credit Rating Prediction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar: reload button (clears cache so new models are picked up) ─────
    with st.sidebar:
        st.markdown("### ⚙ Controls")
        if st.button("🔄 Reload Models", use_container_width=True,
                     help="Clear cache and reload all model files from disk"):
            load_artifacts.clear()
            log.info("Cache cleared — reloading model artifacts")
            st.rerun()
        st.markdown(
            "<small>Use this after running **`python train_models.py`** "
            "to make all models available in the dropdown.</small>",
            unsafe_allow_html=True,
        )

    # ── Load artifacts ────────────────────────────────────────────────────────
    all_models, le, summary_data = load_artifacts()
    if all_models is None:
        st.error(
            "Model artifacts not found in `models/`.  "
            "Run **`python train_models.py`** first."
        )
        st.stop()

    best_model_name = summary_data["best_model"]
    model_summary   = summary_data["summary"]
    per_model_eval  = summary_data.get("per_model_eval", {})

    # ── Model Selector ────────────────────────────────────────────────────────
    st.markdown("### Select Model")
    model_names    = list(all_models.keys())
    default_idx    = model_names.index(best_model_name) if best_model_name in model_names else 0

    col_sel, col_note = st.columns([2, 5])
    with col_sel:
        selected_model_name = st.selectbox(
            "Active model",
            model_names,
            index=default_idx,
            label_visibility="collapsed",
        )
    with col_note:
        best_acc = model_summary.get("summary", {}).get(best_model_name, {}).get("test_accuracy")
        best_roc = model_summary.get("summary", {}).get(best_model_name, {}).get("roc_auc")
        roc_str  = f"  |  ROC-AUC {best_roc:.4f}" if best_roc else ""
        acc_str  = f"Test Acc {best_acc * 100:.1f}%{roc_str}" if best_acc else ""
        if len(model_names) == 1:
            st.warning(
                "Only one model loaded from cache. If you have already run "
                "**`python train_models.py`**, click **🔄 Reload Models** in the "
                "sidebar to refresh all 6 models into the dropdown."
            )
        else:
            st.markdown(
                f'<div class="model-selector-note">'
                f'Best model: <b>{best_model_name}</b>'
                f'{" (" + acc_str + ")" if acc_str else ""}. '
                f'Select any model to compare predictions and metrics.'
                f'</div>',
                unsafe_allow_html=True,
            )

    active_model = all_models[selected_model_name]
    # model_summary["summary"] is the per-model metrics dict
    active_stats = model_summary.get(selected_model_name,
                   summary_data.get("summary", {}).get(selected_model_name, {}))
    active_eval  = per_model_eval.get(selected_model_name, {})

    # ── KPI bar ───────────────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Active Model",   selected_model_name)
    k2.metric("Test Accuracy",  f"{active_stats.get('test_accuracy', 0) * 100:.1f}%")
    k3.metric("CV Mean",        f"{active_stats.get('cv_mean', 0) * 100:.1f}%")
    roc = active_stats.get("roc_auc")
    k4.metric("ROC-AUC",        f"{roc:.4f}" if roc else "N/A")
    k5.metric("Rating Classes", len(summary_data["rating_labels"]))

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["  Predict Rating  ", "  Model Performance  ", "  Classification Report  "]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – PREDICT
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Enter Financial Ratios")
        st.write(
            f"Using model: **{selected_model_name}**.  "
            "Fill in the company's financial ratios and click **Predict Credit Rating**."
        )

        with st.form("prediction_form"):
            inputs    = build_input_form()
            submitted = st.form_submit_button(
                "Predict Credit Rating",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            # ── Input validation ──────────────────────────────────────────────
            val_warnings = validate_inputs(inputs)
            if val_warnings:
                for w in val_warnings:
                    log.warning(f"Input validation: {w}")
                st.markdown("**⚠ Input Warnings** (prediction will still run):")
                for w in val_warnings:
                    st.markdown(
                        f'<div class="val-warning">⚠ {w}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Prediction ────────────────────────────────────────────────────
            log.info(f"Prediction requested using model: {selected_model_name}")
            try:
                input_df     = pd.DataFrame([inputs], columns=FEATURE_COLS)
                pred_encoded = active_model.predict(input_df)[0]
                pred_rating  = le.inverse_transform([pred_encoded])[0]
                pred_proba   = active_model.predict_proba(input_df)[0]
                confidence   = pred_proba.max() * 100
                top3_idx     = np.argsort(pred_proba)[::-1][:3]
                top3_ratings = le.inverse_transform(top3_idx)
                top3_probs   = pred_proba[top3_idx] * 100
                log.info(
                    f"Prediction result: {pred_rating}  "
                    f"confidence={confidence:.1f}%  "
                    f"top3={list(zip(top3_ratings, [f'{p:.1f}%' for p in top3_probs]))}"
                )
            except Exception as exc:
                log.exception(f"Prediction failed using {selected_model_name}: {exc}")
                st.error(f"Prediction failed: {exc}")
                st.stop()

            st.markdown("---")
            st.markdown("### Prediction Result")

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted Credit Rating</div>
                        {rating_badge(pred_rating)}
                        <div class="confidence-text">
                            Model Confidence: <span class="confidence-value">{confidence:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if pred_rating in INVESTMENT_GRADES:
                    st.success("Investment Grade  —  Low credit risk")
                elif pred_rating in SPECULATIVE_GRADES:
                    st.warning("Speculative Grade  —  Higher credit risk")
                else:
                    st.error("Default / Near-Default  —  Very high credit risk")

            with col_right:
                st.markdown("**Top 3 Probable Ratings**")
                try:
                    fig_bar = plot_top3_bar(top3_ratings, top3_probs)
                    st.pyplot(fig_bar, width="stretch")
                    plt.close(fig_bar)
                except Exception as exc:
                    st.warning(f"Could not render probability chart: {exc}")

            with st.expander("View all submitted values"):
                st.dataframe(
                    pd.DataFrame({
                        "Feature": [FEATURE_LABELS[f] for f in FEATURE_COLS],
                        "Value":   [inputs[f] for f in FEATURE_COLS],
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Model Accuracy Comparison")
        st.write(
            "Test accuracy and 5-fold CV mean for each trained model. "
            f"Best model **({best_model_name})** highlighted in green."
        )

        # ── Model accuracy chart – fixed size, centered, NO stretch ──────────
        # use_container_width=True would scale image to page width and push
        # the chart off screen.  Render at native dpi/figsize instead.
        chart_col, _ = st.columns([3, 2])
        with chart_col:
            try:
                fig_comp = plot_model_comparison(model_summary, best_model_name)
                st.pyplot(fig_comp, width="content")
                plt.close(fig_comp)
            except Exception as exc:
                st.warning(f"Could not render comparison chart: {exc}")

        # ── ROC-AUC bar ────────────────────────────────────────────────────
        if any(v.get("roc_auc") for v in model_summary.values()):
            st.markdown("#### ROC-AUC (Macro, One-vs-Rest)")
            roc_col, _ = st.columns([2, 3])
            with roc_col:
                try:
                    fig_roc = plot_roc_auc_bar(model_summary, best_model_name)
                    if fig_roc:
                        st.pyplot(fig_roc, width="content")
                        plt.close(fig_roc)
                except Exception as exc:
                    st.warning(f"Could not render ROC-AUC chart: {exc}")

        # ── Summary table ──────────────────────────────────────────────────
        st.markdown("#### Summary Table")
        rows = [
            {
                "Model":         name,
                "Test Accuracy": f"{v['test_accuracy'] * 100:.2f}%",
                "CV Mean":       f"{v['cv_mean'] * 100:.2f}%",
                "CV Std":        f"±{v['cv_std'] * 100:.2f}%",
                "ROC-AUC":       f"{v['roc_auc']:.4f}" if v.get("roc_auc") else "N/A",
                "Best":          "★" if name == best_model_name else "",
            }
            for name, v in model_summary.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(f"### Confusion Matrix  —  {selected_model_name}")
        st.write(f"Showing predictions by **{selected_model_name}** on the test set.")

        cm_data = active_eval.get("confusion_matrix",
                                  summary_data.get("confusion_matrix"))
        labels  = summary_data["rating_labels"]
        if cm_data is not None:
            # Render at native fixed size (no stretching) to stay on screen
            cm_col, _ = st.columns([2, 3])
            with cm_col:
                try:
                    fig_cm = plot_confusion_matrix(cm_data, labels, selected_model_name)
                    st.pyplot(fig_cm, width="content")
                    plt.close(fig_cm)
                except Exception as exc:
                    st.warning(f"Could not render confusion matrix: {exc}")
        else:
            st.info("Confusion matrix not available. Retrain with `python train_models.py`.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – CLASSIFICATION REPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f"### Classification Report  —  {selected_model_name}")
        st.write("Per-class precision, recall, F1-score and support on the test set.")

        report = active_eval.get(
            "classification_report",
            summary_data.get("classification_report", {}),
        )

        if report:
            rows = [
                {
                    "Rating":    cls,
                    "Precision": f"{v['precision']:.2f}",
                    "Recall":    f"{v['recall']:.2f}",
                    "F1-Score":  f"{v['f1-score']:.2f}",
                    "Support":   int(v["support"]),
                }
                for cls, v in report.items()
                if isinstance(v, dict)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Classification report not available. Retrain models.")

        # ── Feature Importances ────────────────────────────────────────────
        feat_imp = active_eval.get("feature_importances")
        if feat_imp:
            st.markdown("---")
            st.markdown("### Feature Importances")
            st.write(f"Top 15 features ranked by importance in **{selected_model_name}**.")
            fi_col, _ = st.columns([3, 2])
            with fi_col:
                try:
                    fig_fi = plot_feature_importance(feat_imp)
                    st.pyplot(fig_fi, width="content")
                    plt.close(fig_fi)
                except Exception as exc:
                    st.warning(f"Could not render feature importance chart: {exc}")

            # Ranked table
            fi_df = (
                pd.DataFrame(
                    {"Feature": list(feat_imp.keys()), "Importance": list(feat_imp.values())}
                )
                .assign(Label=lambda d: d["Feature"].map(FEATURE_LABELS))
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
            fi_df.index += 1
            st.dataframe(
                fi_df[["Label", "Importance"]].rename(columns={"Label": "Feature"}),
                use_container_width=True,
            )
        else:
            st.info(
                f"Feature importances not available for **{selected_model_name}** "
                "(only tree-based and linear models expose importances)."
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">2026 ValueAdd Research and Analytics LLP '
        '&nbsp;|&nbsp; VACS Score v2.0</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
