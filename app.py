"""
VACS Score – Streamlit Credit Rating Prediction App
ValueAdd Research and Analytics LLP
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

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
    /* Global light background */
    .stApp { background-color: #f5f7fa; }
    html, body, [class*="css"] { font-size: 15px; }

    /* Top header banner */
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

    /* Section headings */
    h3 { color: #1a3e6e !important; font-size: 1.25rem !important; }

    /* Rating result card */
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

    /* Rating badge */
    .rating-badge {
        display: inline-block;
        padding: 0.5rem 2rem;
        border-radius: 6px;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 3px;
    }
    .badge-investment { background-color: #dcfce7; color: #166534; border: 2px solid #16a34a; }
    .badge-speculative { background-color: #fff7ed; color: #9a3412; border: 2px solid #ea580c; }
    .badge-default     { background-color: #fee2e2; color: #991b1b; border: 2px solid #dc2626; }

    /* Confidence text */
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

    /* Input section label */
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

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Streamlit metric tweaks */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #1a3e6e !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #5a7a9e !important; }

    /* Tab font */
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; color: #374151; }
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

# Group feature columns into logical sections for a cleaner form
FEATURE_GROUPS = {
    "Liquidity Ratios": [
        "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding"
    ],
    "Profitability Ratios": [
        "netProfitMargin", "pretaxProfitMargin", "grossProfitMargin",
        "operatingProfitMargin", "returnOnAssets", "returnOnCapitalEmployed", "returnOnEquity"
    ],
    "Efficiency Ratios": [
        "assetTurnover", "fixedAssetTurnover", "effectiveTaxRate"
    ],
    "Leverage Ratios": [
        "debtEquityRatio", "debtRatio", "companyEquityMultiplier"
    ],
    "Cash Flow Ratios": [
        "freeCashFlowOperatingCashFlowRatio", "freeCashFlowPerShare",
        "cashPerShare", "operatingCashFlowPerShare", "operatingCashFlowSalesRatio"
    ],
    "Valuation Ratios": [
        "ebitPerRevenue", "enterpriseValueMultiple", "payablesTurnover"
    ],
}

INVESTMENT_GRADES = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"}
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


# ─── Artifact Loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    required = ["best_model.pkl", "label_encoder.pkl", "model_summary.pkl"]
    if any(not (MODEL_DIR / f).exists() for f in required):
        return None, None, None
    with open(MODEL_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(MODEL_DIR / "model_summary.pkl", "rb") as f:
        summary = pickle.load(f)
    return model, le, summary


# ─── Rating Badge ──────────────────────────────────────────────────────────────
def rating_badge(rating: str) -> str:
    if rating in INVESTMENT_GRADES:
        css = "badge-investment"
    elif rating in SPECULATIVE_GRADES:
        css = "badge-speculative"
    else:
        css = "badge-default"
    return f'<span class="rating-badge {css}">{rating}</span>'


# ─── Input Form (grouped) ─────────────────────────────────────────────────────
def build_input_form() -> dict:
    inputs = {}
    for group_name, features in FEATURE_GROUPS.items():
        st.markdown(f'<div class="input-section-label">{group_name}</div>', unsafe_allow_html=True)
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


# ─── Charts (light theme) ─────────────────────────────────────────────────────
def plot_model_comparison(summary: dict, best_model: str):
    names     = list(summary.keys())
    test_accs = [v["test_accuracy"] * 100 for v in summary.values()]
    cv_means  = [v["cv_mean"]       * 100 for v in summary.values()]
    cv_stds   = [v["cv_std"]        * 100 for v in summary.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")

    bars1 = ax.bar(x - width / 2, test_accs, width, label="Test Accuracy",
                   color="#2563a8", alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width / 2, cv_means, width, label="CV Mean (5-fold)",
                   color="#60a5fa", alpha=0.85, zorder=3,
                   yerr=cv_stds, capsize=4,
                   error_kw={"color": "#64748b", "linewidth": 1.2})

    # Highlight best model
    best_idx = names.index(best_model)
    for bar in [bars1[best_idx], bars2[best_idx]]:
        bar.set_edgecolor("#16a34a")
        bar.set_linewidth(2.5)

    ax.set_xlabel("Model", fontsize=11, color="#374151")
    ax.set_ylabel("Accuracy (%)", fontsize=11, color="#374151")
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color="#374151", rotation=10, ha="right")
    ax.tick_params(colors="#374151", labelsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color="#374151", fontweight="600")

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list, best_model: str):
    if len(labels) > 15:
        top_idx = np.argsort(cm.sum(axis=1))[::-1][:15]
        top_idx.sort()
        cm     = cm[np.ix_(top_idx, top_idx)]
        labels = [labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#ffffff")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor="#e2e8f0",
        annot_kws={"size": 11},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix  -  {best_model}",
                 fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    ax.set_xlabel("Predicted Rating", fontsize=11, color="#374151")
    ax.set_ylabel("Actual Rating",    fontsize=11, color="#374151")
    ax.tick_params(labelsize=11, colors="#374151")
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_tick_params(rotation=0)
    fig.tight_layout()
    return fig


def plot_top3_bar(top3_ratings, top3_probs):
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    colors = ["#2563a8", "#60a5fa", "#93c5fd"]
    ax.barh(top3_ratings[::-1], top3_probs[::-1], color=colors, alpha=0.9)
    for i, (r, p) in enumerate(zip(top3_ratings[::-1], top3_probs[::-1])):
        ax.text(p + 0.8, i, f"{p:.1f}%", va="center", fontsize=10,
                color="#1e293b", fontweight="600")
    ax.set_xlabel("Confidence (%)", fontsize=10, color="#374151")
    ax.set_xlim(0, 115)
    ax.tick_params(labelsize=11, colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    ax.set_facecolor("#f8fafc")
    fig.tight_layout()
    return fig


def plot_feature_importance(clf, feat_df):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.barh(feat_df["Label"][::-1], feat_df["Importance"][::-1],
            color="#2563a8", alpha=0.85)
    ax.set_xlabel("Importance Score", fontsize=11, color="#374151")
    ax.set_title("Top 15 Feature Importances",
                 fontsize=13, fontweight="bold", color="#1e293b")
    ax.tick_params(labelsize=10, colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    fig.tight_layout()
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

    # ── Load model ────────────────────────────────────────────────────────────
    model, le, summary_data = load_artifacts()
    if model is None:
        st.error("Model artifacts not found. Run `python train_models.py` first.")
        st.stop()

    best_model_name = summary_data["best_model"]
    best_acc        = summary_data["summary"][best_model_name]["test_accuracy"]

    # ── KPI bar ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Model",     best_model_name)
    k2.metric("Test Accuracy",  f"{best_acc * 100:.1f}%")
    k3.metric("Rating Classes", len(summary_data["rating_labels"]))
    k4.metric("Features Used",  len(FEATURE_COLS))

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["  Predict Rating  ", "  Model Performance  ", "  Classification Report  "])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – PREDICT
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Enter Financial Ratios")
        st.write("Fill in the company's financial ratios below and click **Predict Credit Rating**.")

        with st.form("prediction_form"):
            inputs = build_input_form()
            st.markdown("")
            submitted = st.form_submit_button(
                "Predict Credit Rating",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            input_df     = pd.DataFrame([inputs], columns=FEATURE_COLS)
            pred_encoded = model.predict(input_df)[0]
            pred_rating  = le.inverse_transform([pred_encoded])[0]
            pred_proba   = model.predict_proba(input_df)[0]
            confidence   = pred_proba.max() * 100
            top3_idx     = np.argsort(pred_proba)[::-1][:3]
            top3_ratings = le.inverse_transform(top3_idx)
            top3_probs   = pred_proba[top3_idx] * 100

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
                    st.success("Investment Grade  -  Low credit risk")
                elif pred_rating in SPECULATIVE_GRADES:
                    st.warning("Speculative Grade  -  Higher credit risk")
                else:
                    st.error("Default / Near-Default  -  Very high credit risk")

            with col_right:
                st.markdown("**Top 3 Probable Ratings**")
                fig_bar = plot_top3_bar(top3_ratings, top3_probs)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

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
        st.write("Test accuracy and 5-fold cross-validation mean for each trained model. Best model is highlighted in green.")

        fig_comp = plot_model_comparison(summary_data["summary"], best_model_name)
        st.pyplot(fig_comp, use_container_width=True)
        plt.close(fig_comp)

        st.markdown("#### Summary Table")
        rows = [
            {
                "Model":         name,
                "Test Accuracy": f"{v['test_accuracy'] * 100:.2f}%",
                "CV Mean":       f"{v['cv_mean'] * 100:.2f}%",
                "CV Std":        f"+/- {v['cv_std'] * 100:.2f}%",
                "Best Model":    "Yes" if name == best_model_name else "",
            }
            for name, v in summary_data["summary"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Confusion Matrix")
        st.write(f"Showing predictions by **{best_model_name}** on the test set.")
        fig_cm = plot_confusion_matrix(
            summary_data["confusion_matrix"],
            summary_data["rating_labels"],
            best_model_name,
        )
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – CLASSIFICATION REPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f"### Classification Report  -  {best_model_name}")
        st.write("Per-class precision, recall, F1-score and support on the test set.")

        report = summary_data["classification_report"]
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

        # Feature importance
        clf = model.named_steps.get("clf")
        if hasattr(clf, "feature_importances_"):
            st.markdown("---")
            st.markdown("### Feature Importances")
            st.write("Top 15 features ranked by importance in the best model.")
            feat_df = (
                pd.DataFrame({"Feature": FEATURE_COLS,
                              "Importance": clf.feature_importances_})
                .assign(Label=lambda d: d["Feature"].map(FEATURE_LABELS))
                .sort_values("Importance", ascending=False)
                .head(15)
            )
            fig_fi = plot_feature_importance(clf, feat_df)
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">2026 ValueAdd Research and Analytics LLP &nbsp;|&nbsp; VACS Score v1.0</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
