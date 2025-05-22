"""Interactive Streamlit dashboard for ASIA Motor Score Prediction.

Launch with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional heavy libs imported lazily when needed
import matplotlib.pyplot as plt  # noqa: E402
import plotly.express as px  # noqa: E402
import shap  # noqa: E402

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PKG_DIR = PROJECT_ROOT / "asia-impairment-track-prediction"
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "visuals" / "figures"
MODELS_DIR = PKG_DIR / "models_exact"

PROCESSED_TRAIN_PATH = DATA_DIR / "train_processed.csv"
XGB_MODEL_PATH = MODELS_DIR / "xgb_exact_model.pkl"

TARGET_COLS: list[str] = [
    "elbfll", "wrextl", "elbexl", "finfll", "finabl", "hipfll",
    "kneexl", "ankdol", "gretol", "ankpll", "elbflr", "wrextr",
    "elbexr", "finflr", "finabr", "hipflr", "kneetr", "ankdor",
    "gretor", "ankplr",
]

# -----------------------------------------------------------------------------
# Caching helpers
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the processed training data (week-1 features + motor scores)."""
    if not PROCESSED_TRAIN_PATH.exists():
        st.error("Processed data not found – run preprocessing first.")
        st.stop()
    df = pd.read_csv(PROCESSED_TRAIN_PATH)
    if "PID" in df.columns:
        df = df.drop(columns=["PID"])
    return df


@st.cache_resource(show_spinner=False)
def load_xgb():
    """Load the exact XGBoost model saved during competition replication."""
    if not XGB_MODEL_PATH.exists():
        st.error("Exact XGBoost model not found – run replication first.")
        st.stop()
    return joblib.load(XGB_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def shap_explainer(model):
    """Create a SHAP TreeExplainer once and reuse."""
    return shap.TreeExplainer(model)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

def show_image_if_exists(path: Path, caption: str | None = None, width: int | None = None):
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=(width is None))
    else:
        st.warning(f"Image not found: {path.name}")


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ASIA Motor Score Prediction", page_icon="🧠", layout="wide"
)

st.sidebar.title("📊 Dashboard Navigation")

SECTIONS = [
    "Project Overview",
    "Data Processing",
    "Modelling Methodology",
    "Per-Target Performance",
    "Calibration Curve",
    "Residual Heatmap",
    "Feature Importance",
    "Patient Feature Differences",
    "Demographic Composition",
    "Key Findings",
    "Next Steps",
    "Acknowledgements",
]

section = st.sidebar.radio("Jump to section", SECTIONS)

# Main content per section
# -----------------------------------------------------------------------------
if section == "Project Overview":
    st.header("From week-1 bedside data to 6- & 12-month motor-score predictions")
    show_image_if_exists(FIG_DIR / "animated_heatmap.gif", caption="Prediction vs Reality", width=None)
    st.markdown(
        "Designed for **60 % clinicians / 40 % data-scientists**. "
        "All tech framed by clinical impact: *What new conversation can I have with my patient today?*"
    )

elif section == "Data Processing":
    st.header("Data Processing & Feature Engineering")
    st.markdown("""
    • Dropped PID identifier; imputed missing values (median/mode).
    
    • One-hot encoded AIS grade, neurological level, and other categoricals.
    
    • Derived composite strength scores & age bands.
    
    • Scaled continuous predictors (z-score) prior to modeling.
    """)

elif section == "Modelling Methodology":
    st.header("Modelling Methodology")
    st.markdown("""
    • Base models: **CatBoost**, **XGBoost**, **HistGB** (20 outputs each).
    
    • Hyperparameters tuned via Optuna with 5-fold CV & early stopping.
    
    • Final ensemble = equal-weighted mean of per-target predictions to avoid over-fitting.
    
    • Validation split at patient level (80/20) to avoid leakage.
    """)

elif section == "Per-Target Performance":
    st.header("Per-Target Performance (Radar)")
    show_image_if_exists(FIG_DIR / "radar_target_rmse.png", caption="Target-wise RMSE – distal muscles hardest")

elif section == "Calibration Curve":
    st.header("Reliability & Uncertainty")
    show_image_if_exists(FIG_DIR / "calibration_curve_enhanced.png", caption="Reliability curve")

elif section == "Residual Heatmap":
    st.header("Residual Heatmap")
    show_image_if_exists(FIG_DIR / "residuals_heatmap.png")

elif section == "Feature Importance":
    st.header("Global Feature Importance – SHAP")
    col1, col2 = st.columns([1, 1])
    with col1:
        show_image_if_exists(FIG_DIR / "shap_summary_xgb.png", caption="Mean |SHAP| (XGB)")
    with col2:
        show_image_if_exists(FIG_DIR / "shap_beeswarm_ensemble.png", caption="Top-20 features (Ensemble)")

elif section == "Patient Feature Differences":
    st.header("Best vs Worst Patient – Feature Differences")
    show_image_if_exists(FIG_DIR / "patient_group_diff_bars.png")

elif section == "Demographic Composition":
    st.header("Demographic Composition – Best vs Worst")
    show_image_if_exists(FIG_DIR / "patient_group_demographics.png")

elif section == "Key Findings":
    st.header("Key Findings")
    st.markdown("""
    • Ensemble attains RMSE **< 0.90** on validation set.
    
    • Predictions well-calibrated at 0/5 scores; slight over-prediction mid-range.
    
    • **Age**, initial AIS grade, and proximal strength are most influential features.
    """)

elif section == "Next Steps":
    st.header("Next Steps")
    st.markdown("""
    • Incorporate MRI imaging & longitudinal fine-tuning.
    
    • Develop bedside decision-support prototype.
    """)

elif section == "Acknowledgements":
    st.header("Acknowledgements")
    st.markdown("""
    • Craig Hospital Research Department & SCI Model Systems.
    
    • Collaborators, clinicians, and patients contributing data.
    
    • Kaggle & ASIA organising committee.
    """)

elif section == "Explainability":
    st.header("Explainability – SHAP Values")
    col1, col2 = st.columns([1, 1])
    with col1:
        show_image_if_exists(FIG_DIR / "shap_summary_xgb.png", caption="Global feature importance (Mean |SHAP|)")
    with col2:
        show_image_if_exists(FIG_DIR / "shap_beeswarm_ensemble.png", caption="Top-20 features across ensemble")

    # Patient-specific waterfall
    st.markdown("### Patient-specific explanation")
    df = load_data()
    idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
    if st.button("Generate waterfall plot"):
        with st.spinner("Computing SHAP ..."):
            model = load_xgb()
            explainer = shap_explainer(model)
            feature_cols = [c for c in df.columns if c not in TARGET_COLS]
            shap_values = explainer(df.iloc[[idx]][feature_cols])
            # Plot using SHAP's built-in waterfall
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close()

elif section == "Demographic Composition":
    # already handled earlier; nothing here (fallback)
    pass

st.sidebar.markdown("---")
st.sidebar.info(
    "[GitHub repo](https://github.com/user/kaggle_winning_solution) |  "
    "[Competition page](https://www.kaggle.com/competitions/asia-impairment-track-prediction)"
)


if __name__ == "__main__":
    # When executed directly (e.g., `python streamlit_app.py`) show message.
    # Proper launch is `streamlit run streamlit_app.py`.
    print("Run with:  streamlit run streamlit_app.py")
    sys.exit(0)
