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
import streamlit.components.v1 as components  # noqa: E402

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
    page_title="ASIA Motor Score Prediction", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced presentation
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .highlight-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    h1 {
        color: #1976d2;
        font-weight: 700;
    }
    h2 {
        color: #424242;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Dashboard Navigation")
st.sidebar.markdown("### American Spinal Institute Association")
st.sidebar.markdown("**Kaggle Winning Solution Presentation**")
st.sidebar.markdown("---")

SECTIONS = [
    "🏠 Project Overview",
    "🔬 Data Processing",
    "🤖 Modelling Methodology",
    "🎯 Per-Target Performance",
    "📈 Interactive 3D Outcomes",
    "🏥 Clinical Impact Dashboard",
    "🔍 Interactive SHAP Explorer",
    "📊 Animated Recovery Paths",
    "📉 Calibration & Reliability",
    "🔥 Residual Analysis",
    "👥 Patient Demographics",
    "💡 Key Findings",
    "🚀 Next Steps",
    "🙏 Acknowledgements",
]

section = st.sidebar.radio("Navigate to:", SECTIONS)

# Main content per section
# -----------------------------------------------------------------------------
if section == "🏠 Project Overview":
    st.title("🧠 ASIA Motor Score Prediction")
    st.markdown("### From week-1 bedside data to 6- & 12-month motor-score predictions")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        show_image_if_exists(FIG_DIR / "animated_heatmap.gif", caption="Prediction vs Reality")
    
    st.markdown("""
    <div class="highlight-box">
    <h4>🎯 Mission</h4>
    Designed for <b>60% clinicians / 40% data-scientists</b>. All tech framed by clinical impact: 
    <i>"What new conversation can I have with my patient today?"</i>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #1976d2; margin: 0;">20</h3>
        <p style="margin: 0;">Motor Scores Predicted</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #388e3c; margin: 0;"><0.90</h3>
        <p style="margin: 0;">RMSE Achieved</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #f57c00; margin: 0;">3</h3>
        <p style="margin: 0;">Model Ensemble</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #d32f2f; margin: 0;">#1</h3>
        <p style="margin: 0;">Kaggle Ranking</p>
        </div>
        """, unsafe_allow_html=True)

elif section == "🔬 Data Processing":
    st.header("🔬 Data Processing & Feature Engineering")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Data Pipeline
        - **Dropped** PID identifier for privacy
        - **Imputed** missing values (median/mode)
        - **Encoded** categorical variables
        - **Created** interaction features
        - **Normalized** continuous features
        """)
    
    with col2:
        st.markdown("""
        ### Feature Categories
        - 📊 **Demographics**: Age, Sex, BMI
        - 🏥 **Clinical**: AIS grade, NLI level
        - 💪 **Motor Scores**: 20 muscle groups
        - 🧠 **Sensory**: Light touch, pin prick
        - ⚡ **Reflexes**: Voluntary anal contraction
        """)

elif section == "🤖 Modelling Methodology":
    st.header("🤖 Modelling Methodology")
    st.markdown("""
    • Base models: **CatBoost**, **XGBoost**, **HistGB** (20 outputs each).
    
    • Hyperparameters tuned via Optuna with 5-fold CV & early stopping.
    
    • Final ensemble = equal-weighted mean of per-target predictions to avoid over-fitting.
    
    • Validation split at patient level (80/20) to avoid leakage.
    """)

elif section == "📈 Interactive 3D Outcomes":
    st.header("📈 Interactive 3D Patient Outcomes Visualization")
    
    st.markdown("""
    <div class="highlight-box">
    <b>🔍 Explore patient outcomes in 3D space</b><br>
    Rotate, zoom, and hover over data points to understand relationships between age, initial severity, and recovery potential.
    </div>
    """, unsafe_allow_html=True)
    
    # Load and display the 3D visualization
    html_path = FIG_DIR / "interactive_3d_outcomes.html"
    if html_path.exists():
        with open(html_path, 'r') as f:
            html_content = f.read()
        components.html(html_content, height=850, scrolling=True)
    else:
        st.warning("3D visualization not found. Run `python -m visuals.interactive_3d_outcomes` to generate.")
    
    # Also show the recovery heatmap
    st.subheader("Recovery Patterns by Demographics")
    heatmap_path = FIG_DIR / "recovery_heatmap.html"
    if heatmap_path.exists():
        with open(heatmap_path, 'r') as f:
            heatmap_content = f.read()
        components.html(heatmap_content, height=650, scrolling=True)

elif section == "🏥 Clinical Impact Dashboard":
    st.header("🏥 Clinical Impact & Decision Support")
    
    tab1, tab2, tab3 = st.tabs(["Patient Recovery Timeline", "Clinical Insights", "Functional Milestones"])
    
    with tab1:
        st.markdown("### Interactive Patient Recovery Timeline")
        timeline_path = FIG_DIR / "recovery_timeline.html"
        if timeline_path.exists():
            with open(timeline_path, 'r') as f:
                timeline_content = f.read()
            components.html(timeline_content, height=650, scrolling=True)
        else:
            st.info("Generate timeline by running `python -m visuals.clinical_impact_dashboard`")
    
    with tab2:
        st.markdown("### Clinical Decision Support Panel")
        show_image_if_exists(FIG_DIR / "clinical_insights_panel.png", 
                           caption="Comprehensive clinical insights for patient care planning")
    
    with tab3:
        st.markdown("""
        ### Functional Milestone Predictions
        
        Our model predicts likelihood of achieving key functional milestones:
        
        **Upper Extremity Functions:**
        - 🤚 Basic Hand Function (finger movements)
        - 🍴 Feeding Independence (elbow/wrist control)
        - ♿ Wheelchair Propulsion (arm strength)
        - 🏋️ Transfer Capability (full arm function)
        
        **Lower Extremity Functions:**
        - 🦵 Standing Balance (hip/knee control)
        - 🚶 Walking Potential (leg coordination)
        - 🪜 Stair Navigation (advanced mobility)
        - 🏃 Community Ambulation (full function)
        """)

elif section == "🔍 Interactive SHAP Explorer":
    st.header("🔍 Interactive SHAP Feature Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
        <b>🎛️ Explore Feature Impacts Interactively</b><br>
        Understand how different patient characteristics influence predictions through interactive visualizations.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Refresh Visualizations", type="primary"):
            st.experimental_rerun()
    
    # Display interactive SHAP explorer
    explorer_path = FIG_DIR / "interactive_shap_explorer.html"
    if explorer_path.exists():
        with open(explorer_path, 'r') as f:
            explorer_content = f.read()
        components.html(explorer_content, height=950, scrolling=True)
    else:
        st.info("Generate explorer by running `python -m visuals.interactive_shap_explorer`")
    
    # Show waterfall plot
    st.subheader("Feature Contribution Waterfall")
    waterfall_path = FIG_DIR / "shap_waterfall.html"
    if waterfall_path.exists():
        with open(waterfall_path, 'r') as f:
            waterfall_content = f.read()
        components.html(waterfall_content, height=650, scrolling=True)

elif section == "📊 Animated Recovery Paths":
    st.header("📊 Animated Recovery Trajectories")
    
    tab1, tab2, tab3 = st.tabs(["Group Comparisons", "Uncertainty Visualization", "Motor Function Heatmap"])
    
    with tab1:
        st.markdown("### Recovery Paths by AIS Grade")
        recovery_path = FIG_DIR / "animated_recovery_paths.html"
        if recovery_path.exists():
            with open(recovery_path, 'r') as f:
                recovery_content = f.read()
            components.html(recovery_content, height=650, scrolling=True)
        else:
            st.info("Generate animation by running `python -m visuals.animated_recovery_paths`")
    
    with tab2:
        st.markdown("### Prediction Uncertainty Over Time")
        uncertainty_path = FIG_DIR / "recovery_uncertainty.html"
        if uncertainty_path.exists():
            with open(uncertainty_path, 'r') as f:
                uncertainty_content = f.read()
            components.html(uncertainty_content, height=750, scrolling=True)
    
    with tab3:
        st.markdown("### Individual Motor Function Recovery")
        heatmap_path = FIG_DIR / "motor_recovery_heatmap.html"
        if heatmap_path.exists():
            with open(heatmap_path, 'r') as f:
                heatmap_content = f.read()
            components.html(heatmap_content, height=750, scrolling=True)

elif section == "📉 Calibration & Reliability":
    st.header("📉 Model Calibration & Reliability")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        show_image_if_exists(FIG_DIR / "calibration_curve_enhanced.png", 
                           caption="Enhanced calibration curve with confidence intervals")
    
    with col2:
        st.markdown("""
        ### Key Insights
        
        ✅ **Well-calibrated** at extreme scores (0 and 5)
        
        ⚠️ **Slight over-prediction** in mid-range (2-3)
        
        📊 **Confidence intervals** show prediction reliability
        
        🎯 **RMSE < 0.90** demonstrates strong performance
        """)

elif section == "🔥 Residual Analysis":
    st.header("🔥 Residual Heatmap")
    show_image_if_exists(FIG_DIR / "residuals_heatmap.png")

elif section == "👥 Patient Demographics":
    st.header("👥 Patient Demographics – Best vs Worst")
    show_image_if_exists(FIG_DIR / "patient_group_demographics.png")

elif section == "💡 Key Findings":
    st.header("💡 Key Findings & Clinical Insights")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Model Performance
        - **RMSE < 0.90** on validation set
        - Well-calibrated predictions
        - Robust across patient subgroups
        
        ### 🔍 Feature Importance
        - **Age** is the strongest predictor
        - Initial AIS grade critical
        - Proximal muscle strength key indicator
        """)
    
    with col2:
        st.markdown("""
        ### 🏥 Clinical Relevance
        - Enables early intervention planning
        - Supports realistic goal-setting
        - Identifies high-risk patients
        
        ### 💡 Novel Insights
        - Non-linear recovery patterns
        - Demographic disparities identified
        - Interaction effects discovered
        """)
    
    st.markdown("""
    <div class="highlight-box">
    <h4>🚀 Impact Statement</h4>
    This model transforms how clinicians approach spinal cord injury recovery prediction, 
    enabling personalized treatment plans and improving patient outcomes through data-driven insights.
    </div>
    """, unsafe_allow_html=True)

elif section == "🚀 Next Steps":
    st.header("🚀 Next Steps")
    st.markdown("""
    • Incorporate MRI imaging & longitudinal fine-tuning.
    
    • Develop bedside decision-support prototype.
    """)

elif section == "🙏 Acknowledgements":
    st.header("🙏 Acknowledgements")
    st.markdown("""
    • Craig Hospital Research Department & SCI Model Systems.
    
    • Collaborators, clinicians, and patients contributing data.
    
    • Kaggle & ASIA organising committee.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "🏆 **Kaggle Competition Winner**\n\n"
    "[GitHub](https://github.com/user/kaggle_winning_solution) | "
    "[Competition](https://www.kaggle.com/competitions/asia-impairment-track-prediction)"
)

# Add session state for interactive features
if 'patient_idx' not in st.session_state:
    st.session_state.patient_idx = 0

if __name__ == "__main__":
    # When executed directly (e.g., `python streamlit_app.py`) show message.
    # Proper launch is `streamlit run streamlit_app.py`.
    print("Run with:  streamlit run streamlit_app.py")
    sys.exit(0)
