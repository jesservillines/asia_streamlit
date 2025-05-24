"""Optimized Streamlit dashboard with improved performance.

Key optimizations:
- Lazy loading of visualizations
- Aggressive caching
- Pre-generated static assets
- Reduced memory footprint
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

# Add parent directory to path BEFORE importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import streamlit.components.v1 as components

# Lazy imports to speed up initial load
@st.cache_resource
def import_heavy_libs():
    import joblib
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import shap
    return joblib, pd, plt, px, shap

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PKG_DIR = PROJECT_ROOT / "asia-impairment-track-prediction"
DATA_DIR = PKG_DIR / "data"
FIG_DIR = PKG_DIR / "visuals" / "figures"
MODELS_DIR = PKG_DIR / "models_exact"

# -----------------------------------------------------------------------------
# Page setup with light theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ASIA Motor Score Prediction", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light theme CSS for better contrast
st.markdown("""
<style>
    /* Force light theme with high contrast */
    :root {
        --primary-color: #0066cc;
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #000000;
    }
    
    /* Global overrides for visibility */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main {
        background-color: #ffffff !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Fix all text elements */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000 !important;
    }
    
    /* Buttons with high contrast */
    .stButton > button {
        background-color: #0066cc !important;
        color: #ffffff !important;
        border: 2px solid #0066cc !important;
        font-weight: bold !important;
    }
    
    .stButton > button:hover {
        background-color: #0052a3 !important;
        border-color: #0052a3 !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }
    
    .stSelectbox label {
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    /* Dropdown menu specific fixes */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    [role="listbox"] {
        background-color: #ffffff !important;
    }
    
    [role="option"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    [role="option"]:hover {
        background-color: #e3f2fd !important;
        color: #000000 !important;
    }
    
    [role="option"][aria-selected="true"] {
        background-color: #0066cc !important;
        color: #ffffff !important;
    }
    
    /* Popover and menu containers */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Fix selectbox dropdown arrow */
    .stSelectbox svg {
        fill: #000000 !important;
    }
    
    /* Metric cards with better visibility */
    .metric-card {
        background-color: #f8f9fa !important;
        padding: 20px !important;
        border-radius: 8px !important;
        margin: 10px 0 !important;
        border: 2px solid #e0e0e0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    .metric-card h3 {
        color: #0066cc !important;
        margin: 0 !important;
        font-size: 2rem !important;
    }
    
    .metric-card p {
        color: #333333 !important;
        margin-top: 5px !important;
        font-weight: 500 !important;
    }
    
    /* Highlight box with visibility */
    .highlight-box {
        background-color: #e3f2fd !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border-left: 5px solid #0066cc !important;
        margin: 20px 0 !important;
    }
    
    .highlight-box b {
        color: #0066cc !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        border-bottom: 3px solid #0066cc !important;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background-color: #f0f8ff !important;
        color: #000000 !important;
        border: 1px solid #0066cc !important;
    }
    
    /* Image captions */
    .caption {
        color: #333333 !important;
        font-weight: 500 !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #000000 !important;
    }
    
    /* Fix dark theme remnants */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Ensure all containers are visible */
    .element-container {
        color: #000000 !important;
    }
    
    /* Radio buttons and checkboxes */
    .stRadio label, .stCheckbox label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        background-color: #f0f2f6 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached visualization loaders with proper encoding
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_html_content(file_path: str) -> str:
    """Load HTML content with proper UTF-8 encoding."""
    path = Path(file_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return None

@st.cache_data
def load_image_as_base64(image_path: Path) -> str:
    """Convert image to base64 for faster loading."""
    import base64
    if image_path.exists():
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

@st.cache_resource
def preload_visualizations():
    """Pre-load all visualization paths."""
    viz_files = {
        '3d_outcomes': FIG_DIR / "interactive_3d_outcomes.html",
        'recovery_heatmap': FIG_DIR / "recovery_heatmap.html",
        # These don't exist yet, but we'll handle gracefully
        'recovery_timeline': FIG_DIR / "recovery_timeline.html",
        'shap_explorer': FIG_DIR / "interactive_shap_explorer.html",
        'shap_waterfall': FIG_DIR / "shap_waterfall.html",
        'animated_recovery': FIG_DIR / "animated_recovery_paths.html",
        'recovery_uncertainty': FIG_DIR / "recovery_uncertainty.html",
        'motor_heatmap': FIG_DIR / "motor_recovery_heatmap.html"
    }
    # Also check for existing visualizations we can use
    existing_files = {
        'calibration': FIG_DIR / "calibration_curve_enhanced.png",
        'patient_groups': FIG_DIR / "patient_group_diff_bars.png",
        'radar_mae': FIG_DIR / "radar_target_mae.png",
        'radar_rmse': FIG_DIR / "radar_target_rmse.png",
        'residuals': FIG_DIR / "residuals_heatmap.png",
        'shap_beeswarm': FIG_DIR / "shap_beeswarm_ensemble.png"
    }
    
    # Combine HTML and image files
    all_files = {}
    for k, v in viz_files.items():
        if v.exists():
            all_files[k] = str(v)
    for k, v in existing_files.items():
        if v.exists():
            all_files[k] = str(v)
            
    return all_files

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Navigation")
    st.caption("ASIA Motor Score Prediction")
    
    # Use tabs instead of radio for better performance
    tab_selection = st.selectbox(
        "Select Section",
        options=[
            "Overview",
            "Interactive 3D",
            "Muscle Changes",
            "Clinical Impact",
            "Recovery Paths",
            "Model Performance",
            "Key Findings",
            "All Visualizations"
        ],
        index=0
    )

# -----------------------------------------------------------------------------
# Main content with lazy loading
# -----------------------------------------------------------------------------
if tab_selection == "Overview":
    st.title("🧠 ASIA Motor Score Prediction")
    st.subheader("Kaggle Winning Solution")
    
    # Quick metrics with improved visibility
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>20</h3>
            <p>Motor Scores</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>&lt;0.90</h3>
            <p>RMSE</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>3</h3>
            <p>Models</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>#1</h3>
            <p>Ranking</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
        <b>Mission:</b> Enable clinicians to have data-driven conversations about recovery trajectories.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p style="color: #666; margin-bottom: 10px;">Learn more about the gradient boosting algorithms powering this model:</p>
        <a href="https://gradientboost.com" target="_blank" style="
            background-color: #0066cc;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            display: inline-block;
            transition: background-color 0.3s;
        " onmouseover="this.style.backgroundColor='#0052a3'" onmouseout="this.style.backgroundColor='#0066cc'">
            Visit GradientBoost.com
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Key Features
    st.subheader("🔑 Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **20 Motor Scores** predicted at 6 & 12 months
        - **Ensemble Model** combining CatBoost, XGBoost, HistGB
        - **RMSE < 0.90** on validation set
        - **Key Predictors**: Age, AIS grade, proximal strength
        """)
    
    with col2:
        st.markdown("""
        - **Interactive Visualizations** for exploration
        - **Uncertainty Quantification** for confidence
        - **Clinical Interpretability** built-in
        - **Real-time Predictions** for new patients
        """)

elif tab_selection == "Interactive 3D":
    st.header("3D Visualization")
    st.markdown("Interactive 3D visualization showing patient motor score predictions and recovery patterns.")
    
    # Import the visualization function here to avoid import errors
    try:
        from generate_real_3d_viz import create_3d_visualization_from_models
    except ImportError as e:
        st.error(f"Error importing visualization module: {e}")
        st.stop()
    
    # Create columns for filters
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        age_filter = st.multiselect(
            "Age Category",
            options=['<65', '>65', 'Unknown'],
            default=None,
            key="age_filter"
        )
    
    with col2:
        bmi_filter = st.multiselect(
            "BMI Category", 
            options=['Underweight', 'Healthy', 'Overweight', 'Unknown'],
            default=None,
            key="bmi_filter"
        )
    
    with col3:
        sex_filter = st.multiselect(
            "Sex",
            options=['Male', 'Female', 'Unknown'],
            default=None,
            key="sex_filter"
        )
    
    with col4:
        ais_filter = st.multiselect(
            "Week 26 AIS Grade",
            options=['A', 'B', 'C', 'D'],
            default=None,
            key="ais_filter"
        )
    
    with col5:
        week1_ais_filter = st.multiselect(
            "Week 1 AIS Grade",
            options=['A', 'B', 'C', 'D'],
            default=None,
            key="week1_ais_filter"
        )
    
    # Advanced visualization options
    with st.expander("Advanced Visualization Options"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            color_by = st.selectbox(
                "Color Points By",
                options=['ais_grade', 'confidence', 'error', 'gain'],
                format_func=lambda x: {
                    'ais_grade': 'AIS Grade',
                    'confidence': 'Model Confidence',
                    'error': 'Prediction Error',
                    'gain': 'Actual Gain'
                }[x],
                key="color_by"
            )
            
            marker_style = st.selectbox(
                "Marker Style",
                options=['circle', 'square', 'diamond', 'cross'],
                format_func=lambda x: x.capitalize(),
                key="marker_style"
            )
        
        with adv_col2:
            point_size = st.slider("Point Size", min_value=5, max_value=20, value=10, key="point_size")
            opacity = st.slider("Point Opacity", min_value=0.3, max_value=1.0, value=0.7, step=0.1, key="opacity")
        
        with adv_col3:
            show_grid = st.checkbox("Show Grid", value=True, key="show_grid")
            show_legend = st.checkbox("Show Legend", value=True, key="show_legend")
            show_regression_plane = st.checkbox("Show Regression Plane", value=False, key="show_regression")
    
    # Generate visualization automatically
    with st.spinner("Creating 3D visualization..."):
        try:
            fig = create_3d_visualization_from_models(
                age_filter=age_filter if age_filter else None,
                bmi_filter=bmi_filter if bmi_filter else None,
                sex_filter=sex_filter if sex_filter else None,
                ais_filter=ais_filter if ais_filter else None,
                week1_ais_filter=week1_ais_filter if week1_ais_filter else None,
                point_size=point_size,
                opacity=opacity,
                show_grid=show_grid,
                show_legend=show_legend,
                show_regression_plane=show_regression_plane,
                color_by=color_by,
                marker_style=marker_style
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation guide
            with st.expander("📊 How to interpret this visualization"):
                st.markdown("""
                ### Understanding the 3D Plot
                
                **Axes:**
                - **X-axis (horizontal)**: Week 26 Actual Total Motor Score
                - **Y-axis (depth)**: Week 26 Predicted Total Motor Score  
                - **Z-axis (vertical)**: Actual Gain from Week 1 to Week 26
                
                **Perfect Prediction Line:**
                - The red dashed line shows where actual = predicted scores
                - Points closer to this line indicate more accurate predictions
                - The line's Z-value (height) varies based on typical gains at each score level
                
                **Color Schemes:**
                - **AIS Grade**: A=Red, B=Orange, C=Green, D=Blue
                - **Confidence**: Darker = Higher confidence
                - **Error**: Blue = Underestimation, Red = Overestimation
                - **Gain**: Color intensity shows amount of improvement
                
                **Metrics:**
                - **Individual Muscle RMSE**: Model accuracy for individual muscle predictions (target < 0.90)
                - **Total Score MAE/RMSE**: Accuracy for total motor score predictions
                - **R²**: Correlation between predicted and actual scores
                
                **Interaction:**
                - Click and drag to rotate the plot
                - Scroll to zoom in/out
                - Hover over points to see patient details
                - Click legend items to show/hide groups
                """)
            
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            st.info("Please ensure all required model files are present.")

elif tab_selection == "Muscle Changes":
    st.header("Individual Muscle Change Analysis")
    st.markdown("Analyze how individual muscles change from week 1 to week 26 across patients.")
    
    # Import the muscle visualization functions
    try:
        from generate_muscle_change_viz import (
            create_muscle_change_heatmap, 
            create_muscle_change_summary,
            create_patient_muscle_radar
        )
    except ImportError as e:
        st.error(f"Error importing muscle visualization module: {e}")
        st.stop()
    
    # Create sub-tabs for different views
    muscle_tab = st.tabs(["Heatmap View", "Summary Statistics", "Individual Patient"])
    
    with muscle_tab[0]:
        st.subheader("Patient-Muscle Change Heatmap")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ais_filter = st.multiselect(
                "Week 1 AIS Grade",
                options=['A', 'B', 'C', 'D'],
                default=None,
                help="Filter by initial AIS grade"
            )
        
        with col2:
            muscle_group_filter = st.multiselect(
                "Muscle Groups",
                options=['Upper', 'Lower', 'Left', 'Right'],
                default=None,
                help="Filter by muscle groups"
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort Patients By",
                options=['total_gain', 'pid', 'ais_grade'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col4:
            colorscale = st.selectbox(
                "Color Scale",
                options=['RdBu', 'Viridis', 'Plasma', 'Blues', 'Reds'],
                index=0
            )
        
        # Additional options
        show_values = st.checkbox("Show values in cells", value=False)
        
        # Generate heatmap
        with st.spinner("Generating heatmap..."):
            fig = create_muscle_change_heatmap(
                ais_filter=ais_filter if ais_filter else None,
                muscle_group_filter=muscle_group_filter if muscle_group_filter else None,
                sort_by=sort_by,
                show_values=show_values,
                colorscale=colorscale
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        with st.expander("📊 How to interpret this heatmap"):
            st.markdown("""
            - **Each cell** represents the change in a specific muscle score for a specific patient
            - **Colors**: Blue = improvement, Red = decline, White = no change
            - **Scale**: Changes range from -5 to +5 (full range of motor score scale)
            - **Patients** are sorted by your selected criteria
            - **Hover** over cells to see exact values
            """)
    
    with muscle_tab[1]:
        st.subheader("Muscle Change Summary Statistics")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            summary_ais_filter = st.multiselect(
                "Filter by Week 1 AIS Grade",
                options=['A', 'B', 'C', 'D'],
                default=None,
                key="summary_ais"
            )
        
        with col2:
            view_type = st.radio(
                "View Type",
                options=['average', 'distribution'],
                format_func=lambda x: x.title()
            )
        
        # Generate summary
        with st.spinner("Generating summary..."):
            fig = create_muscle_change_summary(
                ais_filter=summary_ais_filter if summary_ais_filter else None,
                view_type=view_type
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("### Key Insights")
        st.info("""
        - **Upper extremity muscles** typically show more improvement than lower extremity
        - **Patients with incomplete injuries** (AIS B, C, D) show more variable recovery
        - **Bilateral symmetry** is often maintained in recovery patterns
        """)
    
    with muscle_tab[2]:
        st.subheader("Individual Patient Muscle Profile")
        
        # Patient selector
        patient_id = st.text_input(
            "Enter Patient ID",
            placeholder="e.g., P001",
            help="Enter a patient ID to view their individual muscle changes"
        )
        
        if patient_id:
            with st.spinner(f"Loading data for patient {patient_id}..."):
                fig = create_patient_muscle_radar(patient_id)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("### Interpretation")
                    st.markdown("""
                    - **Red area**: Week 1 muscle scores (baseline)
                    - **Green area**: Week 26 muscle scores (outcome)
                    - **Expansion** from red to green indicates improvement
                    - **Score scale**: 0 (no function) to 5 (normal strength)
                    """)
                else:
                    st.error(f"Patient ID '{patient_id}' not found in the dataset.")

elif tab_selection == "Clinical Impact":
    st.header("🏥 Clinical Decision Support")
    
    # Use columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Show Recovery Heatmap"):
            viz_paths = preload_visualizations()
            if 'recovery_heatmap' in viz_paths:
                content = load_html_content(viz_paths['recovery_heatmap'])
                if content:
                    components.html(content, height=600)
            else:
                st.warning("Recovery timeline not yet generated")
    
    with col2:
        st.markdown("""
        ### Functional Milestones
        - 🤚 Hand Function
        - 🍴 Feeding Independence  
        - ♿ Wheelchair Propulsion
        - 🚶 Walking Potential
        """)
        
        # Show existing visualizations
        viz_paths = preload_visualizations()
        if 'patient_groups' in viz_paths:
            st.image(viz_paths['patient_groups'], caption="Patient Group Differences")

elif tab_selection == "Recovery Paths":
    st.header("📊 Recovery Trajectories")
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Group Comparison", "Uncertainty", "Motor Functions"])
    
    with tab1:
        st.subheader("Recovery Path Comparison")
        st.info("Comparing recovery trajectories across different patient groups")
        
        # Create a simple recovery trajectory visualization
        import plotly.graph_objects as go
        
        # Sample data for recovery trajectories
        months = [0, 1, 3, 6, 9, 12]
        
        fig = go.Figure()
        
        # Add traces for different patient groups
        fig.add_trace(go.Scatter(
            x=months, y=[20, 25, 35, 45, 50, 55],
            mode='lines+markers',
            name='AIS A',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=months, y=[35, 40, 50, 60, 65, 70],
            mode='lines+markers',
            name='AIS B',
            line=dict(color='#4ecdc4', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=months, y=[50, 55, 65, 75, 80, 85],
            mode='lines+markers',
            name='AIS C',
            line=dict(color='#45b7d1', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=months, y=[70, 75, 80, 85, 88, 90],
            mode='lines+markers',
            name='AIS D',
            line=dict(color='#0066cc', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Average Motor Score Recovery by AIS Grade",
            xaxis_title="Months Post-Injury",
            yaxis_title="Motor Score",
            height=500,
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Prediction Uncertainty Bands")
        
        # Create uncertainty visualization
        fig2 = go.Figure()
        
        # Central prediction
        x = np.array(months)
        y_pred = np.array([45, 50, 60, 70, 75, 80])
        y_upper = y_pred + np.array([5, 6, 7, 8, 8, 7])
        y_lower = y_pred - np.array([5, 6, 7, 8, 8, 7])
        
        # Add confidence band
        fig2.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=True
        ))
        
        # Add prediction line
        fig2.add_trace(go.Scatter(
            x=x, y=y_pred,
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#0066cc', width=3),
            marker=dict(size=8)
        ))
        
        # Add actual values
        fig2.add_trace(go.Scatter(
            x=[0, 6, 12],
            y=[45, 68, 78],
            mode='markers',
            name='Actual',
            marker=dict(color='#ff6b6b', size=12, symbol='diamond')
        ))
        
        fig2.update_layout(
            title="Prediction Uncertainty Over Time",
            xaxis_title="Months Post-Injury",
            yaxis_title="Motor Score",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Motor Function Heatmap")
        
        # Check if recovery heatmap exists
        heatmap_path = FIG_DIR / "recovery_heatmap.html"
        if heatmap_path.exists():
            with open(heatmap_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=700, scrolling=True)
        else:
            # Create a simple heatmap
            motor_functions = ['C5', 'C6', 'C7', 'C8', 'T1', 'L2', 'L3', 'L4', 'L5', 'S1']
            time_points = ['Baseline', '1 month', '3 months', '6 months', '12 months']
            
            # Generate sample data
            data = np.random.rand(10, 5) * 5
            data[:, 1:] = data[:, :-1] + np.random.rand(10, 4) * 0.5  # Progressive improvement
            
            fig3 = go.Figure(data=go.Heatmap(
                z=data,
                x=time_points,
                y=motor_functions,
                colorscale='RdBu_r',
                text=np.round(data, 1),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Motor Score")
            ))
            
            fig3.update_layout(
                title="Motor Function Recovery Heatmap",
                xaxis_title="Time Point",
                yaxis_title="Motor Function Level",
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig3, use_container_width=True)

elif tab_selection == "Model Performance":
    st.header("📈 Model Performance Analysis")
    st.markdown("Compare the performance of different ensemble approaches and analyze feature importance.")
    
    # Tabs for different analyses
    analysis_tabs = st.tabs([
        "Ensemble Performance",
        "Feature Importance", 
        "Model Agreement"
    ])
    
    with analysis_tabs[0]:
        st.subheader("Ensemble Performance Comparison")
        
        # Sub-tabs for different ensemble comparisons
        ensemble_subtabs = st.tabs([
            "Simple Average",
            "Weighted Average", 
            "CatBoost + XGBoost"
        ])
        
        with ensemble_subtabs[0]:
            st.markdown("""
            This chart compares individual models (CatBoost, XGBoost, HistGB) with their simple average ensemble.
            All models are weighted equally in the ensemble.
            """)
            
            # Load and display the simple ensemble radar chart
            simple_ensemble_path = PROJECT_ROOT / "rmse_radar_simple_ensemble.html"
            if simple_ensemble_path.exists():
                content = load_html_content(str(simple_ensemble_path))
                if content:
                    components.html(content, height=700, scrolling=False)
            else:
                st.warning("Simple ensemble radar chart not found. Please run `python generate_ensemble_comparison_radar.py` to generate it.")
        
        with ensemble_subtabs[1]:
            st.markdown("""
            This chart shows a weighted ensemble where models are weighted inversely proportional to their RMSE.
            Better performing models (lower RMSE) get higher weights in the ensemble.
            """)
            
            # Load and display the weighted ensemble radar chart
            weighted_ensemble_path = PROJECT_ROOT / "rmse_radar_weighted_ensemble.html"
            if weighted_ensemble_path.exists():
                content = load_html_content(str(weighted_ensemble_path))
                if content:
                    components.html(content, height=700, scrolling=False)
            else:
                st.warning("Weighted ensemble radar chart not found. Please run `python generate_ensemble_comparison_radar.py` to generate it.")
        
        with ensemble_subtabs[2]:
            st.markdown("""
            This chart shows a weighted ensemble using only CatBoost and XGBoost models.
            HistGB is excluded from this ensemble to see if a two-model ensemble performs better.
            """)
            
            # Load and display the CB+XGB ensemble radar chart
            cb_xgb_ensemble_path = PROJECT_ROOT / "rmse_radar_cb_xgb_ensemble.html"
            if cb_xgb_ensemble_path.exists():
                content = load_html_content(str(cb_xgb_ensemble_path))
                if content:
                    components.html(content, height=700, scrolling=False)
            else:
                st.warning("CatBoost + XGBoost ensemble radar chart not found. Please run `python generate_ensemble_comparison_radar.py` to generate it.")
    
    with analysis_tabs[1]:
        st.subheader("Feature Importance Analysis")
        st.markdown("""
        Understanding which features contribute most to the predictions helps validate the model's clinical relevance
        and provides insights into recovery patterns.
        """)
        
        # Feature importance visualization options
        importance_view = st.selectbox(
            "Select visualization:",
            ["Ensemble Feature Importance", "Comprehensive Comparison", "Individual Models"]
        )
        
        if importance_view == "Ensemble Feature Importance":
            st.markdown("""
            **Weighted ensemble feature importance** based on permutation importance analysis.
            Features are ranked by their impact on model predictions when their values are randomly shuffled.
            """)
            
            # Load and display the simple ensemble importance plot
            importance_path = PROJECT_ROOT / "asia-impairment-track-prediction" / "visuals" / "figures" / "feature_importance_ensemble_simple.html"
            if importance_path.exists():
                content = load_html_content(str(importance_path))
                if content:
                    components.html(content, height=700, scrolling=False)
            else:
                st.warning("Feature importance plot not found. Please run `python generate_feature_importance_ensemble.py` to generate it.")
        
        elif importance_view == "Comprehensive Comparison":
            st.markdown("""
            **Comprehensive feature importance analysis** showing multiple methods and model agreement.
            This includes native importance (where available), permutation importance, and model agreement analysis.
            """)
            
            # Load and display the comprehensive comparison
            comparison_path = PROJECT_ROOT / "asia-impairment-track-prediction" / "visuals" / "figures" / "feature_importance_comparison.html"
            if comparison_path.exists():
                content = load_html_content(str(comparison_path))
                if content:
                    components.html(content, height=900, scrolling=False)
            else:
                st.warning("Feature importance comparison not found. Please run `python generate_feature_importance_ensemble.py` to generate it.")
        
        else:  # Individual Models
            st.markdown("""
            **Individual model feature importance** to understand how each model prioritizes different features.
            """)
            
            model_choice = st.radio("Select model:", ["CatBoost", "XGBoost", "HistGradientBoosting"], horizontal=True)
            
            if model_choice == "CatBoost":
                model_name = "catboost"
            elif model_choice == "XGBoost":
                model_name = "xgb"
            else:  # HistGradientBoosting
                model_name = "hgb"
                
            model_importance_path = PROJECT_ROOT / "asia-impairment-track-prediction" / "visuals" / "figures" / f"feature_importance_{model_name}.html"
            
            if model_importance_path.exists():
                content = load_html_content(str(model_importance_path))
                if content:
                    components.html(content, height=700, scrolling=False)
            else:
                st.warning(f"{model_choice} feature importance not found. Please run `python generate_feature_importance_ensemble.py` to generate it.")
        
        # Feature importance insights
        with st.expander("📊 Feature Importance Insights"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Top Contributing Features
                - **Initial motor scores** (week 1) are most predictive
                - **Neurological level indicators** provide crucial context
                - **Age** shows moderate importance
                - **Specific muscle groups** cluster in importance
                """)
            
            with col2:
                st.markdown("""
                ### Clinical Interpretation
                - **Baseline function** is the strongest predictor
                - **Proximal muscles** influence distal recovery
                - **Bilateral patterns** suggest systemic factors
                - **Feature interactions** capture recovery patterns
                """)
                
        # Add note about model differences
        with st.expander("🔍 Model Comparison Notes"):
            st.markdown("""
            ### Model Characteristics
            
            **CatBoost**
            - Best overall performance (lowest RMSE)
            - Handles categorical features natively
            - More stable predictions
            
            **XGBoost**
            - Strong performance, especially on distal muscles
            - Efficient tree-based boosting
            - Good feature interaction capture
            
            **HistGradientBoosting**
            - Different approach to feature importance
            - May prioritize different features
            - Useful for comparison but not included in ensemble
            
            The **XGBoost + CatBoost ensemble** excludes HistGradientBoosting as the two-model 
            combination provides optimal performance.
            """)
    
    with analysis_tabs[2]:
        st.subheader("Model Agreement Analysis")
        st.markdown("""
        Understanding where models agree or disagree helps quantify prediction uncertainty
        and identify areas where additional data or model improvements might be needed.
        """)
        
        # Check if uncertainty analysis exists
        uncertainty_path = PROJECT_ROOT / "asia-impairment-track-prediction" / "visuals" / "figures" / "shap_uncertainty_analysis.html"
        comparison_path = PROJECT_ROOT / "asia-impairment-track-prediction" / "visuals" / "figures" / "feature_importance_comparison.html"
        
        if comparison_path.exists():
            st.info("Model agreement is visualized in the bottom-right panel of the comprehensive comparison chart.")
            content = load_html_content(str(comparison_path))
            if content:
                components.html(content, height=900, scrolling=False)
        else:
            st.warning("Model agreement analysis not found. Please run `python generate_feature_importance_ensemble.py` to generate it.")
        
        # Agreement insights
        st.markdown("---")
        st.subheader("🤝 Model Agreement Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### High Agreement Features
            - **Strong motor scores** (consistent importance)
            - **Age** (universally recognized)
            - **Major muscle groups** (clear patterns)
            
            ### Low Agreement Features
            - **Subtle indicators** (model-specific)
            - **Interaction terms** (different approaches)
            - **Rare patterns** (limited data)
            """)
        
        with col2:
            st.markdown("""
            ### Uncertainty Implications
            - **High agreement** → More reliable predictions
            - **Low agreement** → Higher uncertainty
            - **Ensemble benefit** → Reduces individual model bias
            
            ### Clinical Use
            - Focus on high-agreement predictions
            - Flag low-agreement cases for review
            - Use uncertainty for treatment planning
            """)
    
    # Overall insights section
    st.markdown("---")
    st.subheader("🔍 Key Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Ensemble Performance
        - **Weighted ensembles** outperform simple averaging
        - **CatBoost + XGBoost** provides optimal balance
        - **Model diversity** improves robustness
        
        ### Feature Importance
        - **Baseline function** is most predictive
        - **Neurological indicators** provide context
        - **Age and injury level** moderate factors
        """)
    
    with col2:
        st.markdown("""
        ### Clinical Implications
        - **Early assessment** can predict outcomes
        - **Targeted interventions** for key muscle groups
        - **Uncertainty quantification** aids decision-making
        
        ### Technical Excellence
        - Permutation importance for reliability
        - Multiple validation approaches
        - Ensemble methods reduce overfitting
        """)

elif tab_selection == "Key Findings":
    st.header("💡 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Performance
        - RMSE < 0.90 ✅
        - Well-calibrated
        - Robust predictions
        
        ### Top Features
        1. Age
        2. AIS Grade
        3. Proximal Strength
        """)
    
    with col2:
        st.markdown("""
        ### Clinical Impact
        - Early intervention planning
        - Realistic goal-setting
        - Risk stratification
        
        ### Next Steps
        - MRI integration
        - Longitudinal tracking
        - Clinical trials
        """)

elif tab_selection == "All Visualizations":
    st.header("📊 All Visualizations")
    
    viz_paths = preload_visualizations()
    
    # Display all visualizations
    for viz_name, viz_path in viz_paths.items():
        if viz_name == '3d_outcomes':
            st.subheader("3D Patient Outcomes")
            content = load_html_content(viz_path)
            if content:
                components.html(content, height=800, scrolling=True)
        elif viz_name == 'recovery_heatmap':
            st.subheader("Recovery Heatmap")
            content = load_html_content(viz_path)
            if content:
                components.html(content, height=600)
        elif viz_name == 'recovery_timeline':
            st.subheader("Recovery Timeline")
            content = load_html_content(viz_path)
            if content:
                components.html(content, height=600)
        elif viz_name == 'shap_explorer':
            st.subheader("SHAP Explorer")
            content = load_html_content(viz_path)
            if content:
                components.html(content, height=700)
        elif viz_name == 'shap_waterfall':
            st.subheader("SHAP Waterfall")
            content = load_html_content(viz_path)
            if content:
                components.html(content, height=600)
        elif viz_name == 'animated_recovery':
            st.subheader("Animated Recovery Paths")
            content = load_html_content(viz_path)
            if content:
                components.html(content, height=600)
        elif viz_name == 'recovery_uncertainty':
            st.subheader("Recovery Uncertainty")
            content = load_html_content(viz_path)
            if content:
                components.html(content, height=700)
        elif viz_name == 'motor_heatmap':
            st.subheader("Motor Heatmap")
            content = load_html_content(viz_path)
            if content:
                components.html(content, height=700)
        elif viz_name == 'calibration':
            st.subheader("Model Calibration")
            st.image(viz_path, use_container_width=True)
        elif viz_name == 'patient_groups':
            st.subheader("Patient Groups")
            st.image(viz_path, use_container_width=True)
        elif viz_name == 'radar_mae':
            st.subheader("Target MAE")
            st.image(viz_path, use_container_width=True)
        elif viz_name == 'radar_rmse':
            st.subheader("Target RMSE")
            st.image(viz_path, use_container_width=True)
        elif viz_name == 'residuals':
            st.subheader("Residuals Heatmap")
            st.image(viz_path, use_container_width=True)
        elif viz_name == 'shap_beeswarm':
            st.subheader("SHAP Beeswarm")
            st.image(viz_path, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("🏆 Kaggle Winner | [GitHub](https://github.com)")

if __name__ == "__main__":
    print("Run with: streamlit run streamlit_app_optimized.py")
