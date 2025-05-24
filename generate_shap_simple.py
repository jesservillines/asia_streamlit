"""Generate simple SHAP visualizations that work with MultiOutputRegressor."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

# Add project to path
PROJECT_ROOT = Path(__file__).parent
PKG_DIR = PROJECT_ROOT / "asia-impairment-track-prediction"
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_DIR = PKG_DIR / "data"
MODELS_DIR = PKG_DIR / "models_exact"
FIG_DIR = PKG_DIR / "visuals" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Target columns
TARGET_COLS = [
    'UEMS_6mo', 'LEMS_6mo', 'UEMS_12mo', 'LEMS_12mo',
    'LT_6mo', 'PP_6mo', 'LT_12mo', 'PP_12mo',
    'AIS_6mo_A', 'AIS_6mo_B', 'AIS_6mo_C', 'AIS_6mo_D',
    'AIS_12mo_A', 'AIS_12mo_B', 'AIS_12mo_C', 'AIS_12mo_D',
    'walking_6mo_dependent', 'walking_6mo_independent',
    'walking_12mo_dependent', 'walking_12mo_independent'
]

def create_shap_summary_plot():
    """Create a simple SHAP summary plot for the first target."""
    print("Loading data and models...")
    
    # Load data
    df = pd.read_csv(DATA_DIR / "train_data.csv")
    
    # Load model - try different model types
    model_files = list(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        print("No model files found!")
        return
    
    # Try to find a single model (not MultiOutput)
    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            model_name = model_file.stem
            print(f"Loaded model: {model_name}")
            
            # Get feature columns
            feature_cols = [c for c in df.columns if c not in TARGET_COLS]
            X = df[feature_cols].iloc[:100]  # Use subset for speed
            
            # Try to create explainer
            if hasattr(model, 'estimators_'):
                # It's a MultiOutputRegressor - use first estimator
                single_model = model.estimators_[0]
                explainer = shap.TreeExplainer(single_model)
                shap_values = explainer.shap_values(X)
                
                # Create summary plot
                fig = go.Figure()
                
                # Get feature importance
                feature_importance = np.abs(shap_values).mean(axis=0)
                top_features_idx = np.argsort(feature_importance)[-20:]
                
                # Create bar plot
                fig.add_trace(go.Bar(
                    x=feature_importance[top_features_idx],
                    y=[feature_cols[i] for i in top_features_idx],
                    orientation='h',
                    marker_color='#0066cc'
                ))
                
                fig.update_layout(
                    title=f"SHAP Feature Importance - {TARGET_COLS[0]}",
                    xaxis_title="Mean |SHAP value|",
                    yaxis_title="Features",
                    height=600,
                    width=800,
                    template="plotly_white"
                )
                
                # Save as HTML
                output_path = FIG_DIR / "shap_feature_importance.html"
                fig.write_html(str(output_path))
                print(f"✅ Saved SHAP feature importance to {output_path}")
                
                # Also create a waterfall plot for a single prediction
                fig2 = go.Figure()
                
                # Get SHAP values for first patient
                patient_idx = 0
                shap_vals = shap_values[patient_idx]
                feature_vals = X.iloc[patient_idx].values
                
                # Sort by absolute SHAP value
                sorted_idx = np.argsort(np.abs(shap_vals))[-15:]
                
                # Create waterfall data
                y_pos = list(range(len(sorted_idx)))
                
                fig2.add_trace(go.Bar(
                    x=shap_vals[sorted_idx],
                    y=[feature_cols[i] for i in sorted_idx],
                    orientation='h',
                    marker_color=['red' if x < 0 else 'blue' for x in shap_vals[sorted_idx]],
                    text=[f"{feature_vals[i]:.2f}" for i in sorted_idx],
                    textposition='outside'
                ))
                
                fig2.update_layout(
                    title="SHAP Waterfall - Single Patient Example",
                    xaxis_title="SHAP value",
                    yaxis_title="Features",
                    height=600,
                    width=800,
                    template="plotly_white",
                    showlegend=False
                )
                
                # Save waterfall
                output_path2 = FIG_DIR / "shap_waterfall.html"
                fig2.write_html(str(output_path2))
                print(f"✅ Saved SHAP waterfall to {output_path2}")
                
                return
                
        except Exception as e:
            print(f"Error with {model_file.name}: {e}")
            continue
    
    print("❌ Could not generate SHAP visualizations")

if __name__ == "__main__":
    create_shap_summary_plot()
