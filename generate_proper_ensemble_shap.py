"""Generate real SHAP visualizations using XGBoost + CatBoost weighted ensemble with uncertainty quantification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent
PKG_DIR = PROJECT_ROOT / "asia-impairment-track-prediction"
DATA_DIR = PKG_DIR / "data"
MODELS_DIR = PKG_DIR / "models_exact"
FIG_DIR = PKG_DIR / "visuals" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_data_and_models():
    """Load processed data and trained models - adapted from working radar chart"""
    print("Loading data and models...")
    
    # Load raw training features
    train_features = pd.read_csv(DATA_DIR / "train_features.csv")
    
    # Load processed training data
    train_df = pd.read_csv(DATA_DIR / "train_processed.csv")
    
    # Load outcomes
    outcomes_df = pd.read_csv(DATA_DIR / "train_outcomes_MS.csv")
    week26_outcomes = outcomes_df[outcomes_df['time'] == 26].copy()
    
    # Define target columns (lowercase, no week suffix)
    TARGET_COLS = [
        'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
        'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
        'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
        'gretor', 'ankplr'
    ]
    
    # Filter to patients that have both week 1 and week 26 data
    common_pids = set(train_features['PID']).intersection(set(week26_outcomes['PID']))
    common_pids = common_pids.intersection(set(train_df['PID']))
    common_pids = sorted(list(common_pids))
    
    print(f"Found {len(common_pids)} patients with complete data")
    
    # Filter dataframes to common patients and ensure same order
    train_df_filtered = train_df[train_df['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    week26_outcomes_filtered = week26_outcomes[week26_outcomes['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    
    # Prepare features for model prediction
    train_df_no_pid = train_df_filtered.drop(columns=['PID'])
    
    # Remove target columns from features
    X_cols = [c for c in train_df_no_pid.columns if c not in TARGET_COLS]
    X_train = train_df_no_pid[X_cols].values
    
    # Get actual week 26 values
    y_true = week26_outcomes_filtered[TARGET_COLS].values
    
    # Get PIDs for tracking
    pids = train_df_filtered['PID'].values
    
    # Load models - only XGB and CatBoost for ensemble
    models = {}
    model_names = ['catboost', 'xgb']
    
    for name in model_names:
        model_path = MODELS_DIR / f"{name}_exact_model.pkl"
        if model_path.exists():
            models[name] = joblib.load(model_path)
            print(f"Loaded {name} model")
        else:
            print(f"Warning: Model not found at {model_path}")
    
    return X_train, y_true, X_cols, models, pids, TARGET_COLS

def calculate_model_weights(models, X, y_true):
    """Calculate weights based on model performance."""
    print("\nCalculating model weights based on RMSE...")
    
    rmse_scores = {}
    predictions = {}
    
    for name, model in models.items():
        pred = model.predict(X)
        predictions[name] = pred
        
        # Calculate RMSE for each target
        rmse_per_target = []
        for i in range(y_true.shape[1]):
            rmse = np.sqrt(mean_squared_error(y_true[:, i], pred[:, i]))
            rmse_per_target.append(rmse)
        
        avg_rmse = np.mean(rmse_per_target)
        rmse_scores[name] = avg_rmse
        print(f"{name} average RMSE: {avg_rmse:.4f}")
    
    # Calculate weights (inverse RMSE)
    total_inv_rmse = sum(1/rmse for rmse in rmse_scores.values())
    weights = {name: (1/rmse)/total_inv_rmse for name, rmse in rmse_scores.items()}
    
    print("\nModel weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")
    
    return weights, predictions

def compute_shap_values(models, X, feature_names, sample_size=100):
    """Compute SHAP values for each model."""
    print(f"\nComputing SHAP values for {sample_size} samples...")
    
    # Use a subset for faster computation
    X_sample = X[:sample_size]
    
    shap_values_dict = {}
    base_values_dict = {}
    
    for name, model in models.items():
        print(f"  Computing SHAP values for {name}...")
        
        if name == 'xgb':
            # For XGBoost MultiOutputRegressor, handle each output separately
            shap_vals_list = []
            base_vals_list = []
            
            # Process each target
            for i in range(len(model.estimators_)):
                estimator = model.estimators_[i]
                explainer = shap.TreeExplainer(estimator)
                shap_vals = explainer.shap_values(X_sample)
                base_val = explainer.expected_value
                
                shap_vals_list.append(shap_vals)
                base_vals_list.append(base_val)
            
            # Stack into 3D array: (samples, features, targets)
            shap_values = np.stack(shap_vals_list, axis=2)
            base_values = np.array(base_vals_list)
            
        else:  # CatBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            base_values = explainer.expected_value
            
            # Ensure 3D shape
            if len(shap_values.shape) == 2:
                shap_values = shap_values[:, :, np.newaxis]
            if isinstance(base_values, (int, float)):
                base_values = np.array([base_values])
        
        shap_values_dict[name] = shap_values
        base_values_dict[name] = base_values
        
        print(f"    SHAP values shape: {shap_values.shape}")
        print(f"    Base values shape: {base_values.shape}")
    
    return shap_values_dict, base_values_dict, X_sample

def calculate_ensemble_shap_with_uncertainty(shap_values_dict, weights):
    """Calculate weighted ensemble SHAP values and uncertainty."""
    print("\nCalculating ensemble SHAP values with uncertainty...")
    
    # Calculate weighted ensemble
    ensemble_shap = None
    for name, shap_vals in shap_values_dict.items():
        if ensemble_shap is None:
            ensemble_shap = weights[name] * shap_vals
        else:
            ensemble_shap += weights[name] * shap_vals
    
    # Calculate uncertainty as weighted standard deviation
    shap_list = list(shap_values_dict.values())
    shap_array = np.stack(shap_list, axis=0)  # Shape: (n_models, n_samples, n_features, n_targets)
    
    # Calculate variance across models
    model_variance = np.var(shap_array, axis=0)
    model_std = np.sqrt(model_variance)
    
    # Also calculate the range (max - min) as another uncertainty measure
    model_range = np.max(shap_array, axis=0) - np.min(shap_array, axis=0)
    
    print(f"  Ensemble SHAP shape: {ensemble_shap.shape}")
    print(f"  Uncertainty (std) shape: {model_std.shape}")
    
    return ensemble_shap, model_std, model_range

def create_enhanced_shap_beeswarm_with_uncertainty(ensemble_shap, uncertainty, X_sample, feature_names, target_idx=0):
    """Create SHAP beeswarm plot with uncertainty visualization."""
    
    # Get SHAP values and uncertainty for specific target
    shap_target = ensemble_shap[:, :, target_idx]
    uncertainty_target = uncertainty[:, :, target_idx]
    
    # Calculate feature importance
    feature_importance = np.abs(shap_target).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[-20:][::-1]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each feature
    for i, feat_idx in enumerate(top_features_idx):
        feature_name = feature_names[feat_idx]
        shap_vals = shap_target[:, feat_idx]
        uncertainty_vals = uncertainty_target[:, feat_idx]
        feature_vals = X_sample[:, feat_idx]
        
        # Normalize feature values for color
        if feature_vals.max() > feature_vals.min():
            feat_norm = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
        else:
            feat_norm = np.ones_like(feature_vals) * 0.5
        
        # Add jitter for better visualization
        y_positions = np.ones_like(shap_vals) * i + np.random.normal(0, 0.1, size=len(shap_vals))
        
        # Size based on uncertainty (larger uncertainty = larger point)
        sizes = 4 + 10 * (uncertainty_vals / (uncertainty_vals.max() + 1e-8))
        
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=y_positions,
            mode='markers',
            marker=dict(
                size=sizes,
                color=feat_norm,
                colorscale='RdBu_r',
                showscale=(i == 0),
                colorbar=dict(
                    title="Feature value<br>(normalized)",
                    x=1.02
                ),
                line=dict(width=0.5, color='white'),
                opacity=0.8
            ),
            name=feature_name,
            showlegend=False,
            customdata=np.column_stack((feature_vals, uncertainty_vals)),
            hovertemplate=f"<b>{feature_name}</b><br>" +
                         "SHAP value: %{x:.3f}<br>" +
                         "Feature value: %{customdata[0]:.3f}<br>" +
                         "Uncertainty: %{customdata[1]:.3f}<br>" +
                         "<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="SHAP Beeswarm Plot - XGBoost + CatBoost Ensemble<br><sub>Point size indicates model uncertainty</sub>",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="SHAP value (impact on model output)",
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1.5
        ),
        yaxis=dict(
            title="Features",
            tickmode='array',
            tickvals=list(range(len(top_features_idx))),
            ticktext=[feature_names[idx] for idx in top_features_idx],
            gridcolor='rgba(128,128,128,0.2)'
        ),
        height=700,
        width=1000,
        template="plotly_white",
        hovermode='closest'
    )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_uncertainty_analysis_dashboard(ensemble_shap, model_std, model_range, feature_names):
    """Create dashboard showing uncertainty analysis."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Feature Importance with Uncertainty",
            "Model Agreement by Feature",
            "Uncertainty vs Impact",
            "Feature Uncertainty Distribution"
        ),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "box"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Calculate metrics across all targets
    mean_importance = np.abs(ensemble_shap).mean(axis=(0, 2))
    mean_uncertainty = model_std.mean(axis=(0, 2))
    mean_range = model_range.mean(axis=(0, 2))
    
    # Get top features
    top_features_idx = np.argsort(mean_importance)[-15:]
    
    # 1. Feature Importance with Error Bars
    fig.add_trace(
        go.Bar(
            y=[feature_names[i] for i in top_features_idx],
            x=mean_importance[top_features_idx],
            error_x=dict(
                type='data',
                array=mean_uncertainty[top_features_idx],
                visible=True
            ),
            orientation='h',
            marker=dict(
                color=mean_importance[top_features_idx],
                colorscale='Blues',
                showscale=False
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<br>Uncertainty: %{error_x.array:.3f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # 2. Model Agreement (inverse of range)
    agreement = 1 / (1 + mean_range[top_features_idx])
    
    fig.add_trace(
        go.Scatter(
            x=[feature_names[i] for i in top_features_idx],
            y=agreement,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=agreement,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Agreement",
                    x=1.15
                )
            ),
            line=dict(color='gray', width=1),
            hovertemplate="<b>%{x}</b><br>Model Agreement: %{y:.3f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # 3. Uncertainty vs Impact scatter
    fig.add_trace(
        go.Scatter(
            x=mean_importance,
            y=mean_uncertainty,
            mode='markers',
            marker=dict(
                size=8,
                color=mean_range,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title="Range",
                    x=1.15
                )
            ),
            text=[feature_names[i] for i in range(len(feature_names))],
            hovertemplate="<b>%{text}</b><br>Impact: %{x:.3f}<br>Uncertainty: %{y:.3f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 4. Uncertainty Distribution
    uncertainty_data = []
    for i in top_features_idx[:10]:
        uncertainty_data.append(
            go.Box(
                y=model_std[:, i, :].flatten(),
                name=feature_names[i][:15],  # Truncate long names
                marker_color='lightblue',
                boxpoints='outliers'
            )
        )
    
    for trace in uncertainty_data:
        fig.add_trace(trace, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="SHAP Uncertainty Analysis - Model Disagreement Quantification",
            font=dict(size=22)
        ),
        height=900,
        width=1400,
        showlegend=False,
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Mean |SHAP value|", row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(title_text="Feature Importance", row=2, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=2)
    
    fig.update_yaxes(title_text="Features", row=1, col=1)
    fig.update_yaxes(title_text="Model Agreement Score", row=1, col=2)
    fig.update_yaxes(title_text="Mean Uncertainty (Std Dev)", row=2, col=1)
    fig.update_yaxes(title_text="Uncertainty Distribution", row=2, col=2)
    
    return fig

def main():
    """Generate real SHAP visualizations with uncertainty quantification."""
    print("Starting SHAP visualization generation...")
    try:
        # Load data and models
        X, y_true, feature_names, models, pids, TARGET_COLS = load_data_and_models()
        
        if not models:
            print("No models found! Please ensure models are trained and saved.")
            return
        
        # Calculate model weights
        weights, predictions = calculate_model_weights(models, X, y_true)
        
        # Compute SHAP values
        shap_values_dict, base_values_dict, X_sample = compute_shap_values(
            models, X, feature_names, sample_size=min(200, len(X))
        )
        
        # Calculate ensemble SHAP with uncertainty
        ensemble_shap, model_std, model_range = calculate_ensemble_shap_with_uncertainty(
            shap_values_dict, weights
        )
        
        # Generate visualizations
        print("\nGenerating enhanced SHAP visualizations...")
        
        # 1. Enhanced Beeswarm with Uncertainty
        fig_beeswarm = create_enhanced_shap_beeswarm_with_uncertainty(
            ensemble_shap, model_std, X_sample, feature_names, target_idx=0
        )
        output_path = FIG_DIR / "shap_beeswarm_ensemble_real.html"
        fig_beeswarm.write_html(str(output_path))
        print(f"✅ Generated SHAP beeswarm with uncertainty: {output_path}")
        
        # 2. Uncertainty Analysis Dashboard
        fig_uncertainty = create_uncertainty_analysis_dashboard(
            ensemble_shap, model_std, model_range, feature_names
        )
        output_path = FIG_DIR / "shap_uncertainty_analysis.html"
        fig_uncertainty.write_html(str(output_path))
        print(f"✅ Generated uncertainty analysis: {output_path}")
        
        # 3. Generate muscle-specific plots
        muscle_groups = {
            'Upper Extremity': list(range(7)),  # First 7 targets
            'Lower Extremity': list(range(7, 13))  # Remaining targets
        }
        
        for group_name, target_indices in muscle_groups.items():
            # Average SHAP values across muscle group
            group_shap = ensemble_shap[:, :, target_indices].mean(axis=2, keepdims=True)
            group_uncertainty = model_std[:, :, target_indices].mean(axis=2, keepdims=True)
            
            fig_group = create_enhanced_shap_beeswarm_with_uncertainty(
                group_shap, group_uncertainty, X_sample, feature_names, target_idx=0
            )
            fig_group.update_layout(
                title=dict(text=f"SHAP Beeswarm - {group_name} Muscles<br><sub>Point size indicates model uncertainty</sub>")
            )
            
            filename = f"shap_beeswarm_{group_name.lower().replace(' ', '_')}_real.html"
            output_path = FIG_DIR / filename
            fig_group.write_html(str(output_path))
            print(f"✅ Generated SHAP for {group_name}: {output_path}")
        
        print("\n✨ All SHAP visualizations generated successfully!")
        print(f"📁 Output directory: {FIG_DIR}")
        
        # Save uncertainty metrics for later use
        mean_uncertainty = model_std.mean(axis=(0, 2))
        uncertainty_metrics = {
            'mean_uncertainty_by_feature': {
                feature_names[i]: float(mean_uncertainty[i]) 
                for i in np.argsort(mean_uncertainty)[-20:][::-1]
            },
            'weights': weights
        }
        
        import json
        with open(FIG_DIR / "uncertainty_metrics.json", 'w') as f:
            json.dump(uncertainty_metrics, f, indent=2)
        print("📊 Saved uncertainty metrics")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
