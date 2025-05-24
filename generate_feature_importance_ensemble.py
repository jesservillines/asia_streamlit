"""Generate feature importance visualizations using multiple methods for XGBoost + CatBoost ensemble."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
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
    
    # Filter dataframes to common patients
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
    
    # Load all models including HistGB
    models = {}
    model_paths = {
        'catboost': MODELS_DIR / "catboost_exact_model.pkl",
        'xgb': MODELS_DIR / "xgb_exact_model.pkl",
        'hgb': MODELS_DIR / "hgb_exact_model.pkl"
    }
    
    for name, path in model_paths.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"Loaded {name} model")
        else:
            print(f"Warning: Model not found at {path}")
    
    return X_train, y_true, X_cols, models, pids, TARGET_COLS

def calculate_model_weights(models, X, y_true):
    """Calculate weights based on model performance - only for XGB and CatBoost ensemble."""
    print("\nCalculating model weights based on RMSE...")
    
    rmse_scores = {}
    predictions = {}
    
    # Only calculate weights for XGB and CatBoost (for ensemble)
    ensemble_models = ['catboost', 'xgb']
    
    for name in ensemble_models:
        if name in models:
            model = models[name]
            pred = model.predict(X)
            predictions[name] = pred
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_true.flatten(), pred.flatten()))
            rmse_scores[name] = rmse
            print(f"{name}: RMSE = {rmse:.4f}")
    
    # Calculate weights based on inverse RMSE
    inverse_rmses = {name: 1/rmse for name, rmse in rmse_scores.items()}
    total_inverse = sum(inverse_rmses.values())
    weights = {name: inv_rmse/total_inverse for name, inv_rmse in inverse_rmses.items()}
    
    print("\nModel weights:")
    for name, weight in weights.items():
        print(f"{name}: {weight:.3f}")
    
    return weights, predictions

def get_native_feature_importance(models, feature_names):
    """Extract native feature importance from models where available."""
    print("\nExtracting native feature importance...")
    
    importance_dict = {}
    
    for name, model in models.items():
        if name == 'catboost':
            # CatBoost has direct feature importance
            try:
                importance = model.feature_importances_
                importance_dict[name] = importance
                print(f"  Got CatBoost feature importance")
            except:
                print(f"  Could not get CatBoost feature importance")
                
        elif name == 'xgb':
            # For MultiOutputRegressor, average importance across all estimators
            try:
                importances = []
                for estimator in model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                
                if importances:
                    importance = np.mean(importances, axis=0)
                    importance_dict[name] = importance
                    print(f"  Got XGBoost feature importance (averaged across {len(importances)} estimators)")
            except:
                print(f"  Could not get XGBoost feature importance")
                
        elif name == 'hgb':
            # For HistGradientBoostingRegressor, get feature importances
            try:
                importance = model.feature_importances_
                importance_dict[name] = importance
                print(f"  Got HistGB feature importance")
            except:
                print(f"  Could not get HistGB feature importance")
    
    return importance_dict

def calculate_permutation_importance(models, X, y, feature_names, n_repeats=10, sample_size=500):
    """Calculate permutation importance for each model."""
    print(f"\nCalculating permutation importance (n_repeats={n_repeats}, sample_size={sample_size})...")
    
    # Use a subset for faster computation
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    perm_importance_dict = {}
    
    for name, model in models.items():
        print(f"  Computing permutation importance for {name}...")
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X_sample, y_sample,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        perm_importance_dict[name] = {
            'importances_mean': result.importances_mean,
            'importances_std': result.importances_std
        }
        
        print(f"    Completed {name}")
    
    return perm_importance_dict

def create_feature_importance_comparison(native_importance, perm_importance, weights, feature_names):
    """Create comprehensive feature importance visualization."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Native Feature Importance (if available)",
            "Permutation Importance",
            "Weighted Ensemble Importance",
            "Model Agreement Analysis"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Get top features to display
    n_features = 20
    
    # 1. Native Feature Importance
    if native_importance:
        # Calculate weighted ensemble
        ensemble_native = None
        for name, importance in native_importance.items():
            if ensemble_native is None:
                ensemble_native = weights.get(name, 0) * importance
            else:
                ensemble_native += weights.get(name, 0) * importance
        
        top_idx = np.argsort(ensemble_native)[-n_features:]
        
        # Add traces for each model
        for i, (name, importance) in enumerate(native_importance.items()):
            fig.add_trace(
                go.Bar(
                    y=[feature_names[j] for j in top_idx],
                    x=importance[top_idx],
                    name=f"{name} (w={weights.get(name, 0):.2f})",
                    orientation='h',
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # 2. Permutation Importance
    if perm_importance:
        # Calculate weighted ensemble
        ensemble_perm = None
        ensemble_perm_std = None
        
        for name, perm_data in perm_importance.items():
            if ensemble_perm is None:
                ensemble_perm = weights.get(name, 0) * perm_data['importances_mean']
                ensemble_perm_std = weights.get(name, 0) * perm_data['importances_std']
            else:
                ensemble_perm += weights.get(name, 0) * perm_data['importances_mean']
                ensemble_perm_std += weights.get(name, 0) * perm_data['importances_std']
        
        top_idx = np.argsort(ensemble_perm)[-n_features:]
        
        # Add ensemble permutation importance
        fig.add_trace(
            go.Bar(
                y=[feature_names[j] for j in top_idx],
                x=ensemble_perm[top_idx],
                error_x=dict(
                    type='data',
                    array=ensemble_perm_std[top_idx],
                    visible=True
                ),
                name='Ensemble',
                orientation='h',
                marker_color='darkblue'
            ),
            row=1, col=2
        )
    
    # 3. Weighted Ensemble Importance (combining both methods if available)
    if native_importance and perm_importance:
        # Normalize and combine
        ensemble_native_norm = ensemble_native / (ensemble_native.max() + 1e-8)
        ensemble_perm_norm = ensemble_perm / (ensemble_perm.max() + 1e-8)
        
        # Combined importance (average of normalized values)
        combined_importance = (ensemble_native_norm + ensemble_perm_norm) / 2
        top_idx = np.argsort(combined_importance)[-n_features:]
        
        fig.add_trace(
            go.Bar(
                y=[feature_names[j] for j in top_idx],
                x=combined_importance[top_idx],
                orientation='h',
                marker_color='green',
                name='Combined'
            ),
            row=2, col=1
        )
    elif perm_importance:
        # If only permutation importance available
        top_idx = np.argsort(ensemble_perm)[-n_features:]
        
        fig.add_trace(
            go.Bar(
                y=[feature_names[j] for j in top_idx],
                x=ensemble_perm[top_idx],
                orientation='h',
                marker_color='green',
                name='Permutation-based'
            ),
            row=2, col=1
        )
    
    # 4. Model Agreement Analysis
    if perm_importance and len(perm_importance) > 1:
        # Calculate correlation between model importances
        model_names = list(perm_importance.keys())
        if len(model_names) >= 2:
            imp1 = perm_importance[model_names[0]]['importances_mean']
            imp2 = perm_importance[model_names[1]]['importances_mean']
            
            # Normalize for comparison
            imp1_norm = imp1 / (imp1.max() + 1e-8)
            imp2_norm = imp2 / (imp2.max() + 1e-8)
            
            fig.add_trace(
                go.Scatter(
                    x=imp1_norm,
                    y=imp2_norm,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.abs(imp1_norm - imp2_norm),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title="Disagreement",
                            x=1.15
                        )
                    ),
                    text=[feature_names[i] for i in range(len(feature_names))],
                    hovertemplate="<b>%{text}</b><br>" +
                                f"{model_names[0]}: %{{x:.3f}}<br>" +
                                f"{model_names[1]}: %{{y:.3f}}<br>" +
                                "Difference: %{marker.color:.3f}<extra></extra>"
                ),
                row=2, col=2
            )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Feature Importance Analysis - XGBoost + CatBoost Ensemble",
            font=dict(size=22)
        ),
        height=900,
        width=1400,
        showlegend=True,
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Importance Score", row=1, col=1)
    fig.update_xaxes(title_text="Permutation Importance", row=1, col=2)
    fig.update_xaxes(title_text="Combined Importance", row=2, col=1)
    fig.update_xaxes(title_text=f"{model_names[0] if len(model_names) > 0 else 'Model 1'} Importance", row=2, col=2)
    
    fig.update_yaxes(title_text="Features", row=1, col=1)
    fig.update_yaxes(title_text="Features", row=1, col=2)
    fig.update_yaxes(title_text="Features", row=2, col=1)
    fig.update_yaxes(title_text=f"{model_names[1] if len(model_names) > 1 else 'Model 2'} Importance", row=2, col=2)
    
    return fig

def create_simple_importance_plot(importance_data, feature_names, title="Feature Importance"):
    """Create a simple, clean feature importance plot."""
    
    # Get top features
    n_features = 25
    top_idx = np.argsort(importance_data)[-n_features:]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            y=[feature_names[i] for i in top_idx],
            x=importance_data[top_idx],
            orientation='h',
            marker=dict(
                color=importance_data[top_idx],
                colorscale='Blues',
                showscale=False
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Importance Score",
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title="Features",
            gridcolor='rgba(128,128,128,0.2)'
        ),
        height=700,
        width=1000,
        template="plotly_white",
        margin=dict(l=200)  # More space for feature names
    )
    
    return fig

def main():
    """Generate feature importance visualizations using multiple methods."""
    print("Starting feature importance analysis...")
    try:
        # Load data and models
        X, y_true, feature_names, models, pids, TARGET_COLS = load_data_and_models()
        
        if not models:
            print("No models found! Please ensure models are trained and saved.")
            return
        
        # Calculate model weights
        weights, predictions = calculate_model_weights(models, X, y_true)
        
        # Method 1: Native feature importance
        native_importance = get_native_feature_importance(models, feature_names)
        
        # Method 2: Permutation importance
        perm_importance = calculate_permutation_importance(
            models, X, y_true, feature_names, 
            n_repeats=5, sample_size=min(500, len(X))
        )
        
        # Generate visualizations
        print("\nGenerating feature importance visualizations...")
        
        # 1. Comprehensive comparison
        fig_comparison = create_feature_importance_comparison(
            native_importance, perm_importance, weights, feature_names
        )
        output_path = FIG_DIR / "feature_importance_comparison.html"
        fig_comparison.write_html(str(output_path))
        print(f"✅ Generated comprehensive comparison: {output_path}")
        
        # 2. Simple ensemble importance plot (permutation-based)
        if perm_importance:
            # Calculate weighted ensemble
            ensemble_perm = None
            for name, perm_data in perm_importance.items():
                if ensemble_perm is None:
                    ensemble_perm = weights.get(name, 0) * perm_data['importances_mean']
                else:
                    ensemble_perm += weights.get(name, 0) * perm_data['importances_mean']
            
            fig_simple = create_simple_importance_plot(
                ensemble_perm, feature_names,
                "XGBoost + CatBoost Ensemble - Feature Importance"
            )
            output_path = FIG_DIR / "feature_importance_ensemble_simple.html"
            fig_simple.write_html(str(output_path))
            print(f"✅ Generated simple importance plot: {output_path}")
        
        # 3. Individual model importance plots
        for name, perm_data in perm_importance.items():
            fig_model = create_simple_importance_plot(
                perm_data['importances_mean'], feature_names,
                f"{name.upper()} - Feature Importance"
            )
            output_path = FIG_DIR / f"feature_importance_{name}.html"
            fig_model.write_html(str(output_path))
            print(f"✅ Generated {name} importance plot: {output_path}")
        
        # 4. Save importance scores for later use
        importance_data = {
            'weights': weights,
            'feature_names': feature_names
        }
        
        if native_importance:
            importance_data['native_importance'] = {
                name: imp.tolist() for name, imp in native_importance.items()
            }
        
        if perm_importance:
            importance_data['permutation_importance'] = {
                name: {
                    'mean': data['importances_mean'].tolist(),
                    'std': data['importances_std'].tolist()
                }
                for name, data in perm_importance.items()
            }
        
        import json
        with open(FIG_DIR / "feature_importance_data.json", 'w') as f:
            json.dump(importance_data, f, indent=2)
        print("📊 Saved importance data")
        
        print("\n✨ All feature importance visualizations generated successfully!")
        print(f"📁 Output directory: {FIG_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
