import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

def load_data_and_models():
    """Load processed data and trained models"""
    base_path = Path("asia-impairment-track-prediction")
    
    # Load raw training features
    train_features = pd.read_csv(base_path / "data" / "train_features.csv")
    
    # Load processed training data
    train_df = pd.read_csv(base_path / "data" / "train_processed.csv")
    
    # Load outcomes
    outcomes_df = pd.read_csv(base_path / "data" / "train_outcomes_MS.csv")
    week26_outcomes = outcomes_df[outcomes_df['time'] == 26].copy()
    
    # Load metadata
    metadata = pd.read_csv(base_path / "data" / "metadata.csv")
    
    # Define target columns (lowercase, no week suffix)
    TARGET_COLS = [
        'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
        'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
        'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
        'gretor', 'ankplr'
    ]
    
    # Get week 1 motor scores
    week1_motor_cols = [f"{col}01" for col in TARGET_COLS]
    
    # Filter to patients that have both week 1 and week 26 data
    common_pids = set(train_features['PID']).intersection(set(week26_outcomes['PID']))
    common_pids = common_pids.intersection(set(train_df['PID']))
    common_pids = sorted(list(common_pids))
    
    print(f"Found {len(common_pids)} patients with complete data")
    
    # Filter dataframes to common patients and ensure same order
    train_features_filtered = train_features[train_features['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    week26_outcomes_filtered = week26_outcomes[week26_outcomes['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    train_df_filtered = train_df[train_df['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    metadata_filtered = metadata[metadata['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    
    # Calculate week 1 total motor scores
    week1_scores = train_features_filtered[week1_motor_cols].sum(axis=1).values
    
    # Calculate week 26 actual total motor scores
    week26_actual_scores = week26_outcomes_filtered[TARGET_COLS].sum(axis=1).values
    
    # Get week 1 AIS grades
    week1_ais = []
    for score in week1_scores:
        if score < 20:
            ais = 'A'
        elif score < 50:
            ais = 'B'
        elif score < 80:
            ais = 'C'
        else:
            ais = 'D'
        week1_ais.append(ais)
    
    # Prepare features for model prediction
    train_df_no_pid = train_df_filtered.drop(columns=['PID'])
    
    # Remove target columns from features
    X_cols = [c for c in train_df_no_pid.columns if c not in TARGET_COLS]
    X_train = train_df_no_pid[X_cols].values
    
    # Get PIDs for tracking
    pids = train_df_filtered['PID'].values
    
    # Load models
    model_paths = {
        'catboost': base_path / "models_exact" / "catboost_exact_model.pkl",
        'xgb': base_path / "models_exact" / "xgb_exact_model.pkl",
        'hgb': base_path / "models_exact" / "hgb_exact_model.pkl"
    }
    
    models = {}
    model_names = []
    
    for name, path in model_paths.items():
        if path.exists():
            models[name] = joblib.load(path)
            model_names.append(name)
            print(f"Loaded {name} model")
        else:
            print(f"Warning: Model not found at {path}")
    
    return X_train, week1_scores, week26_outcomes_filtered, metadata_filtered, models, model_names, pids, TARGET_COLS, week1_ais

def calculate_rmse_for_models(X_train, week26_actual_scores, models, TARGET_COLS):
    """Calculate RMSE for all models and different ensemble approaches"""
    
    # Convert X_train to numpy array for model predictions
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    
    # Get predictions from all models
    all_predictions = {}
    model_names = ['catboost', 'xgb', 'hgb']
    
    for i, (model_name, model) in enumerate(zip(model_names, models.values())):
        try:
            predictions = model.predict(X_train_array)
            all_predictions[model_name] = predictions
            print(f"Made predictions with {model_name} - shape: {predictions.shape}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    # Prepare actual values
    week26_outcomes_filtered = week26_actual_scores[TARGET_COLS]
    
    # Calculate individual model RMSE
    muscle_rmse = {}
    model_avg_rmse = {}
    
    for model_name, predictions in all_predictions.items():
        muscle_rmse[model_name] = {}
        rmse_values = []
        
        # Calculate RMSE for each muscle
        for i, muscle in enumerate(TARGET_COLS):
            y_true = week26_outcomes_filtered[muscle].values
            y_pred = predictions[:, i]
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            muscle_rmse[model_name][muscle] = rmse
            rmse_values.append(rmse)
        
        # Store average RMSE for this model
        model_avg_rmse[model_name] = np.mean(rmse_values)
        print(f"{model_name} average RMSE: {model_avg_rmse[model_name]:.3f}")
    
    # 1. Simple Average Ensemble
    simple_ensemble_predictions = np.zeros((len(X_train), len(TARGET_COLS)))
    for predictions in all_predictions.values():
        simple_ensemble_predictions += predictions
    simple_ensemble_predictions /= len(all_predictions)
    
    # Calculate simple ensemble RMSE
    muscle_rmse['Simple Ensemble'] = {}
    simple_ensemble_rmse_values = []
    for i, muscle in enumerate(TARGET_COLS):
        y_true = week26_outcomes_filtered[muscle].values
        y_pred = simple_ensemble_predictions[:, i]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        muscle_rmse['Simple Ensemble'][muscle] = rmse
        simple_ensemble_rmse_values.append(rmse)
    
    print(f"\nSimple Ensemble average RMSE: {np.mean(simple_ensemble_rmse_values):.3f}")
    
    # 2. Weighted Average Ensemble (all models)
    # Calculate weights (inverse of RMSE, normalized)
    inverse_rmse = {name: 1.0 / rmse for name, rmse in model_avg_rmse.items()}
    total_inverse = sum(inverse_rmse.values())
    ensemble_weights = {name: inv / total_inverse for name, inv in inverse_rmse.items()}
    
    print("\nWeighted Ensemble weights (all models):")
    for name, weight in ensemble_weights.items():
        print(f"  {name}: {weight:.3f}")
    
    # Calculate weighted ensemble predictions
    weighted_ensemble_predictions = np.zeros((len(X_train), len(TARGET_COLS)))
    for model_name, predictions in all_predictions.items():
        weighted_ensemble_predictions += predictions * ensemble_weights[model_name]
    
    # Calculate weighted ensemble RMSE
    muscle_rmse['Weighted Ensemble'] = {}
    weighted_ensemble_rmse_values = []
    for i, muscle in enumerate(TARGET_COLS):
        y_true = week26_outcomes_filtered[muscle].values
        y_pred = weighted_ensemble_predictions[:, i]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        muscle_rmse['Weighted Ensemble'][muscle] = rmse
        weighted_ensemble_rmse_values.append(rmse)
    
    print(f"\nWeighted Ensemble average RMSE: {np.mean(weighted_ensemble_rmse_values):.3f}")
    
    # 3. Weighted Average Ensemble (CatBoost + XGBoost only)
    cb_xgb_models = {name: rmse for name, rmse in model_avg_rmse.items() if name in ['catboost', 'xgb']}
    cb_xgb_inverse = {name: 1.0 / rmse for name, rmse in cb_xgb_models.items()}
    cb_xgb_total_inverse = sum(cb_xgb_inverse.values())
    cb_xgb_weights = {name: inv / cb_xgb_total_inverse for name, inv in cb_xgb_inverse.items()}
    
    print("\nCatBoost + XGBoost Ensemble weights:")
    for name, weight in cb_xgb_weights.items():
        print(f"  {name}: {weight:.3f}")
    
    # Calculate CatBoost + XGBoost ensemble predictions
    cb_xgb_ensemble_predictions = np.zeros((len(X_train), len(TARGET_COLS)))
    for model_name in ['catboost', 'xgb']:
        cb_xgb_ensemble_predictions += all_predictions[model_name] * cb_xgb_weights[model_name]
    
    # Calculate CatBoost + XGBoost ensemble RMSE
    muscle_rmse['CB+XGB Ensemble'] = {}
    cb_xgb_ensemble_rmse_values = []
    for i, muscle in enumerate(TARGET_COLS):
        y_true = week26_outcomes_filtered[muscle].values
        y_pred = cb_xgb_ensemble_predictions[:, i]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        muscle_rmse['CB+XGB Ensemble'][muscle] = rmse
        cb_xgb_ensemble_rmse_values.append(rmse)
    
    print(f"\nCatBoost + XGBoost Ensemble average RMSE: {np.mean(cb_xgb_ensemble_rmse_values):.3f}")
    
    return muscle_rmse

def create_radar_chart(muscle_rmse, title, filename):
    """Create a radar chart for the given RMSE data"""
    
    # Group muscles for visualization
    muscle_groups = {
        'elbfll': 'Elbow Flex',
        'wrextl': 'Wrist Ext',
        'elbexl': 'Elbow Ext',
        'finfll': 'Finger Flex',
        'finabl': 'Finger Abd',
        'hipfll': 'Hip Flex',
        'kneexl': 'Knee Ext',
        'ankdol': 'Ankle Dorsi',
        'gretol': 'Great Toe Ext',
        'ankpll': 'Ankle Plant'
    }
    
    # Calculate group-level RMSE (average of left and right)
    group_rmse = {}
    for model_name in muscle_rmse.keys():
        group_rmse[model_name] = {}
        for base_muscle, display_name in muscle_groups.items():
            left_muscle = base_muscle
            right_muscle = base_muscle.replace('l', 'r')
            
            left_rmse = muscle_rmse[model_name].get(left_muscle, 0)
            right_rmse = muscle_rmse[model_name].get(right_muscle, 0)
            avg_rmse = (left_rmse + right_rmse) / 2
            
            group_rmse[model_name][display_name] = avg_rmse
    
    # Create figure
    fig = go.Figure()
    
    # Define colors for models
    colors = {
        'catboost': 'rgba(34, 139, 34, 0.8)',  # Forest Green
        'xgb': 'rgba(138, 43, 226, 0.8)',      # Blue Violet
        'hgb': 'rgba(255, 20, 147, 0.8)',      # Deep Pink
        'Simple Ensemble': 'rgba(0, 0, 255, 0.9)',     # Blue
        'Weighted Ensemble': 'rgba(255, 140, 0, 0.9)', # Dark Orange
        'CB+XGB Ensemble': 'rgba(220, 20, 60, 0.9)'    # Crimson
    }
    
    # Add traces for each model
    categories = list(muscle_groups.values())
    
    for model_name in muscle_rmse.keys():
        values = [group_rmse[model_name][cat] for cat in categories]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model_name,
            line=dict(color=colors.get(model_name, 'rgba(128, 128, 128, 0.8)'), width=2),
            marker=dict(size=8)
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.2],
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                gridcolor='rgba(128, 128, 128, 0.3)',
                gridwidth=1,
                showline=True,
                linewidth=2,
                linecolor='rgba(128, 128, 128, 0.5)'
            ),
            angularaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.3)',
                gridwidth=1,
                showline=True,
                linewidth=2,
                linecolor='rgba(128, 128, 128, 0.5)'
            ),
            bgcolor='rgba(240, 240, 240, 0.3)'
        ),
        showlegend=True,
        legend=dict(
            x=1.1,
            y=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(128, 128, 128, 0.5)',
            borderwidth=1
        ),
        title=dict(
            text=title,
            font=dict(size=20, family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='white',
        plot_bgcolor='rgba(250, 250, 250, 0.8)',
        width=900,
        height=700
    )
    
    # Save the figure
    fig.write_html(filename)
    print(f"\nSaved: {filename}")

def main():
    """Generate three radar charts comparing different ensemble methods"""
    print("Loading data and models...")
    X_train, week1_scores, week26_actual_scores, metadata, models, model_names, pids, TARGET_COLS, week1_ais = load_data_and_models()
    
    print(f"\nLoaded data for {len(X_train)} patients")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of target muscles: {len(TARGET_COLS)}")
    
    print("\nCalculating RMSE for all models and ensembles...")
    muscle_rmse = calculate_rmse_for_models(X_train, week26_actual_scores, models, TARGET_COLS)
    
    # Create three different charts
    print("\nGenerating radar charts...")
    
    # 1. Simple Average Ensemble
    simple_models = ['catboost', 'xgb', 'hgb', 'Simple Ensemble']
    simple_rmse = {k: v for k, v in muscle_rmse.items() if k in simple_models}
    create_radar_chart(
        simple_rmse,
        "Target-wise RMSE - Simple Average Ensemble",
        "rmse_radar_simple_ensemble.html"
    )
    
    # 2. Weighted Average Ensemble (all models)
    weighted_models = ['catboost', 'xgb', 'hgb', 'Weighted Ensemble']
    weighted_rmse = {k: v for k, v in muscle_rmse.items() if k in weighted_models}
    create_radar_chart(
        weighted_rmse,
        "Target-wise RMSE - Weighted Average Ensemble",
        "rmse_radar_weighted_ensemble.html"
    )
    
    # 3. CatBoost + XGBoost Weighted Ensemble
    cb_xgb_models = ['catboost', 'xgb', 'CB+XGB Ensemble']
    cb_xgb_rmse = {k: v for k, v in muscle_rmse.items() if k in cb_xgb_models}
    create_radar_chart(
        cb_xgb_rmse,
        "Target-wise RMSE - CatBoost + XGBoost Weighted Ensemble",
        "rmse_radar_cb_xgb_ensemble.html"
    )
    
    print("\nAll charts generated successfully!")

if __name__ == "__main__":
    main()
