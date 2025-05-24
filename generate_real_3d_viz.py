"""Generate 3D visualization using real model predictions."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data_and_models():
    """Load processed data and trained models."""
    base_path = Path("asia-impairment-track-prediction")
    
    # Load raw features to get week 1 motor scores
    train_features = pd.read_csv(base_path / "data" / "train_features.csv")
    
    # Load processed data for model input
    train_df = pd.read_csv(base_path / "data" / "train_processed.csv")
    
    # Load week 26 outcomes
    outcomes_df = pd.read_csv(base_path / "data" / "train_outcomes_MS.csv")
    week26_outcomes = outcomes_df[outcomes_df['time'] == 26].copy()
    
    # Load metadata
    metadata = pd.read_csv(base_path / "data" / "metadata.csv")
    
    # Define target columns
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
    common_pids = sorted(list(common_pids))  # Sort for consistency
    
    print(f"Found {len(common_pids)} patients with complete data")
    
    # Filter dataframes to common patients and ensure same order
    train_features_filtered = train_features[train_features['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    week26_outcomes_filtered = week26_outcomes[week26_outcomes['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    train_df_filtered = train_df[train_df['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    
    print(f"Filtered data shapes - features: {train_features_filtered.shape}, outcomes: {week26_outcomes_filtered.shape}, processed: {train_df_filtered.shape}")
    
    # Calculate week 1 total motor scores
    week1_scores = train_features_filtered[week1_motor_cols].sum(axis=1).values
    
    # Calculate week 26 actual total motor scores
    week26_actual_scores = week26_outcomes_filtered[TARGET_COLS].sum(axis=1).values
    
    # Get week 1 AIS grades
    week1_ais = train_features_filtered['ais1'].values if 'ais1' in train_features_filtered.columns else None
    
    # Prepare features for model prediction
    train_df_no_pid = train_df_filtered.drop(columns=['PID'])
    
    # Remove target columns from features
    X_cols = [c for c in train_df_no_pid.columns if c not in TARGET_COLS]
    X_train = train_df_no_pid[X_cols].values
    
    # Get PIDs for tracking
    pids = train_df_filtered['PID'].values
    
    # Load only CatBoost and XGBoost models for the weighted ensemble
    model_paths = {
        'CatBoost': base_path / "models_exact" / "catboost_exact_model.pkl",
        'XGBoost': base_path / "models_exact" / "xgb_exact_model.pkl"
    }
    
    models = []
    model_names = []
    
    for name, path in model_paths.items():
        if path.exists():
            model = joblib.load(path)
            models.append(model)
            model_names.append(name)
            print(f"Loaded {name} model")
        else:
            print(f"Warning: {name} model not found at {path}")
    
    if len(models) != 2:
        raise ValueError("Both CatBoost and XGBoost models are required for the weighted ensemble")
    
    return X_train, week1_scores, week26_actual_scores, metadata, models, model_names, pids, TARGET_COLS, week1_ais

def make_predictions_for_week26(models, X, target_cols):
    """Make predictions for week 26 using XGBoost + CatBoost weighted ensemble."""
    predictions = []
    
    for model in models:
        try:
            # Make predictions for all 20 individual muscles
            pred = model.predict(X)
            predictions.append(pred)
            print(f"Made predictions with model - shape: {pred.shape}")
        except Exception as e:
            print(f"Warning: Could not make predictions with model: {e}")
    
    if len(predictions) != 2:
        raise ValueError("Both CatBoost and XGBoost predictions are required")
    
    # Calculate weighted ensemble predictions based on inverse RMSE
    # These weights are from the ensemble comparison analysis
    # CatBoost typically has lower RMSE, so gets higher weight
    catboost_weight = 0.55  # Approximate weight based on performance
    xgboost_weight = 0.45   # Approximate weight based on performance
    
    # Normalize weights to sum to 1
    total_weight = catboost_weight + xgboost_weight
    weights = [catboost_weight / total_weight, xgboost_weight / total_weight]
    
    # Calculate weighted average
    ensemble_pred = weights[0] * predictions[0] + weights[1] * predictions[1]
    
    # Round and clip as done in end_to_end_reproduction.py
    ensemble_pred_rounded = np.round(ensemble_pred)
    ensemble_pred_clipped = np.clip(ensemble_pred_rounded, 0, 5).astype(int)
    
    # Calculate total motor scores for week 26 predictions
    week26_pred_scores = ensemble_pred_clipped.sum(axis=1)
    
    # Calculate confidence based on agreement between models
    pred_diff = np.abs(predictions[0] - predictions[1])
    # Average difference across all muscles for each patient
    avg_diff = np.mean(pred_diff, axis=1)
    # Convert to confidence (lower diff = higher confidence)
    confidence = 1 - (avg_diff / 2.5)  # Normalize by max expected diff
    confidence = np.clip(confidence, 0.5, 0.95)
    
    print(f"Using weighted ensemble: CatBoost={weights[0]:.2f}, XGBoost={weights[1]:.2f}")
    
    # Return both total scores and individual muscle predictions
    return week26_pred_scores, confidence, ensemble_pred_clipped

def calculate_ais_grade(motor_score):
    """Calculate AIS grade based on total motor score."""
    if motor_score < 20:
        return 'A'
    elif motor_score < 50:
        return 'B'
    elif motor_score < 80:
        return 'C'
    else:
        return 'D'

def create_3d_visualization_from_models(
    age_filter=None,
    bmi_filter=None,
    sex_filter=None,
    ais_filter=None,
    week1_ais_filter=None,
    point_size=10,
    opacity=0.7,
    show_grid=True,
    show_legend=True,
    show_regression_plane=False,
    color_by='ais_grade',
    marker_style='circle'
):
    """Create 3D visualization using actual model predictions."""
    
    # Load data and models
    X_train, week1_scores, week26_actual_scores, metadata, models, model_names, pids, target_cols, week1_ais = load_data_and_models()
    
    # Make predictions
    week26_pred_scores, confidence, individual_predictions = make_predictions_for_week26(models, X_train, target_cols)
    
    # Get actual individual muscle scores for RMSE calculation
    # Load week 26 outcomes to get individual muscle scores
    base_path = Path("asia-impairment-track-prediction")
    outcomes_df = pd.read_csv(base_path / "data" / "train_outcomes_MS.csv")
    week26_outcomes = outcomes_df[outcomes_df['time'] == 26].copy()
    
    # Filter to common PIDs and ensure same order
    common_pids = pids
    week26_outcomes_filtered = week26_outcomes[week26_outcomes['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    actual_individual_scores = week26_outcomes_filtered[target_cols].values
    
    # Create visualization dataframe
    viz_df = pd.DataFrame({
        'PID': pids,
        'week1_score': week1_scores,
        'week26_actual_score': week26_actual_scores,
        'week26_pred_score': week26_pred_scores,
        'actual_gain': week26_actual_scores - week1_scores,
        'predicted_gain': week26_pred_scores - week1_scores,
        'prediction_error': week26_pred_scores - week26_actual_scores,
        'week26_ais_actual': [calculate_ais_grade(score) for score in week26_actual_scores],
        'week26_ais_pred': [calculate_ais_grade(score) for score in week26_pred_scores],
        'week1_ais': week1_ais if week1_ais is not None else ['Unknown'] * len(pids),
        'confidence': confidence
    })
    
    # Add individual muscle predictions and actuals to dataframe for filtering
    for i, col in enumerate(target_cols):
        viz_df[f'{col}_actual'] = actual_individual_scores[:, i]
        viz_df[f'{col}_pred'] = individual_predictions[:, i]
    
    # Merge with metadata for demographics
    viz_df = viz_df.merge(metadata[['PID', 'age_category', 'bmi_category', 'sexcd']], 
                          on='PID', how='left')
    
    # Handle missing values and map sex codes
    viz_df['age_category'] = viz_df['age_category'].fillna('Unknown')
    viz_df['bmi_category'] = viz_df['bmi_category'].fillna('Unknown')
    viz_df['sex'] = viz_df['sexcd'].map({1: 'Male', 2: 'Female'}).fillna('Unknown')
    
    # Apply filters
    mask = pd.Series([True] * len(viz_df))
    
    if age_filter:
        mask &= viz_df['age_category'].isin(age_filter)
    
    if bmi_filter:
        mask &= viz_df['bmi_category'].isin(bmi_filter)
    
    if sex_filter:
        mask &= viz_df['sex'].isin(sex_filter)
    
    if ais_filter:
        mask &= viz_df['week26_ais_actual'].isin(ais_filter)
    
    if week1_ais_filter:
        mask &= viz_df['week1_ais'].isin([f'AIS {grade}' for grade in week1_ais_filter])
    
    viz_df_filtered = viz_df[mask].copy()
    
    # Calculate individual muscle RMSE for filtered data
    if len(viz_df_filtered) > 0:
        filtered_actual = viz_df_filtered[[f'{col}_actual' for col in target_cols]].values
        filtered_pred = viz_df_filtered[[f'{col}_pred' for col in target_cols]].values
        filtered_individual_rmse = np.sqrt(np.mean((filtered_pred - filtered_actual)**2))
    else:
        filtered_individual_rmse = 0.0
    
    # Add AIS grade for display
    viz_df_filtered['week1_ais_grade'] = viz_df_filtered['week1_ais'].apply(lambda x: x.split()[-1] if 'AIS' in str(x) else 'Unknown')
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Define color schemes
    if color_by == 'ais_grade':
        color_map = {'A': '#FF4136', 'B': '#FF851B', 'C': '#2ECC40', 'D': '#0074D9'}
        viz_df_filtered['color'] = viz_df_filtered['week26_ais_actual'].map(color_map)
        color_label = 'Week 26 AIS Grade'
    elif color_by == 'confidence':
        viz_df_filtered['color'] = viz_df_filtered['confidence']
        color_label = 'Model Confidence'
    elif color_by == 'error':
        viz_df_filtered['color'] = viz_df_filtered['prediction_error']
        color_label = 'Prediction Error'
    else:  # gain
        viz_df_filtered['color'] = viz_df_filtered['actual_gain']
        color_label = 'Actual Gain'
    
    # Define marker symbols
    symbol_map = {
        'circle': 'circle',
        'square': 'square',
        'diamond': 'diamond',
        'cross': 'cross'
    }
    
    # Add scatter points
    if color_by == 'ais_grade':
        # Group by AIS grade for legend
        for grade in ['A', 'B', 'C', 'D']:
            grade_data = viz_df_filtered[viz_df_filtered['week26_ais_actual'] == grade]
            if len(grade_data) > 0:
                fig.add_trace(go.Scatter3d(
                    x=grade_data['week26_actual_score'],
                    y=grade_data['week26_pred_score'],
                    z=grade_data['actual_gain'],
                    mode='markers',
                    name=f'AIS {grade} (n={len(grade_data)})',
                    marker=dict(
                        size=point_size,
                        color=color_map[grade],
                        opacity=opacity,
                        symbol=symbol_map[marker_style],
                        line=dict(color='white', width=1)
                    ),
                    text=[f"Patient: {pid}<br>Age: {age}<br>BMI: {bmi}<br>Sex: {sex}<br>Week 1: {week1:.0f} (AIS {w1_ais})<br>Week 26 Actual: {week26_actual:.0f}<br>Week 26 Predicted: {week26_pred:.0f}<br>Actual Gain: {act_gain:.0f}<br>Predicted Gain: {pred_gain:.0f}<br>Prediction Error: {error:+.0f}<br>Confidence: {conf:.2f}" 
                          for pid, age, bmi, sex, week1, w1_ais, week26_actual, week26_pred, act_gain, pred_gain, error, conf in zip(
                              grade_data['PID'],
                              grade_data['age_category'],
                              grade_data['bmi_category'],
                              grade_data['sex'],
                              grade_data['week1_score'],
                              grade_data['week1_ais_grade'],
                              grade_data['week26_actual_score'],
                              grade_data['week26_pred_score'],
                              grade_data['actual_gain'],
                              grade_data['predicted_gain'],
                              grade_data['prediction_error'],
                              grade_data['confidence']
                          )],
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=show_legend
                ))
    else:
        # Single trace with continuous color scale
        fig.add_trace(go.Scatter3d(
            x=viz_df_filtered['week26_actual_score'],
            y=viz_df_filtered['week26_pred_score'],
            z=viz_df_filtered['actual_gain'],
            mode='markers',
            name='Patients',
            marker=dict(
                size=point_size,
                color=viz_df_filtered['color'],
                colorscale='Viridis' if color_by != 'error' else 'RdBu',
                opacity=opacity,
                symbol=symbol_map[marker_style],
                line=dict(color='white', width=1),
                colorbar=dict(
                    title=color_label,
                    thickness=15,
                    len=0.7
                ),
                cmid=0 if color_by == 'error' else None
            ),
            text=[f"Patient: {pid}<br>Age: {age}<br>BMI: {bmi}<br>Sex: {sex}<br>Week 1: {week1:.0f} (AIS {w1_ais})<br>Week 26 Actual: {week26_actual:.0f}<br>Week 26 Predicted: {week26_pred:.0f}<br>Actual Gain: {act_gain:.0f}<br>Predicted Gain: {pred_gain:.0f}<br>Prediction Error: {error:+.0f}<br>Confidence: {conf:.2f}" 
                  for pid, age, bmi, sex, week1, w1_ais, week26_actual, week26_pred, act_gain, pred_gain, error, conf in zip(
                      viz_df_filtered['PID'],
                      viz_df_filtered['age_category'],
                      viz_df_filtered['bmi_category'],
                      viz_df_filtered['sex'],
                      viz_df_filtered['week1_score'],
                      viz_df_filtered['week1_ais_grade'],
                      viz_df_filtered['week26_actual_score'],
                      viz_df_filtered['week26_pred_score'],
                      viz_df_filtered['actual_gain'],
                      viz_df_filtered['predicted_gain'],
                      viz_df_filtered['prediction_error'],
                      viz_df_filtered['confidence']
                  )],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
    
    # Add perfect prediction line (where actual = predicted)
    if len(viz_df_filtered) > 0:
        min_score = min(viz_df_filtered['week26_actual_score'].min(), viz_df_filtered['week26_pred_score'].min())
        max_score = max(viz_df_filtered['week26_actual_score'].max(), viz_df_filtered['week26_pred_score'].max())
        
        # Create multiple points for the perfect prediction line
        line_points = np.linspace(min_score, max_score, 20)
        
        # For each point on the perfect prediction line, calculate the expected gain
        # This creates a 3D line where x=y and z varies based on the data
        z_values = []
        for score in line_points:
            # Find patients with similar actual scores and get their average gain
            similar_mask = np.abs(viz_df_filtered['week26_actual_score'] - score) < 5
            if similar_mask.any():
                z_values.append(viz_df_filtered.loc[similar_mask, 'actual_gain'].mean())
            else:
                z_values.append(viz_df_filtered['actual_gain'].mean())
        
        fig.add_trace(go.Scatter3d(
            x=line_points,
            y=line_points,
            z=z_values,
            mode='lines',
            name='Perfect Prediction',
            line=dict(
                color='red',
                width=4,
                dash='dash'
            ),
            showlegend=True
        ))
        
        # Add regression plane if requested
        if show_regression_plane:
            # Create a mesh grid for the regression plane
            x_range = np.linspace(viz_df_filtered['week26_actual_score'].min(), viz_df_filtered['week26_actual_score'].max(), 10)
            y_range = np.linspace(viz_df_filtered['week26_pred_score'].min(), viz_df_filtered['week26_pred_score'].max(), 10)
            xx, yy = np.meshgrid(x_range, y_range)
            
            # Fit a plane to the data
            from sklearn.linear_model import LinearRegression
            X_plane = viz_df_filtered[['week26_actual_score', 'week26_pred_score']].values
            y_plane = viz_df_filtered['actual_gain'].values
            
            reg = LinearRegression().fit(X_plane, y_plane)
            zz = reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            fig.add_trace(go.Surface(
                x=xx,
                y=yy,
                z=zz,
                opacity=0.3,
                colorscale='Blues',
                showscale=False,
                name='Regression Plane'
            ))
    
    # Update layout with light theme
    fig.update_layout(
        title=dict(
            text=f'ASIA Motor Score Predictions: XGBoost + CatBoost Weighted Ensemble<br><sub>Showing {len(viz_df_filtered)} patients | Individual Muscle RMSE: {filtered_individual_rmse:.2f}</sub>',
            font=dict(size=20, color='#333333'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title='Week 26 Actual Total Motor Score',
                gridcolor='#E0E0E0',
                showbackground=True,
                backgroundcolor='#FAFAFA',
                zerolinecolor='#666666',
                titlefont=dict(color='#333333'),
                tickfont=dict(color='#333333'),
                showgrid=show_grid
            ),
            yaxis=dict(
                title='Week 26 Predicted Total Motor Score',
                gridcolor='#E0E0E0',
                showbackground=True,
                backgroundcolor='#FAFAFA',
                zerolinecolor='#666666',
                titlefont=dict(color='#333333'),
                tickfont=dict(color='#333333'),
                showgrid=show_grid
            ),
            zaxis=dict(
                title='Actual Gain (Week 1 to 26)',
                gridcolor='#E0E0E0',
                showbackground=True,
                backgroundcolor='#FAFAFA',
                zerolinecolor='#666666',
                titlefont=dict(color='#333333'),
                tickfont=dict(color='#333333'),
                showgrid=show_grid
            ),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=show_legend,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#CCCCCC',
            borderwidth=1,
            font=dict(color='#333333')
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#333333'),
        margin=dict(l=0, r=0, b=0, t=80),
        height=700
    )
    
    # Add interpretation guide
    fig.add_annotation(
        text="<b>Interpretation:</b><br>Points closer to the red dashed line have more accurate predictions",
        xref="paper", yref="paper",
        x=0.5, y=-0.08,
        showarrow=False,
        font=dict(size=10, color="#666666"),
        align="center"
    )
    
    # Calculate and display metrics
    if len(viz_df_filtered) > 0:
        # Total score metrics
        total_mae = np.mean(np.abs(viz_df_filtered['prediction_error']))
        total_rmse = np.sqrt(np.mean(viz_df_filtered['prediction_error']**2))
        r2 = np.corrcoef(viz_df_filtered['week26_actual_score'], viz_df_filtered['week26_pred_score'])[0, 1]**2
        
        # Add metrics box with both individual and total metrics - LARGER AND MORE VISIBLE
        fig.add_annotation(
            text=f"<b>Model Performance:</b><br>" +
                 f"<span style='font-size:14px'>• Individual Muscle RMSE: <b>{filtered_individual_rmse:.2f}</b></span><br>" +
                 f"<span style='font-size:14px'>• Total Score MAE: <b>{total_mae:.2f}</b></span><br>" +
                 f"<span style='font-size:14px'>• Total Score RMSE: <b>{total_rmse:.2f}</b></span><br>" +
                 f"<span style='font-size:14px'>• R²: <b>{r2:.3f}</b></span><br>" +
                 f"<span style='font-size:12px'>• Avg Actual Gain: {viz_df_filtered['actual_gain'].mean():.1f}</span><br>" +
                 f"<span style='font-size:12px'>• Avg Predicted Gain: {viz_df_filtered['predicted_gain'].mean():.1f}</span>",
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=12, color="#333333", family="Arial"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#333333",
            borderwidth=2,
            borderpad=10,
            align="left"
        )
    
    return fig

if __name__ == "__main__":
    # Test the visualization
    try:
        fig = create_3d_visualization_from_models()
        
        # Save to file
        output_dir = Path("asia-impairment-track-prediction/visuals/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig.write_html(str(output_dir / "real_3d_predictions.html"))
        print("Generated 3D visualization with actual model predictions")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        import traceback
        traceback.print_exc()
