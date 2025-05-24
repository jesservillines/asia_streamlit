"""Generate interactive 3D RMSE radar plot for individual muscle predictions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_data_and_calculate_rmse():
    """Load data and calculate RMSE for each muscle."""
    base_path = Path("asia-impairment-track-prediction")
    
    # Load data
    train_features = pd.read_csv(base_path / "data" / "train_features.csv")
    train_df = pd.read_csv(base_path / "data" / "train_processed.csv")
    outcomes_df = pd.read_csv(base_path / "data" / "train_outcomes_MS.csv")
    week26_outcomes = outcomes_df[outcomes_df['time'] == 26].copy()
    
    # Define target columns (20 muscles)
    TARGET_COLS = [
        'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
        'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
        'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
        'gretor', 'ankplr'
    ]
    
    # Define 10 muscle groups (combining left and right)
    MUSCLE_GROUPS = {
        'Elbow Flexors': ['elbfll', 'elbflr'],
        'Wrist Extensors': ['wrextl', 'wrextr'],
        'Elbow Extensors': ['elbexl', 'elbexr'],
        'Finger Flexors': ['finfll', 'finflr'],
        'Finger Abductors': ['finabl', 'finabr'],
        'Hip Flexors': ['hipfll', 'hipflr'],
        'Knee Extensors': ['kneexl', 'kneetr'],
        'Ankle Dorsiflexors': ['ankdol', 'ankdor'],
        'Great Toe Extensors': ['gretol', 'gretor'],
        'Ankle Plantarflexors': ['ankpll', 'ankplr']
    }
    
    # Get common PIDs
    common_pids = set(train_features['PID']).intersection(set(week26_outcomes['PID']))
    common_pids = common_pids.intersection(set(train_df['PID']))
    common_pids = sorted(list(common_pids))
    
    # Filter dataframes
    week26_outcomes_filtered = week26_outcomes[week26_outcomes['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    train_df_filtered = train_df[train_df['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    
    # Prepare features
    train_df_no_pid = train_df_filtered.drop(columns=['PID'])
    X_cols = [c for c in train_df_no_pid.columns if c not in TARGET_COLS]
    X_train = train_df_no_pid[X_cols].values
    
    # Load models
    model_paths = {
        'CatBoost': base_path / "models_exact" / "catboost_exact_model.pkl",
        'XGBoost': base_path / "models_exact" / "xgb_exact_model.pkl",
        'HistGB': base_path / "models_exact" / "hgb_exact_model.pkl"
    }
    
    models = []
    model_names = []
    all_predictions = {}
    
    for name, path in model_paths.items():
        if path.exists():
            model = joblib.load(path)
            models.append(model)
            model_names.append(name)
            # Store predictions for ensemble calculation
            predictions = model.predict(X_train)
            all_predictions[name] = np.clip(predictions, 0, 5)
            print(f"Loaded {name} model")
    
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
    
    # Calculate weights for ensemble (inverse of RMSE, normalized)
    # Lower RMSE = Higher weight
    inverse_rmse = {name: 1.0 / rmse for name, rmse in model_avg_rmse.items()}
    total_inverse = sum(inverse_rmse.values())
    ensemble_weights = {name: inv / total_inverse for name, inv in inverse_rmse.items()}
    
    print("\nEnsemble weights:")
    for name, weight in ensemble_weights.items():
        print(f"  {name}: {weight:.3f}")
    
    # Calculate weighted ensemble predictions
    ensemble_predictions = np.zeros((len(X_train), len(TARGET_COLS)))
    for model_name, predictions in all_predictions.items():
        ensemble_predictions += predictions * ensemble_weights[model_name]
    
    # Calculate ensemble RMSE
    muscle_rmse['Ensemble'] = {}
    ensemble_rmse_values = []
    for i, muscle in enumerate(TARGET_COLS):
        y_true = week26_outcomes_filtered[muscle].values
        y_pred = ensemble_predictions[:, i]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        muscle_rmse['Ensemble'][muscle] = rmse
        ensemble_rmse_values.append(rmse)
    
    print(f"\nWeighted Ensemble average RMSE: {np.mean(ensemble_rmse_values):.3f}")
    
    # Calculate group-level RMSE
    group_rmse = {}
    
    # For individual models, average across left and right
    for model_name in model_names:
        group_rmse[model_name] = {}
        for group_name, muscles in MUSCLE_GROUPS.items():
            # Average RMSE across left and right muscles
            rmse_values = [muscle_rmse[model_name][muscle] for muscle in muscles]
            group_rmse[model_name][group_name] = np.mean(rmse_values)
    
    # For ensemble, separate left and right
    group_rmse['Ensemble Left'] = {}
    group_rmse['Ensemble Right'] = {}
    group_rmse['Ensemble (Combined)'] = {}
    
    for group_name, muscles in MUSCLE_GROUPS.items():
        # Left muscle (ends with 'l')
        left_muscle = [m for m in muscles if m.endswith('l')][0]
        group_rmse['Ensemble Left'][group_name] = muscle_rmse['Ensemble'][left_muscle]
        
        # Right muscle (ends with 'r')
        right_muscle = [m for m in muscles if m.endswith('r')][0]
        group_rmse['Ensemble Right'][group_name] = muscle_rmse['Ensemble'][right_muscle]
        
        # Combined ensemble
        group_rmse['Ensemble (Combined)'][group_name] = np.mean([
            muscle_rmse['Ensemble'][left_muscle],
            muscle_rmse['Ensemble'][right_muscle]
        ])
    
    return group_rmse, MUSCLE_GROUPS


def create_3d_rmse_radar(group_rmse, muscle_groups):
    """Create interactive 3D RMSE radar plot with enhanced visuals."""
    
    # Prepare data for 3D visualization
    fig = go.Figure()
    
    # Define angles for radar plot (10 muscle groups)
    group_names = list(muscle_groups.keys())
    n_groups = len(group_names)
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
    
    # Enhanced color palette with gradients
    colors = {
        'CatBoost': 'rgba(255, 107, 107, 0.8)',      # Warm red
        'XGBoost': 'rgba(78, 205, 196, 0.8)',        # Teal
        'HistGB': 'rgba(69, 183, 209, 0.8)',         # Sky blue
        'Ensemble Left': 'rgba(138, 43, 226, 0.9)',   # Blue violet
        'Ensemble Right': 'rgba(255, 140, 0, 0.9)',   # Dark orange
        'Ensemble (Combined)': 'rgba(34, 139, 34, 0.9)' # Forest green
    }
    
    # Z-heights for different traces
    z_heights = {
        'CatBoost': 0.05,
        'XGBoost': 0.10,
        'HistGB': 0.15,
        'Ensemble (Combined)': 0.20,
        'Ensemble Left': 0.25,
        'Ensemble Right': 0.30
    }
    
    # Add gradient background circles
    for radius in [0.2, 0.4, 0.6, 0.8, 1.0]:
        circle_angles = np.linspace(0, 2 * np.pi, 100)
        x_circle = radius * np.cos(circle_angles)
        y_circle = radius * np.sin(circle_angles)
        z_circle = [0] * len(circle_angles)
        
        opacity = 0.1 + (1.0 - radius) * 0.2  # Darker towards center
        
        fig.add_trace(go.Scatter3d(
            x=x_circle,
            y=y_circle,
            z=z_circle,
            mode='lines',
            line=dict(color=f'rgba(200, 200, 200, {opacity})', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add radius labels with better styling
        fig.add_trace(go.Scatter3d(
            x=[radius * 1.05],
            y=[0],
            z=[0],
            mode='text',
            text=[f'{radius:.1f}'],
            textposition='middle right',
            textfont=dict(size=12, color='rgba(100, 100, 100, 0.8)', family='Arial'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add radial lines from center
    for angle in angles:
        x_rad = [0, 1.1 * np.cos(angle)]
        y_rad = [0, 1.1 * np.sin(angle)]
        z_rad = [0, 0]
        
        fig.add_trace(go.Scatter3d(
            x=x_rad,
            y=y_rad,
            z=z_rad,
            mode='lines',
            line=dict(color='rgba(180, 180, 180, 0.3)', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add traces for each model with enhanced styling
    for model_name in ['CatBoost', 'XGBoost', 'HistGB', 'Ensemble (Combined)', 'Ensemble Left', 'Ensemble Right']:
        if model_name in group_rmse:
            # Get RMSE values in order
            rmse_values = [group_rmse[model_name][group] for group in group_names]
            
            # Close the radar plot
            rmse_values_closed = rmse_values + [rmse_values[0]]
            angles_closed = np.append(angles, angles[0])
            
            # Convert to 3D coordinates
            x = rmse_values_closed * np.cos(angles_closed)
            y = rmse_values_closed * np.sin(angles_closed)
            z = [z_heights[model_name]] * len(rmse_values_closed)
            
            # Determine line style
            is_ensemble = 'Ensemble' in model_name
            line_width = 6 if is_ensemble else 3
            
            # Add shadow/glow effect for ensemble
            if is_ensemble:
                fig.add_trace(go.Scatter3d(
                    x=x,
                    y=y,
                    z=[z_heights[model_name] - 0.01] * len(rmse_values_closed),
                    mode='lines',
                    line=dict(color='rgba(0, 0, 0, 0.2)', width=line_width + 2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add the main trace
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines+markers',
                name=model_name,
                line=dict(
                    color=colors[model_name], 
                    width=line_width,
                    dash='solid' if is_ensemble else 'dash'
                ),
                marker=dict(
                    size=12 if is_ensemble else 8, 
                    color=colors[model_name],
                    line=dict(color='white', width=2) if is_ensemble else dict()
                ),
                hovertemplate='<b>%{text}</b><br>RMSE: %{customdata:.3f}<extra></extra>',
                text=group_names + [group_names[0]],
                customdata=rmse_values_closed
            ))
    
    # Add muscle group labels with enhanced styling
    label_distance = 1.25
    
    for i, (angle, group) in enumerate(zip(angles, group_names)):
        x_label = label_distance * np.cos(angle)
        y_label = label_distance * np.sin(angle)
        
        # Add background for labels
        fig.add_trace(go.Scatter3d(
            x=[x_label],
            y=[y_label],
            z=[0.35],
            mode='markers',
            marker=dict(size=40, color='white', opacity=0.8),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add the text
        fig.add_trace(go.Scatter3d(
            x=[x_label],
            y=[y_label],
            z=[0.35],
            mode='text',
            text=[group],
            textposition='middle center',
            textfont=dict(size=13, color='#2C3E50', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Calculate statistics
    avg_ensemble = np.mean(list(group_rmse['Ensemble (Combined)'].values()))
    avg_ensemble_left = np.mean(list(group_rmse['Ensemble Left'].values()))
    avg_ensemble_right = np.mean(list(group_rmse['Ensemble Right'].values()))
    
    best_overall = min(group_rmse['Ensemble (Combined)'].items(), key=lambda x: x[1])
    worst_overall = max(group_rmse['Ensemble (Combined)'].items(), key=lambda x: x[1])
    
    # Update layout with professional styling
    fig.update_layout(
        title=dict(
            text='<b>3D RMSE Performance Analysis</b><br><sup>Muscle Group Prediction Accuracy</sup>',
            font=dict(size=28, color='#2C3E50', family='Arial'),
            x=0.5,
            xanchor='center',
            y=0.98
        ),
        scene=dict(
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                showline=False,
                zeroline=False,
                title='',
                showspikes=False,
                showbackground=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                showline=False,
                zeroline=False,
                title='',
                showspikes=False,
                showbackground=False
            ),
            zaxis=dict(
                showgrid=False,
                showticklabels=False,
                showline=False,
                zeroline=False,
                title='',
                range=[-0.05, 0.5],
                showspikes=False,
                showbackground=False
            ),
            bgcolor='rgba(250, 250, 250, 0.95)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0.1),
                up=dict(x=0, y=0, z=1)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#E0E0E0',
            borderwidth=2,
            font=dict(color='#2C3E50', size=12, family='Arial'),
            title=dict(text='<b>Models</b>', font=dict(size=14, color='#2C3E50')),
            itemsizing='constant',
            itemwidth=30
        ),
        paper_bgcolor='#FAFAFA',
        plot_bgcolor='rgba(250, 250, 250, 0.95)',
        height=850,
        margin=dict(l=20, r=20, b=20, t=80),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            bordercolor="#E0E0E0"
        )
    )
    
    # Add performance summary with gradient background
    fig.add_annotation(
        text=(f"<b>Performance Metrics</b><br>"
              f"<span style='color: #34A853'>Ensemble Combined: {avg_ensemble:.3f}</span><br>"
              f"<span style='color: #8A2BE2'>Ensemble Left: {avg_ensemble_left:.3f}</span><br>"
              f"<span style='color: #FF8C00'>Ensemble Right: {avg_ensemble_right:.3f}</span><br>"
              f"<br><b>Best Performance:</b><br>{best_overall[0]} ({best_overall[1]:.3f})<br>"
              f"<b>Worst Performance:</b><br>{worst_overall[0]} ({worst_overall[1]:.3f})"),
        xref="paper", yref="paper",
        x=0.98, y=0.02,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=12, color="#2C3E50", family='Arial'),
        bgcolor="rgba(255,255,255,0.98)",
        bordercolor="#E0E0E0",
        borderwidth=2,
        borderpad=10,
        opacity=0.95
    )
    
    # Add interpretation guide with icons
    fig.add_annotation(
        text=(f"<b>📊 Visualization Guide</b><br>"
              f"• <b>Distance from center</b> = RMSE value<br>"
              f"• <b>Closer to center</b> = Better accuracy<br>"
              f"• <b>Solid lines</b> = Ensemble models<br>"
              f"• <b>Dashed lines</b> = Individual models<br>"
              f"• <b>Height (Z-axis)</b> = Model layers"),
        xref="paper", yref="paper",
        x=0.98, y=0.55,
        xanchor="right", yanchor="middle",
        showarrow=False,
        font=dict(size=11, color="#34495E", family='Arial'),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#BDC3C7",
        borderwidth=1,
        borderpad=8
    )
    
    # Add a subtle watermark
    fig.add_annotation(
        text="ASIA Motor Score Prediction Model",
        xref="paper", yref="paper",
        x=0.5, y=0.01,
        xanchor="center", yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color="#BDC3C7", family='Arial'),
        opacity=0.5
    )
    
    return fig


def create_interactive_rmse_visualization():
    """Main function to create the visualization."""
    print("Loading data and calculating RMSE...")
    group_rmse, muscle_groups = load_data_and_calculate_rmse()
    
    print("\nModel Performance Summary:")
    for model in ['CatBoost', 'XGBoost', 'HistGB', 'Ensemble (Combined)']:
        if model in group_rmse:
            avg_rmse = np.mean(list(group_rmse[model].values()))
            print(f"{model}: Average RMSE = {avg_rmse:.3f}")
    
    print("\nCreating 3D RMSE radar plot...")
    fig = create_3d_rmse_radar(group_rmse, muscle_groups)
    
    return fig


if __name__ == "__main__":
    fig = create_interactive_rmse_visualization()
    fig.show()
    # Save as HTML
    fig.write_html("asia-impairment-track-prediction/visuals/figures/3d_rmse_radar.html")
    print("\nSaved to asia-impairment-track-prediction/visuals/figures/3d_rmse_radar.html")
