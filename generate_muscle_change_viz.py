"""Generate visualization for individual muscle changes from week 1 to week 26."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_muscle_data():
    """Load data for muscle-level analysis."""
    base_path = Path("asia-impairment-track-prediction")
    
    # Load raw features to get week 1 motor scores
    train_features = pd.read_csv(base_path / "data" / "train_features.csv")
    
    # Load week 26 outcomes
    outcomes_df = pd.read_csv(base_path / "data" / "train_outcomes_MS.csv")
    week26_outcomes = outcomes_df[outcomes_df['time'] == 26].copy()
    
    # Load metadata
    metadata = pd.read_csv(base_path / "data" / "metadata.csv")
    
    # Define muscle columns
    MUSCLE_NAMES = {
        'elbfll': 'L Elbow Flexors',
        'wrextl': 'L Wrist Extensors', 
        'elbexl': 'L Elbow Extensors',
        'finfll': 'L Finger Flexors',
        'finabl': 'L Finger Abductors',
        'hipfll': 'L Hip Flexors',
        'kneexl': 'L Knee Extensors',
        'ankdol': 'L Ankle Dorsiflexors',
        'gretol': 'L Great Toe Extensors',
        'ankpll': 'L Ankle Plantarflexors',
        'elbflr': 'R Elbow Flexors',
        'wrextr': 'R Wrist Extensors',
        'elbexr': 'R Elbow Extensors',
        'finflr': 'R Finger Flexors',
        'finabr': 'R Finger Abductors',
        'hipflr': 'R Hip Flexors',
        'kneetr': 'R Knee Extensors',
        'ankdor': 'R Ankle Dorsiflexors',
        'gretor': 'R Great Toe Extensors',
        'ankplr': 'R Ankle Plantarflexors'
    }
    
    TARGET_COLS = list(MUSCLE_NAMES.keys())
    
    # Get week 1 muscle scores
    week1_muscle_cols = [f"{col}01" for col in TARGET_COLS]
    
    # Filter to common patients
    common_pids = set(train_features['PID']).intersection(set(week26_outcomes['PID']))
    common_pids = sorted(list(common_pids))
    
    # Filter and align data
    train_features_filtered = train_features[train_features['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    week26_outcomes_filtered = week26_outcomes[week26_outcomes['PID'].isin(common_pids)].set_index('PID').loc[common_pids].reset_index()
    
    # Get muscle scores
    week1_muscles = train_features_filtered[week1_muscle_cols].values
    week26_muscles = week26_outcomes_filtered[TARGET_COLS].values
    
    # Calculate changes
    muscle_changes = week26_muscles - week1_muscles
    
    # Get patient info
    pids = train_features_filtered['PID'].values
    
    # Get AIS grades
    week1_ais = train_features_filtered['ais1'].values if 'ais1' in train_features_filtered.columns else None
    
    return pids, week1_muscles, week26_muscles, muscle_changes, TARGET_COLS, MUSCLE_NAMES, metadata, week1_ais


def create_muscle_change_heatmap(
    patient_filter=None,
    ais_filter=None,
    muscle_group_filter=None,
    sort_by='total_gain',
    show_values=True,
    colorscale='RdBu'
):
    """Create heatmap showing muscle changes for each patient."""
    
    # Load data
    pids, week1_muscles, week26_muscles, muscle_changes, target_cols, muscle_names, metadata, week1_ais = load_muscle_data()
    
    # Create dataframe for filtering
    df = pd.DataFrame({
        'PID': pids,
        'total_gain': muscle_changes.sum(axis=1),
        'week1_total': week1_muscles.sum(axis=1),
        'week26_total': week26_muscles.sum(axis=1)
    })
    
    # Add AIS grades
    if week1_ais is not None:
        df['week1_ais'] = week1_ais
    
    # Calculate week 26 AIS based on total score
    df['week26_ais'] = pd.cut(df['week26_total'], 
                               bins=[-1, 20, 50, 80, 100], 
                               labels=['A', 'B', 'C', 'D'])
    
    # Apply filters
    mask = pd.Series([True] * len(df))
    
    if patient_filter:
        mask &= df['PID'].isin(patient_filter)
    
    if ais_filter:
        mask &= df['week1_ais'].isin([f'AIS {grade}' for grade in ais_filter])
    
    filtered_indices = df[mask].index
    filtered_pids = df.loc[filtered_indices, 'PID'].values
    filtered_changes = muscle_changes[filtered_indices]
    
    # Filter muscles if requested
    if muscle_group_filter:
        muscle_indices = []
        muscle_labels = []
        for i, (col, name) in enumerate(zip(target_cols, muscle_names.values())):
            include = False
            if 'Upper' in muscle_group_filter and any(part in name for part in ['Elbow', 'Wrist', 'Finger']):
                include = True
            if 'Lower' in muscle_group_filter and any(part in name for part in ['Hip', 'Knee', 'Ankle', 'Toe']):
                include = True
            if 'Left' in muscle_group_filter and 'L ' in name:
                include = True
            if 'Right' in muscle_group_filter and 'R ' in name:
                include = True
            
            if include:
                muscle_indices.append(i)
                muscle_labels.append(name)
        
        filtered_changes = filtered_changes[:, muscle_indices]
    else:
        muscle_labels = list(muscle_names.values())
    
    # Sort patients
    if sort_by == 'total_gain':
        sort_indices = np.argsort(-filtered_changes.sum(axis=1))
    elif sort_by == 'pid':
        sort_indices = np.argsort(filtered_pids)
    else:  # by AIS grade
        ais_order = {'AIS A': 0, 'AIS B': 1, 'AIS C': 2, 'AIS D': 3}
        ais_values = df.loc[filtered_indices, 'week1_ais'].map(ais_order).fillna(4).values
        sort_indices = np.argsort(ais_values)
    
    sorted_changes = filtered_changes[sort_indices]
    sorted_pids = filtered_pids[sort_indices]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sorted_changes.T,
        x=[f"{pid} (Δ={sorted_changes[i].sum():.0f})" for i, pid in enumerate(sorted_pids)],
        y=muscle_labels,
        colorscale=colorscale,
        zmid=0,
        zmin=-5,
        zmax=5,
        text=sorted_changes.T if show_values else None,
        texttemplate='%{text}' if show_values else None,
        textfont={"size": 8},
        hovertemplate='Patient: %{x}<br>Muscle: %{y}<br>Change: %{z}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Individual Muscle Changes: Week 1 to Week 26<br><sub>Change in motor score (0-5 scale) for each muscle | {len(sorted_pids)} patients</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Patients (sorted by ' + sort_by.replace('_', ' ') + ')',
            tickangle=-45,
            tickfont=dict(size=8)
        ),
        yaxis=dict(
            title='Muscles',
            tickfont=dict(size=10)
        ),
        height=600,
        margin=dict(l=150, r=50, t=100, b=150)
    )
    
    # Add colorbar title
    fig.update_coloraxes(
        colorbar_title_text="Score<br>Change",
        colorbar_thickness=15
    )
    
    return fig


def create_muscle_change_summary(
    ais_filter=None,
    view_type='average'
):
    """Create summary visualization of muscle changes by group."""
    
    # Load data
    pids, week1_muscles, week26_muscles, muscle_changes, target_cols, muscle_names, metadata, week1_ais = load_muscle_data()
    
    # Create dataframe
    df = pd.DataFrame({
        'PID': pids,
        'week1_ais': week1_ais if week1_ais is not None else ['Unknown'] * len(pids)
    })
    
    # Apply AIS filter
    if ais_filter:
        mask = df['week1_ais'].isin([f'AIS {grade}' for grade in ais_filter])
        filtered_changes = muscle_changes[mask]
    else:
        filtered_changes = muscle_changes
    
    if view_type == 'average':
        # Calculate average change per muscle
        avg_changes = np.mean(filtered_changes, axis=0)
        std_changes = np.std(filtered_changes, axis=0)
        
        # Create bar chart
        fig = go.Figure()
        
        # Sort by average change
        sort_indices = np.argsort(-avg_changes)
        
        fig.add_trace(go.Bar(
            x=[muscle_names[target_cols[i]] for i in sort_indices],
            y=avg_changes[sort_indices],
            error_y=dict(
                type='data',
                array=std_changes[sort_indices],
                visible=True
            ),
            marker_color=['#2E86AB' if 'L ' in muscle_names[target_cols[i]] else '#A23B72' 
                         for i in sort_indices],
            hovertemplate='%{x}<br>Avg Change: %{y:.2f} ± %{error_y.array:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Average Muscle Score Changes (Week 1 to 26)',
            xaxis_title='Muscle',
            yaxis_title='Average Score Change',
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
    else:  # distribution view
        # Create box plots for each muscle
        fig = go.Figure()
        
        for i, (col, name) in enumerate(zip(target_cols, muscle_names.values())):
            fig.add_trace(go.Box(
                y=filtered_changes[:, i],
                name=name,
                boxpoints='outliers',
                marker_color='#2E86AB' if 'L ' in name else '#A23B72',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'Distribution of Muscle Score Changes (Week 1 to 26)',
            yaxis_title='Score Change',
            xaxis_tickangle=-45,
            height=500
        )
    
    return fig


def create_patient_muscle_radar(patient_id):
    """Create radar chart showing individual patient's muscle changes."""
    
    # Load data
    pids, week1_muscles, week26_muscles, muscle_changes, target_cols, muscle_names, metadata, week1_ais = load_muscle_data()
    
    # Find patient index
    try:
        patient_idx = list(pids).index(patient_id)
    except ValueError:
        return None
    
    # Get patient data
    week1_scores = week1_muscles[patient_idx]
    week26_scores = week26_muscles[patient_idx]
    changes = muscle_changes[patient_idx]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add week 1 scores
    fig.add_trace(go.Scatterpolar(
        r=week1_scores,
        theta=list(muscle_names.values()),
        fill='toself',
        name='Week 1',
        line_color='red',
        opacity=0.6
    ))
    
    # Add week 26 scores
    fig.add_trace(go.Scatterpolar(
        r=week26_scores,
        theta=list(muscle_names.values()),
        fill='toself',
        name='Week 26',
        line_color='green',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=True,
        title=f"Patient {patient_id}: Muscle Score Progression<br><sub>Total gain: {changes.sum():.0f} points</sub>",
        height=600
    )
    
    return fig


if __name__ == "__main__":
    # Test visualizations
    output_dir = Path("asia-impairment-track-prediction/visuals/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create heatmap
    fig1 = create_muscle_change_heatmap(sort_by='total_gain')
    fig1.write_html(str(output_dir / "muscle_change_heatmap.html"))
    print("Generated muscle change heatmap")
    
    # Create summary
    fig2 = create_muscle_change_summary(view_type='average')
    fig2.write_html(str(output_dir / "muscle_change_summary.html"))
    print("Generated muscle change summary")
