"""Generate dynamic 3D visualization with customizable parameters."""
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path

def create_dynamic_3d_visualization(
    age_filter=None, 
    bmi_filter=None, 
    sex_filter=None, 
    ais_filter=None,
    point_size=10,
    opacity=0.7,
    show_grid=True,
    show_legend=True
):
    """Create a 3D scatter plot with custom parameters."""
    
    # Generate sample data (in production, this would filter real data)
    np.random.seed(42)
    n_patients = 500
    
    # Create synthetic patient data
    data = {
        'patient_id': [f'P{i:04d}' for i in range(n_patients)],
        'age_category': np.random.choice(['<45', '45-65', '>65'], n_patients),
        'bmi_category': np.random.choice(['Underweight', 'Healthy', 'Overweight', 'Obese'], n_patients),
        'sex': np.random.choice(['Male', 'Female'], n_patients),
        'ais_grade': np.random.choice(['A', 'B', 'C', 'D'], n_patients, p=[0.2, 0.3, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Apply filters
    mask = pd.Series([True] * len(df))
    if age_filter:
        mask &= df['age_category'].isin(age_filter)
    if bmi_filter:
        mask &= df['bmi_category'].isin(bmi_filter)
    if sex_filter:
        mask &= df['sex'].isin(sex_filter)
    if ais_filter:
        mask &= df['ais_grade'].isin(ais_filter)
    
    df_filtered = df[mask].copy()
    
    # Generate outcome scores based on AIS grade
    ais_base_scores = {'A': 20, 'B': 40, 'C': 60, 'D': 80}
    
    df_filtered['motor_6mo'] = df_filtered['ais_grade'].map(ais_base_scores) + np.random.normal(0, 10, len(df_filtered))
    df_filtered['motor_12mo'] = df_filtered['motor_6mo'] + np.random.normal(10, 5, len(df_filtered))
    df_filtered['confidence'] = 0.7 + 0.3 * np.random.rand(len(df_filtered))
    
    # Clip values to reasonable ranges
    df_filtered['motor_6mo'] = np.clip(df_filtered['motor_6mo'], 0, 100)
    df_filtered['motor_12mo'] = np.clip(df_filtered['motor_12mo'], 0, 100)
    
    # Create color mapping
    color_map = {'A': '#ff6b6b', 'B': '#ffd93d', 'C': '#6bcf7f', 'D': '#4ecdc4'}
    df_filtered['color'] = df_filtered['ais_grade'].map(color_map)
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add trace for each AIS grade
    for grade in ['A', 'B', 'C', 'D']:
        if grade in df_filtered['ais_grade'].values:
            grade_data = df_filtered[df_filtered['ais_grade'] == grade]
            
            fig.add_trace(go.Scatter3d(
                x=grade_data['motor_6mo'],
                y=grade_data['motor_12mo'],
                z=grade_data['confidence'],
                mode='markers',
                name=f'AIS {grade}',
                marker=dict(
                    size=point_size,
                    color=color_map[grade],
                    opacity=opacity,
                    line=dict(color='white', width=1)
                ),
                text=[f"Patient: {pid}<br>Age: {age}<br>BMI: {bmi}<br>Sex: {sex}<br>6mo: {m6:.1f}<br>12mo: {m12:.1f}<br>Confidence: {conf:.2f}" 
                      for pid, age, bmi, sex, m6, m12, conf in zip(
                          grade_data['patient_id'],
                          grade_data['age_category'],
                          grade_data['bmi_category'],
                          grade_data['sex'],
                          grade_data['motor_6mo'],
                          grade_data['motor_12mo'],
                          grade_data['confidence']
                      )],
                hovertemplate='%{text}<extra></extra>',
                showlegend=show_legend
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'3D Patient Outcome Visualization<br><sub>Showing {len(df_filtered)} patients</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title='Motor Score at 6 Months',
                showgrid=show_grid,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Motor Score at 12 Months',
                showgrid=show_grid,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            zaxis=dict(
                title='Prediction Confidence',
                showgrid=show_grid,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=700,
        template='plotly_white',
        showlegend=show_legend,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    return fig

if __name__ == "__main__":
    # Generate default visualization
    fig = create_dynamic_3d_visualization()
    
    # Save to file
    output_dir = Path("asia-impairment-track-prediction/visuals/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(output_dir / "dynamic_3d_outcomes.html"))
    print("✅ Generated dynamic 3D visualization")
