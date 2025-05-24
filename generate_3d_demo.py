"""Generate a demo 3D visualization showing the concept of the real predictions."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def create_3d_demo_visualization(
    age_filter=None,
    bmi_filter=None, 
    sex_filter=None,
    ais_filter=None,
    point_size=10,
    opacity=0.7,
    show_grid=True,
    show_legend=True
):
    """Create a demo 3D visualization showing how real predictions would look."""
    
    # Generate realistic demo data based on typical ASIA patterns
    np.random.seed(42)
    n_patients = 300
    
    # Create patient data with realistic distributions
    ais_grades = np.random.choice(['A', 'B', 'C', 'D'], n_patients, p=[0.15, 0.25, 0.35, 0.25])
    age_categories = np.random.choice(['<45', '45-65', '>65'], n_patients, p=[0.3, 0.4, 0.3])
    bmi_categories = np.random.choice(['Underweight', 'Healthy', 'Overweight', 'Obese'], n_patients, p=[0.1, 0.4, 0.35, 0.15])
    sex = np.random.choice(['Male', 'Female'], n_patients, p=[0.7, 0.3])  # Typical SCI demographics
    
    # Generate realistic motor scores based on AIS grade
    # AIS A: severe (0-20 total), B: moderate (20-50), C: moderate-good (50-80), D: good (80-100)
    motor_6mo = np.zeros(n_patients)
    motor_12mo = np.zeros(n_patients)
    confidence = np.zeros(n_patients)
    
    for i, grade in enumerate(ais_grades):
        if grade == 'A':
            motor_6mo[i] = np.random.normal(15, 5)
            motor_12mo[i] = motor_6mo[i] + np.random.normal(5, 3)  # Small improvement
            confidence[i] = np.random.uniform(0.7, 0.9)  # High confidence (less variability)
        elif grade == 'B':
            motor_6mo[i] = np.random.normal(35, 10)
            motor_12mo[i] = motor_6mo[i] + np.random.normal(10, 5)  # Moderate improvement
            confidence[i] = np.random.uniform(0.6, 0.85)
        elif grade == 'C':
            motor_6mo[i] = np.random.normal(65, 12)
            motor_12mo[i] = motor_6mo[i] + np.random.normal(15, 7)  # Good improvement
            confidence[i] = np.random.uniform(0.5, 0.8)
        else:  # D
            motor_6mo[i] = np.random.normal(85, 8)
            motor_12mo[i] = motor_6mo[i] + np.random.normal(8, 4)  # Smaller improvement (already high)
            confidence[i] = np.random.uniform(0.65, 0.95)
    
    # Clip values to realistic ranges
    motor_6mo = np.clip(motor_6mo, 0, 100)
    motor_12mo = np.clip(motor_12mo, 0, 100)
    
    # Create dataframe
    df = pd.DataFrame({
        'PID': [f'P{i:04d}' for i in range(n_patients)],
        'age_category': age_categories,
        'bmi_category': bmi_categories,
        'sex': sex,
        'ais_grade': ais_grades,
        'motor_6mo': motor_6mo,
        'motor_12mo': motor_12mo,
        'confidence': confidence
    })
    
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
    
    # Create color mapping
    color_map = {'A': '#ff6b6b', 'B': '#ffd93d', 'C': '#6bcf7f', 'D': '#4ecdc4'}
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add trace for each AIS grade
    for grade in ['A', 'B', 'C', 'D']:
        grade_data = df_filtered[df_filtered['ais_grade'] == grade]
        if len(grade_data) > 0:
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
                          grade_data['PID'],
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
            text=f'3D Patient Outcome Predictions (Demo)<br><sub>Week 1 Features → Week 26 & 52 Motor Score Predictions | {len(df_filtered)} patients</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title='Predicted Total Motor Score<br>at 6 Months (Week 26)',
                showgrid=show_grid,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, 100]
            ),
            yaxis=dict(
                title='Predicted Total Motor Score<br>at 12 Months (Week 52)',
                showgrid=show_grid,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, 100]
            ),
            zaxis=dict(
                title='Model Confidence',
                showgrid=show_grid,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0.4, 1]
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
        ),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    # Add annotations
    fig.add_annotation(
        text="Demo visualization with realistic ASIA motor score patterns<br>In production, this uses actual ensemble model predictions from FeatureTools-engineered Week 1 data",
        xref="paper", yref="paper",
        x=0.5, y=-0.08,
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="center"
    )
    
    # Add insight box
    fig.add_annotation(
        text="<b>Key Insights:</b><br>• AIS A (red): Limited recovery<br>• AIS B (yellow): Moderate recovery<br>• AIS C (green): Good recovery<br>• AIS D (blue): Best outcomes",
        xref="paper", yref="paper",
        x=0.98, y=0.02,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=9),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4
    )
    
    return fig

if __name__ == "__main__":
    # Test the visualization
    fig = create_3d_demo_visualization()
    
    # Save to file
    output_dir = Path("asia-impairment-track-prediction/visuals/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(str(output_dir / "3d_demo_predictions.html"))
    print("✅ Generated demo 3D visualization")
