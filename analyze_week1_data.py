"""Analyze week 1 baseline data structure"""
import pandas as pd
from pathlib import Path

base_path = Path("asia-impairment-track-prediction")

# Load the raw training features (week 1 data)
train_features = pd.read_csv(base_path / "data" / "train_features.csv")
print(f"Train features shape: {train_features.shape}")
print(f"Columns: {train_features.columns.tolist()}")

# Check if week 1 motor scores are in the features
motor_cols = ['elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
              'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
              'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
              'gretor', 'ankplr']

week1_motor_cols = [col for col in train_features.columns if any(motor in col for motor in motor_cols)]
print(f"\nWeek 1 motor-related columns: {week1_motor_cols}")

# Load outcomes to get week 26 actual values
outcomes = pd.read_csv(base_path / "data" / "train_outcomes_MS.csv")
week26_outcomes = outcomes[outcomes['time'] == 26].copy()
print(f"\nWeek 26 outcomes shape: {week26_outcomes.shape}")

# Check if we can match PIDs
common_pids = set(train_features['PID']).intersection(set(week26_outcomes['PID']))
print(f"\nPatients with both week 1 and week 26 data: {len(common_pids)}")

# Calculate week 1 total motor scores if available
if week1_motor_cols:
    # Check first few values
    print(f"\nFirst 5 rows of week 1 motor columns:")
    print(train_features[week1_motor_cols].head())
    
    # Check if they're numeric
    print(f"\nData types of motor columns:")
    print(train_features[week1_motor_cols].dtypes)
    
    # Try to calculate total scores
    week1_totals = train_features[week1_motor_cols].sum(axis=1)
    print(f"\nWeek 1 total motor scores (first 10): {week1_totals.head(10).tolist()}")
else:
    print("\nNo week 1 motor scores found in features - need to check for different column names")
