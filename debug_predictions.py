"""Debug script to understand prediction structure"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load data and models
base_path = Path("asia-impairment-track-prediction")

# Load processed data
train_df = pd.read_csv(base_path / "data" / "train_processed.csv")
print(f"Train data shape: {train_df.shape}")

# Define target columns
TARGET_COLS = [
    'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
    'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
    'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
    'gretor', 'ankplr'
]

# Remove PID and targets
if 'PID' in train_df.columns:
    train_df_no_pid = train_df.drop(columns=['PID'])
else:
    train_df_no_pid = train_df

X_cols = [c for c in train_df_no_pid.columns if c not in TARGET_COLS]
X_train = train_df_no_pid[X_cols].values

print(f"X_train shape: {X_train.shape}")

# Load one model to test
xgb_model = joblib.load(base_path / "models_exact" / "xgb_exact_model.pkl")

# Make predictions on first 5 samples
X_sample = X_train[:5]
predictions = xgb_model.predict(X_sample)

print(f"\nPrediction shape: {predictions.shape}")
print(f"Expected shape for 20 muscles × 2 timepoints: (5, 40)")
print(f"Expected shape for 20 muscles × 1 timepoint: (5, 20)")

# Check if it's 40 columns (20 muscles × 2 timepoints)
if predictions.shape[1] == 40:
    print("\nModel predicts 40 values per patient (20 muscles × 2 timepoints)")
    print("First half (0:20) = Week 26, Second half (20:40) = Week 52")
    
    # Calculate total motor scores
    week26_scores = predictions[:, :20].sum(axis=1)
    week52_scores = predictions[:, 20:].sum(axis=1)
    
    print(f"\nWeek 26 total motor scores: {week26_scores}")
    print(f"Week 52 total motor scores: {week52_scores}")
elif predictions.shape[1] == 20:
    print("\nModel predicts 20 values per patient (20 muscles for one timepoint)")
    total_scores = predictions.sum(axis=1)
    print(f"Total motor scores: {total_scores}")
