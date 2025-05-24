"""Check the structure of outcomes data"""
import pandas as pd
from pathlib import Path

base_path = Path("asia-impairment-track-prediction")

# Load outcomes
outcomes_df = pd.read_csv(base_path / "data" / "train_outcomes_MS.csv")
print(f"Outcomes shape: {outcomes_df.shape}")
print(f"Columns: {outcomes_df.columns.tolist()}")

# Check unique time values
print(f"\nUnique time values: {sorted(outcomes_df['time'].unique())}")

# Count patients per time point
time_counts = outcomes_df['time'].value_counts().sort_index()
print(f"\nPatients per time point:")
print(time_counts)

# Check if we have the same PIDs for both timepoints
pids_26 = set(outcomes_df[outcomes_df['time'] == 26]['PID'])
pids_52 = set(outcomes_df[outcomes_df['time'] == 52]['PID'])

print(f"\nPatients with week 26 data: {len(pids_26)}")
print(f"Patients with week 52 data: {len(pids_52)}")
print(f"Patients with both timepoints: {len(pids_26.intersection(pids_52))}")

# Check the processed training data
train_df = pd.read_csv(base_path / "data" / "train_processed.csv")

# Define target columns
TARGET_COLS = [
    'elbfll', 'wrextl', 'elbexl', 'finfll', 'finabl', 'hipfll',
    'kneexl', 'ankdol', 'gretol', 'ankpll', 'elbflr', 'wrextr',
    'elbexr', 'finflr', 'finabr', 'hipflr', 'kneetr', 'ankdor',
    'gretor', 'ankplr'
]

# Check if target columns are in the processed data
targets_in_processed = [col for col in TARGET_COLS if col in train_df.columns]
print(f"\nTarget columns in processed data: {len(targets_in_processed)}")

if targets_in_processed:
    print("Targets are included in processed data - this suggests single timepoint training")
    # Check which timepoint
    train_pids = set(train_df['PID'])
    if train_pids == pids_26:
        print("Training data matches week 26 outcomes")
    elif train_pids == pids_52:
        print("Training data matches week 52 outcomes")
    else:
        print("Training data PIDs don't match a specific timepoint exactly")
