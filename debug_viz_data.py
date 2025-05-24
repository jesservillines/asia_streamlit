"""Debug visualization data issues"""
import pandas as pd
import numpy as np
from pathlib import Path

base_path = Path("asia-impairment-track-prediction")

# Load all data
train_features = pd.read_csv(base_path / "data" / "train_features.csv")
train_df = pd.read_csv(base_path / "data" / "train_processed.csv")
outcomes_df = pd.read_csv(base_path / "data" / "train_outcomes_MS.csv")
metadata = pd.read_csv(base_path / "data" / "metadata.csv")

print(f"Train features shape: {train_features.shape}")
print(f"Train processed shape: {train_df.shape}")
print(f"Metadata shape: {metadata.shape}")

# Check week 26 outcomes
week26_outcomes = outcomes_df[outcomes_df['time'] == 26]
print(f"\nWeek 26 outcomes shape: {week26_outcomes.shape}")

# Check common PIDs
pids_features = set(train_features['PID'])
pids_processed = set(train_df['PID'])
pids_week26 = set(week26_outcomes['PID'])
pids_metadata = set(metadata['PID'])

print(f"\nPIDs in train_features: {len(pids_features)}")
print(f"PIDs in train_processed: {len(pids_processed)}")
print(f"PIDs in week26_outcomes: {len(pids_week26)}")
print(f"PIDs in metadata: {len(pids_metadata)}")

# Find common PIDs across all datasets
common_pids = pids_features.intersection(pids_processed).intersection(pids_week26)
print(f"\nCommon PIDs across all datasets: {len(common_pids)}")

# Check if metadata has the expected columns
print(f"\nMetadata columns: {metadata.columns.tolist()}")

# Check what demographic data we actually have
if 'age' in metadata.columns:
    print(f"\nAge range: {metadata['age'].min()} - {metadata['age'].max()}")
    print(f"Age distribution:")
    print(metadata['age'].value_counts(bins=5).sort_index())

if 'bmi' in metadata.columns:
    print(f"\nBMI range: {metadata['bmi'].min():.1f} - {metadata['bmi'].max():.1f}")
    print(f"BMI distribution:")
    print(metadata['bmi'].value_counts(bins=5).sort_index())

if 'sexcd' in metadata.columns:
    print(f"\nSex distribution:")
    print(metadata['sexcd'].value_counts())

# Check if age_category and bmi_category exist
if 'age_category' not in metadata.columns:
    print("\nage_category not found in metadata - need to create it")
if 'bmi_category' not in metadata.columns:
    print("bmi_category not found in metadata - need to create it")

# Check AIS grades from train_features
if 'ais1' in train_features.columns:
    print(f"\nWeek 1 AIS distribution:")
    print(train_features['ais1'].value_counts().sort_index())

# Sample a few common PIDs to verify data
sample_pids = list(common_pids)[:5]
print(f"\nSample PIDs to check: {sample_pids}")

# Check if these PIDs have all required data
for pid in sample_pids:
    has_features = pid in pids_features
    has_processed = pid in pids_processed
    has_week26 = pid in pids_week26
    has_metadata = pid in pids_metadata
    print(f"PID {pid}: features={has_features}, processed={has_processed}, week26={has_week26}, metadata={has_metadata}")
