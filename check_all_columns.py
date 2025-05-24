"""Check all columns in the processed data."""
import pandas as pd
from pathlib import Path

# Check train_processed.csv
train_path = Path("asia-impairment-track-prediction/data/train_processed.csv")
df = pd.read_csv(train_path)

print("=== ALL COLUMNS IN TRAIN_PROCESSED.CSV ===")
print(f"Total columns: {len(df.columns)}")
print("\nAll columns:")
for i, col in enumerate(df.columns):
    print(f"{i+1:3d}. {col}")

# Check if PID is there
print(f"\nPID column present: {'PID' in df.columns}")

# Check for any motor score related columns
print("\n=== MOTOR SCORE RELATED COLUMNS ===")
motor_cols = [col for col in df.columns if 'motor' in col.lower() or 'score' in col.lower()]
if motor_cols:
    for col in motor_cols:
        print(f"  - {col}")
else:
    print("No motor score columns found!")

# Check train_outcomes_MS.csv
outcomes_path = Path("asia-impairment-track-prediction/data/train_outcomes_MS.csv")
if outcomes_path.exists():
    print("\n\n=== TRAIN_OUTCOMES_MS.CSV ===")
    outcomes_df = pd.read_csv(outcomes_path)
    print(f"Shape: {outcomes_df.shape}")
    print(f"Columns: {outcomes_df.columns.tolist()}")
    print("\nFirst 3 rows:")
    print(outcomes_df.head(3))
