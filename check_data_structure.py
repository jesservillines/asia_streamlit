"""Check the structure of the processed data files."""
import pandas as pd
from pathlib import Path

# Check train_processed.csv
train_path = Path("asia-impairment-track-prediction/data/train_processed.csv")
if train_path.exists():
    print("=== TRAIN_PROCESSED.CSV ===")
    df = pd.read_csv(train_path)
    print(f"Shape: {df.shape}")
    print(f"\nFirst 30 columns:")
    for i, col in enumerate(df.columns[:30]):
        print(f"{i+1}. {col}")
    
    print(f"\nLast 10 columns:")
    for i, col in enumerate(df.columns[-10:]):
        print(f"{i+1}. {col}")
    
    # Check for target columns
    print("\n=== TARGET COLUMNS ===")
    target_cols = ['TotalMotorScore26', 'TotalMotorScore52']
    for col in target_cols:
        if col in df.columns:
            print(f"✓ Found {col}")
        else:
            print(f"✗ Missing {col}")
    
    # Check for week 1 features
    print("\n=== WEEK 1 FEATURES ===")
    week1_cols = [col for col in df.columns if col.endswith('01')]
    print(f"Found {len(week1_cols)} week 1 features:")
    for col in week1_cols[:10]:
        print(f"  - {col}")
    
    # Check data types
    print("\n=== DATA SAMPLE ===")
    print(df[['PID'] + target_cols].head(3))
    
else:
    print("train_processed.csv not found!")

# Check metadata
metadata_path = Path("asia-impairment-track-prediction/data/metadata.csv")
if metadata_path.exists():
    print("\n\n=== METADATA.CSV ===")
    meta_df = pd.read_csv(metadata_path)
    print(f"Shape: {meta_df.shape}")
    print(f"Columns: {meta_df.columns.tolist()}")
    print("\nSample:")
    print(meta_df.head(3))
