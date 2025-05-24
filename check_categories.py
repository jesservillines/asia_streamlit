"""Check category distributions in metadata"""
import pandas as pd
from pathlib import Path

base_path = Path("asia-impairment-track-prediction")
metadata = pd.read_csv(base_path / "data" / "metadata.csv")

print("Age categories:")
print(metadata['age_category'].value_counts().sort_index())

print("\nBMI categories:")
print(metadata['bmi_category'].value_counts().sort_index())

print("\nChecking for missing values in categories:")
print(f"Age category missing: {metadata['age_category'].isna().sum()}")
print(f"BMI category missing: {metadata['bmi_category'].isna().sum()}")

# Check a few sample rows
print("\nSample metadata rows:")
print(metadata[['PID', 'age_category', 'bmi_category', 'sexcd']].head(10))
