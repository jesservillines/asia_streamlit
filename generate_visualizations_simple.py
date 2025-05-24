"""Simple script to generate all visualizations by running each script directly."""
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VISUALS_DIR = PROJECT_ROOT / "asia-impairment-track-prediction" / "visuals"
FIG_DIR = VISUALS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

scripts = [
    "interactive_3d_outcomes.py",
    "clinical_impact_dashboard.py",
    "interactive_shap_explorer.py", 
    "animated_recovery_paths.py"
]

print("🎨 Generating visualizations...")
print(f"📁 Output directory: {FIG_DIR}\n")

for script in scripts:
    script_path = VISUALS_DIR / script
    if script_path.exists():
        print(f"⏳ Running {script}...")
        try:
            # Run script in its own directory with proper Python path
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(VISUALS_DIR),
                capture_output=True,
                text=True,
                env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)}
            )
            
            if result.returncode == 0:
                print(f"✅ {script} completed successfully")
            else:
                print(f"❌ {script} failed with error:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Failed to run {script}: {e}")
    else:
        print(f"⚠️  {script} not found")

print("\n📊 Generated visualizations:")
for html_file in FIG_DIR.glob("*.html"):
    print(f"   - {html_file.name}")
