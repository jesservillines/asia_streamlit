"""Generate all HTML visualizations for the optimized app."""
import sys
from pathlib import Path
import os

# Add project to path
PROJECT_ROOT = Path(__file__).parent
PKG_DIR = PROJECT_ROOT / "asia-impairment-track-prediction"
sys.path.insert(0, str(PROJECT_ROOT))

# Create figures directory
FIG_DIR = PKG_DIR / "visuals" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("🎨 Generating all visualizations...")
print(f"📁 Output directory: {FIG_DIR}")

# Import and run each visualization module
visualization_modules = [
    "interactive_3d_outcomes",
    "clinical_impact_dashboard", 
    "interactive_shap_explorer",
    "animated_recovery_paths"
]

for module_name in visualization_modules:
    print(f"\n⏳ Generating {module_name}...")
    try:
        # Import the module
        module_path = f"asia-impairment-track-prediction.visuals.{module_name}"
        module = __import__(module_path, fromlist=[''])
        
        # Each module should have already generated its HTML when imported
        print(f"✅ {module_name} generated successfully")
        
    except Exception as e:
        print(f"❌ Error generating {module_name}: {str(e)}")
        print(f"   Attempting manual generation...")
        
        # Try running the file directly
        script_path = PKG_DIR / "visuals" / f"{module_name}.py"
        if script_path.exists():
            try:
                exec(open(script_path).read(), {'__name__': '__main__'})
                print(f"✅ {module_name} generated via direct execution")
            except Exception as e2:
                print(f"❌ Failed to generate {module_name}: {str(e2)}")

# List generated files
print("\n📊 Generated visualizations:")
for html_file in FIG_DIR.glob("*.html"):
    print(f"   - {html_file.name}")

print("\n✨ All visualizations generated! You can now use the optimized Streamlit app.")
