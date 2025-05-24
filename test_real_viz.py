"""Test if the real visualization can be loaded."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    print("Attempting to import generate_real_3d_viz...")
    from generate_real_3d_viz import create_3d_visualization_from_models
    print("✓ Successfully imported the function")
    
    print("\nTrying to create visualization...")
    fig = create_3d_visualization_from_models()
    print("✓ Successfully created visualization")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
