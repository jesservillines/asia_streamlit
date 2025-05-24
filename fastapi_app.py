"""FastAPI application for high-performance presentation serving.

Advantages over Streamlit:
- Much faster response times
- Better caching control
- Async operations
- Static file serving
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import asyncio
from typing import Dict, Any

app = FastAPI(title="ASIA Motor Score Prediction")

# Setup paths
PROJECT_ROOT = Path(__file__).parent
PKG_DIR = PROJECT_ROOT / "asia-impairment-track-prediction"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Create directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Cache for pre-loaded visualizations
visualization_cache: Dict[str, str] = {}

@app.on_event("startup")
async def startup_event():
    """Pre-load visualizations on startup."""
    viz_dir = PKG_DIR / "visuals" / "figures"
    if viz_dir.exists():
        for html_file in viz_dir.glob("*.html"):
            key = html_file.stem
            with open(html_file, 'r') as f:
                visualization_cache[key] = f.read()
    print(f"Loaded {len(visualization_cache)} visualizations into cache")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    # Create a simple template if it doesn't exist
    template_path = TEMPLATES_DIR / "index.html"
    if not template_path.exists():
        template_content = """<!DOCTYPE html>
<html>
<head>
    <title>ASIA Motor Score Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .nav { background: #1a1a2e; color: white; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
        .nav a { color: white; text-decoration: none; margin: 0 1rem; }
        .viz-frame { width: 100%; height: 800px; border: none; background: white; border-radius: 8px; }
        .section { margin: 2rem 0; }
        h1 { color: #1a1a2e; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <h1 style="color: white; margin: 0;">🧠 ASIA Motor Score Prediction</h1>
            <div style="margin-top: 1rem;">
                <a href="/viz/interactive_3d_outcomes">3D Outcomes</a>
                <a href="/viz/recovery_timeline">Clinical Impact</a>
                <a href="/viz/interactive_shap_explorer">SHAP Analysis</a>
                <a href="/viz/animated_recovery_paths">Recovery Paths</a>
            </div>
        </div>
        
        <div class="section">
            <h2>Welcome to the ASIA Prediction Dashboard</h2>
            <p>Select a visualization from the navigation above to explore the model insights.</p>
        </div>
    </div>
</body>
</html>"""
        with open(template_path, 'w') as f:
            f.write(template_content)
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/viz/{viz_name}", response_class=HTMLResponse)
async def get_visualization(viz_name: str):
    """Serve cached visualization."""
    if viz_name in visualization_cache:
        return HTMLResponse(content=visualization_cache[viz_name])
    return HTMLResponse(content="<h1>Visualization not found</h1>", status_code=404)

@app.get("/api/metrics")
async def get_metrics():
    """API endpoint for metrics."""
    return {
        "motor_scores": 20,
        "rmse": 0.89,
        "models": ["CatBoost", "XGBoost", "HistGB"],
        "rank": 1
    }

if __name__ == "__main__":
    import uvicorn
    # Run with: uvicorn fastapi_app:app --reload --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
