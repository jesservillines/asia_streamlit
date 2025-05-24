"""Generate a static HTML presentation with pre-rendered visualizations.

This provides the fastest possible loading times by pre-generating everything.
"""
from pathlib import Path
import json
import shutil

# Create HTML template with modern design
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASIA Motor Score Prediction - Interactive Presentation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
        }
        
        /* Navigation */
        nav {
            background: #1a1a2e;
            color: white;
            padding: 1rem 2rem;
            position: fixed;
            width: 100%;
            top: 0;
            z-index: 1000;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        nav ul {
            list-style: none;
            display: flex;
            align-items: center;
            gap: 2rem;
        }
        
        nav a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            transition: opacity 0.3s;
        }
        
        nav a:hover {
            opacity: 0.8;
        }
        
        .nav-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-right: auto;
        }
        
        /* Sections */
        .section {
            display: none;
            padding: 5rem 2rem 2rem;
            max-width: 1400px;
            margin: 0 auto;
            animation: fadeIn 0.3s ease-in;
        }
        
        .section.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Hero Section */
        .hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4rem 2rem;
            border-radius: 1rem;
            margin-bottom: 3rem;
            text-align: center;
        }
        
        .hero h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .hero p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 2rem;
            border-radius: 0.5rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: transform 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            color: #666;
            margin-top: 0.5rem;
        }
        
        /* Visualization Container */
        .viz-container {
            background: white;
            padding: 2rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin: 2rem 0;
        }
        
        .viz-container h2 {
            margin-bottom: 1.5rem;
            color: #1a1a2e;
        }
        
        .viz-iframe {
            width: 100%;
            border: none;
            border-radius: 0.5rem;
        }
        
        /* Loading State */
        .loading {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 2rem auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            nav ul {
                flex-wrap: wrap;
                gap: 1rem;
            }
            
            .hero h1 {
                font-size: 2rem;
            }
            
            .section {
                padding: 4rem 1rem 2rem;
            }
        }
    </style>
</head>
<body>
    <nav>
        <ul>
            <li class="nav-title">🧠 ASIA Prediction</li>
            <li><a href="#" data-section="overview">Overview</a></li>
            <li><a href="#" data-section="3d-viz">3D Outcomes</a></li>
            <li><a href="#" data-section="clinical">Clinical Impact</a></li>
            <li><a href="#" data-section="shap">SHAP Analysis</a></li>
            <li><a href="#" data-section="recovery">Recovery Paths</a></li>
        </ul>
    </nav>
    
    <!-- Overview Section -->
    <section id="overview" class="section active">
        <div class="hero">
            <h1>ASIA Motor Score Prediction</h1>
            <p>🏆 Kaggle Winning Solution - Predicting Spinal Cord Injury Recovery</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">20</div>
                <div class="metric-label">Motor Scores Predicted</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">&lt;0.90</div>
                <div class="metric-label">RMSE Performance</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">3</div>
                <div class="metric-label">Ensemble Models</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">#1</div>
                <div class="metric-label">Competition Rank</div>
            </div>
        </div>
        
        <div class="viz-container">
            <h2>Mission</h2>
            <p>Enable clinicians to have data-driven conversations about recovery trajectories with patients and families.</p>
        </div>
    </section>
    
    <!-- 3D Visualization Section -->
    <section id="3d-viz" class="section">
        <div class="viz-container">
            <h2>Interactive 3D Patient Outcomes</h2>
            <iframe class="viz-iframe" src="visuals/figures/interactive_3d_outcomes.html" height="800" loading="lazy"></iframe>
        </div>
    </section>
    
    <!-- Clinical Impact Section -->
    <section id="clinical" class="section">
        <div class="viz-container">
            <h2>Clinical Decision Support</h2>
            <iframe class="viz-iframe" src="visuals/figures/recovery_timeline.html" height="600" loading="lazy"></iframe>
        </div>
    </section>
    
    <!-- SHAP Section -->
    <section id="shap" class="section">
        <div class="viz-container">
            <h2>Feature Impact Analysis</h2>
            <iframe class="viz-iframe" src="visuals/figures/interactive_shap_explorer.html" height="700" loading="lazy"></iframe>
        </div>
    </section>
    
    <!-- Recovery Paths Section -->
    <section id="recovery" class="section">
        <div class="viz-container">
            <h2>Recovery Trajectories</h2>
            <iframe class="viz-iframe" src="visuals/figures/animated_recovery_paths.html" height="700" loading="lazy"></iframe>
        </div>
    </section>
    
    <script>
        // Simple navigation
        document.addEventListener('DOMContentLoaded', function() {
            const navLinks = document.querySelectorAll('nav a');
            const sections = document.querySelectorAll('.section');
            
            navLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetSection = this.getAttribute('data-section');
                    
                    // Hide all sections
                    sections.forEach(section => {
                        section.classList.remove('active');
                    });
                    
                    // Show target section
                    const target = document.getElementById(targetSection);
                    if (target) {
                        target.classList.add('active');
                    }
                });
            });
        });
    </script>
</body>
</html>"""

def generate_static_site():
    """Generate the static site."""
    project_root = Path(__file__).parent
    pkg_dir = project_root / "asia-impairment-track-prediction"
    
    # Create output directory
    output_dir = project_root / "static_presentation"
    output_dir.mkdir(exist_ok=True)
    
    # Write main HTML
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)
    
    # Copy visualization files
    src_viz_dir = pkg_dir / "visuals" / "figures"
    dst_viz_dir = output_dir / "visuals" / "figures"
    
    if src_viz_dir.exists():
        shutil.copytree(src_viz_dir, dst_viz_dir, dirs_exist_ok=True)
    
    print(f"✅ Static site generated at: {output_dir}")
    print(f"📁 Open {output_dir / 'index.html'} in your browser")
    
    # Generate lightweight server script
    server_script = '''"""Simple HTTP server for the static site."""
import http.server
import socketserver
import os

os.chdir("static_presentation")
PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server running at http://localhost:{PORT}")
    httpd.serve_forever()
'''
    
    with open(project_root / "serve_static.py", "w", encoding="utf-8") as f:
        f.write(server_script)
    
    print(f"🚀 Run 'python serve_static.py' to start the server")

if __name__ == "__main__":
    generate_static_site()
