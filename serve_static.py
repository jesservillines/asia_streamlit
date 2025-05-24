"""Simple HTTP server for the static site."""
import http.server
import socketserver
import os

os.chdir("static_presentation")
PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server running at http://localhost:{PORT}")
    httpd.serve_forever()
