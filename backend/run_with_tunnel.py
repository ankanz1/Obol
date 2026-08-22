#!/usr/bin/env python3
"""Run backend with Cloudflare tunnel for public access."""
import os
import subprocess
import sys
import time

os.environ["PORT"] = "8000"

# Start backend
print("Starting backend on port 8000...")
backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"])

# Wait for backend to start
time.sleep(5)

# Start cloudflared tunnel
print("Starting Cloudflare tunnel...")
try:
    tunnel = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://localhost:8000"])
    print("Tunnel started! Check the URL above.")
    tunnel.wait()
except FileNotFoundError:
    print("cloudflared not installed. Install from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/")
    backend.terminate()
except KeyboardInterrupt:
    backend.terminate()
    if 'tunnel' in locals():
        tunnel.terminate()