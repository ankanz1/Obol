#!/usr/bin/env python3
"""Run backend locally with production config."""
import os
import subprocess
import sys

os.environ["PORT"] = "8000"

# Run uvicorn
cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
print(f"Starting backend: {' '.join(cmd)}")
subprocess.run(cmd)