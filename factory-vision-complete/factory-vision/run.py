"""Compatibility launcher for the complete pipeline in ../nilesh.

Examples:
  python run.py all ..\nilesh\egohack-data\clips_480p\clip.mp4 --no-gemini
  python run.py objects ..\nilesh\egohack-data\clips_480p\clip.mp4
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT.parent / "nilesh" / "run.py"

if not PIPELINE.exists():
    raise SystemExit(f"Pipeline entry point not found: {PIPELINE}")

command = [sys.executable, str(PIPELINE), *sys.argv[1:]]
raise SystemExit(subprocess.run(command, cwd=PIPELINE.parent).returncode)
