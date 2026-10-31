"""Run the complete factory-vision pipeline.

Usage:
  python src\main.py path\to\clip.mp4 --no-gemini
  python src\main.py                    # processes videos in egohack-data
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "nilesh" / "run.py"
DATA = Path(__file__).resolve().parents[1] / "egohack-data"
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def main():
    extra = sys.argv[1:]
    clips = [Path(extra[0])] if extra and not extra[0].startswith("-") else sorted(
        p for p in DATA.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if not clips:
        raise SystemExit("No video found. Put a video in egohack-data or pass its path.")
    for clip in clips:
        command = [sys.executable, str(PIPELINE), "all", str(clip)]
        if "--no-gemini" in extra:
            command.append("--no-gemini")
        result = subprocess.run(command, cwd=PIPELINE.parent)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
