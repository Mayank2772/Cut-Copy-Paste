# Egocentric Vision Hackathon: PS3 Task-step timeline

Compute path: laptop CPU (clips_480p, every 2nd frame) + Gemini API for meaning.

## Setup
    python -m venv egohack && source egohack/bin/activate
    pip install -r requirements.txt
    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    export GEMINI_API_KEY=...   # never commit keys
    # put data in egohack-data/ (git-ignored)

## Run
    python run.py hands egohack-data/clips_480p/factory_001.mp4
    python run.py timeline egohack-data/clips_480p/factory_001.mp4
    streamlit run app.py

## Libraries
MediaPipe, OpenCV, NumPy, pandas, google-genai (Gemini), Streamlit.

No footage, frames or screenshots are in this repo.
