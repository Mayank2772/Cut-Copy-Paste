# Factory Memory

Local video-analysis prototype for hand/object tracking and searchable event timelines.

## 1. Environment (macOS)

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The hand model is required at the project root as `hand_landmarker.task`. Download the MediaPipe Hand Landmarker task model and place it next to `run.py`.

The object-detection stage also expects the YOLO world model configured by `src/objects.py` (commonly `yolov8s-worldv2.pt`). Put the required weights in the configured location before processing.

## 2. Start the application

```bash
source .venv/bin/activate
python run.py serve
```

Open `http://127.0.0.1:8765/frontend/`. Select a normal MP4, MOV, AVI, WebM or M4V video and click **Upload & analyse**. The server saves the uploaded file under `uploads/`, runs the existing `all` pipeline, writes event data under `outputs/<uploaded-clip>/`, and writes the annotated video under `output_v/<uploaded-clip>/`.

The UI polls processing status, loads the rendered video and displays the searchable event timeline. Clicking an event seeks the output video to its start timestamp. The local server includes HTTP byte-range support for browser video seeking.

Processing is local and may take several minutes depending on video length, resolution, hardware and model sizes. Keep the terminal running while analysis is in progress. The upload endpoint currently accepts files up to 2 GB. Jobs are held in server memory, so do not restart the server during a running job.

## 3. Existing command-line workflow

The original CLI remains available:

```bash
python run.py all egohack-data/clips_480p/your-video.mp4 --no-gemini
python run.py serve
```

You can omit `--no-gemini` if you have configured the Gemini API and want Gemini naming. Event names such as possible pickup/release are heuristics derived from the current tracking pipeline; they are not guaranteed ground truth.

## 4. Output locations

- `uploads/`: uploaded source videos
- `outputs/<clip>/events.json`: searchable event records
- `outputs/<clip>/contacts.csv`: contact/state timeline
- `output_v/<clip>/overlay.mp4` or `overlay_raw.avi`: annotated output video

This server binds to `127.0.0.1` and is intended for local use only. Do not expose it to a public network; factory footage may be restricted.
