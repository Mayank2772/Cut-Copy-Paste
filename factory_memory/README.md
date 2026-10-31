# Factory Memory

**Turn raw factory videos into searchable, timestamped hand–object interaction events.**

Factory Memory processes first-person factory footage, detects hands and nearby objects, estimates contact / pickup / release events, and lets you **search those events and jump straight to the moment in the video**.

> Prototype / hackathon project. Events are detections and heuristics — validate important claims against the video.

---

## What you get

| Capability | Implementation |
|---|---|
| Hand tracking | MediaPipe Hand Landmarker + recovery + smoothing |
| Object detection | YOLO-World on expanded hand crops + workspace crop |
| Contact / grab / release | State machine over fingertip–box proximity |
| Searchable events | `events.json` + `events.csv` |
| UI | Search, filter, click → video jumps to timestamp |

---

## Pipeline

```text
FACTORY VIDEO
      │
      ▼
1. track     MediaPipe hands          → 1_hands.jsonl
2. grip      left/right, gaps         → 2_hands_grip.jsonl
3. objects   YOLO-World near hands    → 3_hands_objects.jsonl
4. contacts  grab / hold / release    → contacts.csv + 5_hands_final.jsonl
5. events    Factory Memory records   → events.json
6. render    overlay video            → output_v/<clip>/overlay.mp4
```

---

## Quick start

```bash
# Environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: Gemini for better object names
export GEMINI_API_KEY="..."

# Place footage (never commit it)
# egohack-data/clips_480p/<clip>.mp4
# hand_landmarker.task   (MediaPipe model)
# yolov8s-worldv2.pt     (YOLO-World weights)

# Full pipeline
python run.py all egohack-data/clips_480p/<clip>.mp4

# Or step by step
python run.py track    egohack-data/clips_480p/<clip>.mp4
python run.py grip     egohack-data/clips_480p/<clip>.mp4
python run.py objects  egohack-data/clips_480p/<clip>.mp4
python run.py contacts egohack-data/clips_480p/<clip>.mp4
python run.py events   egohack-data/clips_480p/<clip>.mp4
python run.py render   egohack-data/clips_480p/<clip>.mp4
```

### Open the Factory Memory UI

```bash
python run.py serve
# → http://127.0.0.1:8765/frontend/
```

1. Enter the clip folder name (e.g. `MTL-SGF00106-023`).
2. Click **Load events**.
3. Set the video path if needed.
4. Search for `screwdriver`, `right hand`, `pickup`, `release`, …
5. Click a row — the video jumps to that timestamp.

---

## Event record format

```json
{
  "event_id": "evt_001",
  "timestamp_start": 12.2,
  "timestamp_end": 16.0,
  "hand": "right",
  "object": "power sander",
  "event": "possible_pickup",
  "confidence": 0.89,
  "confidence_label": "High",
  "source": "detector",
  "clip": "MTL-SGF00106-023.mp4"
}
```

Event types: `possible_pickup`, `hold`, `possible_release`, `approach`, `contact`, `tracking_uncertain`.

---

## Project layout

```text
nilesh/   (this folder)
├── run.py                 entry point
├── src/
│   ├── tracking.py        hands
│   ├── grip.py
│   ├── objects.py         nearby + contact objects
│   ├── states.py          grab/hold/release state machine
│   ├── events.py          → events.json  (Factory Memory)
│   ├── naming.py          optional Gemini names
│   ├── render.py          overlay video
│   ├── timeline.py        PS3 actions (optional)
│   ├── label.py           accuracy helper
│   └── common.py
├── frontend/
│   ├── index.html
│   └── app.js
├── outputs/<clip>/        data only (json/csv)
├── output_v/<clip>/       overlay videos
├── requirements.txt
└── README.md
```

---

## Evaluation ideas

- Hand tracking: % of frames with correct left/right on a 100-frame sample.
- Events: precision of `possible_pickup` / `possible_release` vs manual review.
- Search: does “screwdriver” return the expected timestamps?
- Runtime: frames/sec on laptop CPU.

---

## Data rules

- Use footage only as authorized for the hackathon.
- Do not commit videos, frames, or screenshots.
- Keep `egohack-data/` and model weights out of Git (`.gitignore`).
- Delete temporary uploads when the event ends.

---

## License / team

Add team details and license only if distribution is permitted. Dataset rights remain with the data provider.
