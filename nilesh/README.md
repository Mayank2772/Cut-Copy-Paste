# Egocentric Vision Hackathon - hand states from factory video (PS3 + hand layer)

Compute path: laptop CPU (clips_480p, every 2nd frame). Gemini API used for meaning (object names, task steps).

## Folder layout
    nilesh/
      run.py                one entry point:  python run.py STEP CLIP
      hand_landmarker.task  MediaPipe model (git-ignored)
      egohack-data/         the footage (git-ignored, never committed)
      src/
        common.py           paths + file names (the only place they are defined)
        tracking.py   step 1  hands: MediaPipe, recovery near last position, smoothing
        grip.py       step 2  left/right fix, gap fill, bent-finger count
        objects.py    step 3  objects near the hands (YOLO-World on hand crops)
        naming.py     step 4a Gemini names each grip from small hand crops
        states.py     step 4  approach / grab / hold / release / touching_fixed  -> contacts.csv
        render.py     step 5  draws the video
        label.py              accuracy check on frames you judge
        timeline.py           PS3 task-step timeline (Gemini)
      outputs/<clip>/       DATA ONLY (json / jsonl / csv)
        1_hands.jsonl  1_hands_stats.json
        2_hands_grip.jsonl  2_grip_stats.json
        3_hands_objects.jsonl  3_detections_*.json
        4_hold_names_gemini.json  5_hands_final.jsonl
        contacts.csv          <- the final state timeline
        actions.csv  6_actions_gemini.json   (PS3)     labels.csv  (your accuracy check)
      output_v/<clip>/      VIDEOS ONLY
        overlay.mp4

## Run
    export GEMINI_API_KEY="..."        # never commit keys
    python run.py all egohack-data/clips_480p/<clip>.mp4
    python run.py timeline egohack-data/clips_480p/<clip>.mp4
    python run.py label egohack-data/clips_480p/<clip>.mp4

## Rules
No footage, frames, screenshots or videos in the repo (see .gitignore). Demo videos go only to the organisers' Drive folder.
