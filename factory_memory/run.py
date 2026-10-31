"""Factory Memory — one entry point.

  python run.py STEP CLIP

STEP
  track      step 1  MediaPipe hands + recovery + smoothing
  grip       step 2  left/right fix, gap fill, bent fingers
  objects    step 3  objects near hands (expanded search)
  contacts   step 4  approach / grab / hold / release  → contacts.csv
  events     step 5  Factory Memory searchable events → events.json
  render     step 6  draw overlay video
  all        track → grip → objects → contacts → events → render
  timeline   PS3 Gemini task-step timeline
  label      manual accuracy check on ~100 frames
  serve      open the Factory Memory UI (simple local server)

Options:
  --every N     process every Nth frame in track (default 2)
  --no-gemini   skip Gemini naming
  --n 100 / --report   for label
"""
import argparse
import json
import os
import sys
from pathlib import Path

from src.common import F, dirs

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument(
    "step",
    choices=[
        "track", "grip", "objects", "contacts", "events", "render",
        "all", "timeline", "label", "serve",
    ],
)
ap.add_argument("clip", nargs="?", default="")
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--no-gemini", action="store_true")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--report", action="store_true")
ap.add_argument("--port", type=int, default=8765)
a = ap.parse_args()

if a.step != "serve":
    if not a.clip:
        sys.exit("CLIP path is required for this step")
    if not Path(a.clip).exists():
        sys.exit(f"Clip not found: {a.clip}")

if a.no_gemini:
    os.environ["NO_GEMINI"] = "1"

if a.step == "serve":
    import http.server
    import socketserver
    import webbrowser
    root = Path(__file__).resolve().parent
    os.chdir(root)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", a.port), handler) as httpd:
        url = f"http://127.0.0.1:{a.port}/frontend/"
        print(f"Factory Memory UI → {url}")
        print("Press Ctrl+C to stop.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()
    sys.exit(0)

out, vid = dirs(a.clip)
todo = (
    ["track", "grip", "objects", "contacts", "events", "render"]
    if a.step == "all"
    else [a.step]
)

for step in todo:
    print(f"\n=== {step} ===", flush=True)
    if step == "track":
        from src.tracking import run_hands
        print(json.dumps(run_hands(a.clip, out, step=a.every), indent=2))
    elif step == "grip":
        from src.grip import process
        print(json.dumps(process(out), indent=2))
    elif step == "objects":
        from src.objects import run
        run(a.clip, out)
    elif step == "contacts":
        from src.states import run
        run(Path(a.clip), out)
    elif step == "events":
        from src.events import build_events
        events = build_events(out, Path(a.clip).name)
        print(json.dumps({"n_events": len(events)}, indent=2))
    elif step == "render":
        from src.render import render
        print("saved", render(a.clip, out, vid))
    elif step == "label":
        from src.label import main
        main(a.clip, out, a.n, a.report)
    elif step == "timeline":
        import cv2
        import pandas as pd
        from src.timeline import clean, get_timeline
        cap = cv2.VideoCapture(a.clip)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        df = pd.DataFrame(clean(get_timeline(a.clip, out), dur)).assign(
            clip=Path(a.clip).name
        )
        df.to_csv(out / F["actions"], index=False)
        print(df)
