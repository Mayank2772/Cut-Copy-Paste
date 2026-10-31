"""One entry point for the whole project.      python run.py STEP CLIP

STEP      what it does                                              writes
track     step 1  find hands (MediaPipe + recovery + smoothing)     outputs/<clip>/1_hands.jsonl
grip      step 2  fix left/right, fill gaps, count bent fingers     outputs/<clip>/2_hands_grip.jsonl
objects   step 3  find objects near the hands                       outputs/<clip>/3_hands_objects.jsonl
contacts  step 4  states: approach / grab / hold / release          outputs/<clip>/contacts.csv  (+ Gemini names)
render    step 5  draw everything on the video                      output_v/<clip>/overlay.mp4
all       track -> grip -> objects -> contacts -> render
timeline  PS3: Gemini task-step timeline                            outputs/<clip>/actions.csv
label     measure accuracy on ~100 frames you judge by hand         outputs/<clip>/labels.csv

Options:  --every N   process every Nth frame in 'track' (default 2)     --no-gemini   skip Gemini naming
          --n 100 / --report   for 'label'"""
import argparse, json, os, sys
from pathlib import Path
from src.common import dirs, F

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("step", choices=["track", "grip", "objects", "contacts", "render", "all", "timeline", "label"])
ap.add_argument("clip")
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--no-gemini", action="store_true")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--report", action="store_true")
a = ap.parse_args()
if not Path(a.clip).exists():
    sys.exit(f"Clip not found: {a.clip}")
if a.no_gemini:
    os.environ["NO_GEMINI"] = "1"
out, vid = dirs(a.clip)
todo = ["track", "grip", "objects", "contacts", "render"] if a.step == "all" else [a.step]

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
    elif step == "render":
        from src.render import render
        print("saved", render(a.clip, out, vid))
    elif step == "label":
        from src.label import main
        main(a.clip, out, a.n, a.report)
    elif step == "timeline":
        import cv2, pandas as pd
        from src.timeline import get_timeline, clean
        cap = cv2.VideoCapture(a.clip)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        df = pd.DataFrame(clean(get_timeline(a.clip, out), dur)).assign(clip=Path(a.clip).name)
        df.to_csv(out / F["actions"], index=False)
        print(df)
