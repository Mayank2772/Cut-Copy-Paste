"""Usage:
  python run.py hands    egohack-data/clips_480p/factory_001.mp4 [--step 2]
  python run.py timeline egohack-data/clips_480p/factory_001.mp4
"""
import argparse, json
from pathlib import Path
import cv2
from src.hands import run_hands
from src.timeline import get_timeline, clean
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["hands", "timeline"])
ap.add_argument("clip")
ap.add_argument("--step", type=int, default=2)
a = ap.parse_args()
out = Path("outputs") / Path(a.clip).stem

if a.cmd == "hands":
    print(json.dumps(run_hands(a.clip, out, step=a.step), indent=2))
else:
    cap = cv2.VideoCapture(a.clip)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    acts = clean(get_timeline(a.clip, out), dur)
    df = pd.DataFrame(acts).assign(clip=Path(a.clip).name)
    df.to_csv(out / "actions.csv", index=False)
    print(df)
