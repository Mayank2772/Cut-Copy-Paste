"""Usage:
  python run.py hands    CLIP [--step 2] [--clahe]
  python run.py render   CLIP [--clahe]
  python run.py timeline CLIP
"""
import argparse, json
from pathlib import Path
import cv2
import pandas as pd
from src.hands import run_hands
from src.render import render
from src.timeline import get_timeline, clean


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["hands", "render", "timeline"])
ap.add_argument("clip")
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--clahe", action="store_true")
a = ap.parse_args()
stem = Path(a.clip).stem + ("_clahe" if a.clahe else "")
out = Path("outputs") / stem

if a.cmd == "hands":
    print(json.dumps(run_hands(a.clip, out, step=a.step, preprocess=clahe if a.clahe else None), indent=2))
elif a.cmd == "render":
    print("saved", render(a.clip, out))
else:
    out = Path("outputs") / Path(a.clip).stem
    cap = cv2.VideoCapture(a.clip)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    acts = clean(get_timeline(a.clip, out), dur)
    df = pd.DataFrame(acts).assign(clip=Path(a.clip).name)
    df.to_csv(out / "actions.csv", index=False)
    print(df)
