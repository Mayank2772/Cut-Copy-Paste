"""Measure your own accuracy on ~100 random frames.      python run.py label CLIP [--n 100] [--report]
For each frame the tracking and states are drawn on it. Answer with keys (click the window first):
  A) Are ALL visible hands detected, with no ghost skeletons?   y = yes   n = no   0 = no hand in view
  B) (if hands found) Joints on the right places AND left/right correct?   y / n
  C) (if hands found) State and object name correct (approach/grab/hold/release + what it holds)?   y / n / s = nothing to judge
  u = undo last frame      q = save and quit
Results: outputs/<clip>/labels.csv (resumes where you stopped)."""
import csv, json, random
from pathlib import Path
import cv2
from src.common import F, read_jsonl
from src.render import draw, final_rows

FIELDS = ["frame", "t", "A", "B", "C"]


def load(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def summary(rows):
    def rate(key):
        v = [r[key] for r in rows if r[key] in ("y", "n")]
        return (len(v), sum(x == "y" for x in v) / len(v)) if v else (0, None)
    (nA, a), (nB, b), (nC, c) = rate("A"), rate("B"), rate("C")
    r3 = lambda x: None if x is None else round(x, 3)
    return {"frames_labelled": len(rows), "frames_with_hands_in_view": nA,
            "all_hands_found": r3(a), "joints_and_sides_correct": r3(b), "state_and_object_correct": r3(c),
            "n_for_each": [nA, nB, nC]}


def pick_frames(n_rows, n, seed=0):
    idx = list(range(n_rows))
    random.Random(seed).shuffle(idx)
    return idx[:n]


def main(clip, out_dir, n=100, report=False):
    rows = final_rows(out_dir)
    path = out_dir / F["labels"]
    done = load(path)
    if report:
        print(json.dumps(summary(done), indent=2)); return
    seen = {d["frame"] for d in done}
    todo = [k for k in pick_frames(len(rows), n) if str(rows[k]["frame"]) not in seen]
    cap = cv2.VideoCapture(str(clip))
    print(f"{len(done)} done, {len(todo)} to go. Keys: y / n / 0 / s, u = undo, q = quit")

    def ask(img, text, keys):
        view = img.copy()
        cv2.rectangle(view, (0, view.shape[0] - 34), (view.shape[1], view.shape[0]), (0, 0, 0), -1)
        cv2.putText(view, text, (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("label", view)
        while True:
            k = chr(cv2.waitKey(0) & 0xFF)
            if k in keys or k in "uq":
                return k

    for k in todo:
        r = rows[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(draw(img, r, r["frame"]), None, fx=1.6, fy=1.6)
        a = ask(img, f"[{len(done)+1}/{n}] A) ALL visible hands found, no ghosts?  y / n / 0=no hand in view", "yn0")
        if a == "q": break
        if a == "u":
            if done: done.pop()
            continue
        b = c = ""
        if a in "yn":
            b = ask(img, "B) joints on the right places AND left/right correct?  y / n", "yn")
            if b == "q": break
            c = ask(img, "C) state + object name correct?  y / n / s=nothing to judge", "yns")
            if c == "q": break
        done.append({"frame": r["frame"], "t": round(r["t"], 2), "A": a, "B": b, "C": c})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(done)
    cv2.destroyAllWindows()
    print(json.dumps(summary(done), indent=2))
