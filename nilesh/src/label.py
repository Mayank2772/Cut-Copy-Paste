"""Measure your own accuracy on ~100 random frames.   python -m src.label CLIP [--n 100]
For each frame the tracking is drawn on it. Answer with keys (window must be focused):
  A) Are ALL visible hands detected, with no ghost skeletons?   y = yes   n = no   0 = no hand in view
  B) (if hands found) Are the joints on the right places AND left/right correct?   y / n
  C) (if hands found) Is the grab/object label correct?   y / n / s = nothing to judge
  u = undo last frame      q = save and quit
Results: outputs/<clip>/labels.csv  (resumes where you stopped).   Report only:  python -m src.label CLIP --report"""
import csv, json, random, sys
from pathlib import Path
import cv2
from src.render import draw

FIELDS = ["frame", "t", "A", "B", "C"]


def load(path):
    if not Path(path).exists():
        return []
    return list(csv.DictReader(open(path)))


def summary(rows):
    def rate(key, good="y"):
        v = [r[key] for r in rows if r[key] in ("y", "n")]
        return (len(v), sum(x == good for x in v) / len(v)) if v else (0, None)
    nA, a = rate("A")
    nB, b = rate("B")
    nC, c = rate("C")
    visible = [r for r in rows if r["A"] in ("y", "n")]
    return {"frames_labelled": len(rows), "frames_with_hands_in_view": len(visible),
            "all_hands_found": None if a is None else round(a, 3),
            "joints_and_sides_correct": None if b is None else round(b, 3),
            "grab_object_correct": None if c is None else round(c, 3),
            "n_for_each": [nA, nB, nC]}


def pick_frames(n_rows, n, seed=0):
    idx = list(range(n_rows))
    random.Random(seed).shuffle(idx)
    return idx[:n]


def main():
    clip = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
    out_dir = Path("outputs") / Path(clip).stem
    src = out_dir / "hands_labeled.jsonl"
    if not src.exists():
        src = out_dir / "hands.jsonl"
    rows = [json.loads(l) for l in open(src)]
    path = out_dir / "labels.csv"
    done = load(path)
    if "--report" in sys.argv:
        print(json.dumps(summary(done), indent=2)); return
    todo = [k for k in pick_frames(len(rows), n) if str(rows[k]["frame"]) not in {d["frame"] for d in done}]
    cap = cv2.VideoCapture(clip)
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

    for n_done, k in enumerate(todo):
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
            c = ask(img, "C) grab/object label correct?  y / n / s=nothing to judge", "yns")
            if c == "q": break
        done.append({"frame": r["frame"], "t": round(r["t"], 2), "A": a, "B": b, "C": c})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(done)
    cv2.destroyAllWindows()
    print(json.dumps(summary(done), indent=2))


if __name__ == "__main__":
    main()
