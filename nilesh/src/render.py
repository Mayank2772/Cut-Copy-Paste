"""Draw hand tracking onto a clip. Output stays local (outputs/ is git-ignored)."""
import json
from pathlib import Path
import cv2

CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
COL = {"left": (0, 200, 0), "right": (0, 120, 255)}


def render(clip, out_dir):
    clip, out_dir = Path(clip), Path(out_dir)
    rows = {r["frame"]: r for r in map(json.loads, open(out_dir / "hands.jsonl"))}
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i, last = 0, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in rows:
            last = rows[i]
        hands = (last or {}).get("hands", [])
        for hd in hands:
            kp = [(int(x), int(y)) for x, y in hd["kp2d"]]
            c = COL[hd["side"]]
            for a, b in CONN:
                cv2.line(img, kp[a], kp[b], c, 2)
            for p in kp:
                cv2.circle(img, p, 3, (255, 255, 255), -1)
            cv2.putText(img, hd["side"], kp[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        if not hands:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(img, f"frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(img)
        i += 1
    out.release(); cap.release()
    return out_dir / "overlay.mp4"
