"""Draw hands, grab/contact state and the held object's box onto a clip. Output stays local (git-ignored)."""
import json, os, shutil, subprocess
from pathlib import Path
import cv2

CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
COL = {"left": (0, 200, 0), "right": (0, 120, 255)}


def draw(img, last, i):
    """Draw one row of results (hands_labeled.jsonl line) on a frame, in place."""
    h_, w_ = img.shape[:2]
    hands = (last or {}).get("hands", [])
    if os.environ.get("SHOW_ALL_OBJECTS"):   # debug: every detection, thin white
        for o in (last or {}).get("objects", []):
            x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    for hd in hands:                          # normal: only the object being held/touched
        if hd.get("object_bbox"):
            x1, y1, x2, y2 = [int(v) for v in hd["object_bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2 if hd.get("bbox_est") else 3)
            cv2.putText(img, str(hd.get("object")), (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for hd in hands:
        kp = [(int(x), int(y)) for x, y in hd["kp2d"]]
        grab, contact = hd.get("grasp"), hd.get("contact")
        hot = bool(grab or contact == "portable")
        c = COL[hd["side"]]
        if hd.get("interp") or hd.get("rescued") == "roi":
            c = tuple(int(v * 0.6) for v in c)   # dimmer = estimated from context, not directly detected
        for a, b in CONN:
            cv2.line(img, kp[a], kp[b], c, 4 if hot else 2)
        for p in kp:
            cv2.circle(img, p, 3, (0, 255, 255) if hot else (255, 255, 255), -1)
        label = hd["side"].upper()
        if contact == "portable":
            label += " HOLDING " + str(hd.get("object"))
        elif contact == "fixed":
            label += " ON " + str(hd.get("object"))
        elif grab:
            label += " GRAB"
        cv2.putText(img, label, (kp[0][0], kp[0][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255) if hot else c, 2)
    if not hands:
        cv2.rectangle(img, (0, 0), (w_ - 1, h_ - 1), (0, 0, 255), 6)   # red border = no hands found
    cv2.putText(img, f"frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def render(clip, out_dir):
    clip, out_dir = Path(clip), Path(out_dir)
    src = out_dir / "hands_labeled.jsonl"
    if not src.exists():
        src = out_dir / "hands.jsonl"
    rows = {r["frame"]: r for r in map(json.loads, open(src))}
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    # Intra-only MJPG: no inter-frame prediction, so fast motion cannot smear/corrupt the overlay video.
    avi = out_dir / "overlay_raw.avi"
    out = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    try:
        out.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    except Exception:
        pass
    i, last = 0, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in rows:
            last = rows[i]
        out.write(draw(img, last, i))
        i += 1
    out.release(); cap.release()
    mp4 = out_dir / "overlay.mp4"
    ff = shutil.which("ffmpeg")
    if ff:   # high-quality H.264 that plays everywhere
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(avi), "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", str(mp4)])
        if r.returncode == 0:
            avi.unlink()
            return mp4
    if mp4.exists():
        mp4.unlink()
    return avi
