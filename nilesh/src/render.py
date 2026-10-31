"""STEP 5 - render: draw hands, states and the object box onto the clip.
Reads outputs/<clip>/5_hands_final.jsonl ; writes the VIDEO to output_v/<clip>/overlay.mp4 (never to outputs/)."""
import json, os, shutil, subprocess
from pathlib import Path
import cv2
from src.common import F, read_jsonl

CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
COL = {"left": (0, 200, 0), "right": (0, 120, 255)}
STATE_COL = {"grab": (0, 255, 255), "hold": (0, 255, 255), "release": (0, 200, 255),
             "approach": (255, 200, 0), "touching_fixed": (200, 200, 200)}
# Hand Stability Meter colours (BGR)
STAB_COL = {
    "STABLE":   (0, 220, 0),      # green  – low micro-movement
    "MODERATE": (0, 220, 255),    # yellow
    "UNSTABLE": (0, 0, 255),      # red    – high jitter
    "MOVING":   (255, 200, 0),    # cyan   – intentional gross motion
}


def _draw_stability_meter(img, hd, kp):
    """
    Draw a compact stability bar + label next to the wrist.
    Green = stable, yellow = moderate, red = unstable.
    """
    stab = hd.get("stability")
    if not stab:
        return
    status = stab.get("status", "")
    score = stab.get("score")
    colour = STAB_COL.get(status, (180, 180, 180))

    # Anchor just above the wrist landmark
    wx, wy = kp[0]
    bar_w, bar_h = 70, 10
    x0 = max(4, wx - bar_w // 2)
    y0 = max(4, wy - 38)

    # Background
    cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + bar_w + 2, y0 + bar_h + 16), (0, 0, 0), -1)
    # Empty bar
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (40, 40, 40), -1)
    # Filled portion (score 0-100 → width)
    if score is not None:
        fill = int(bar_w * max(0, min(100, score)) / 100)
        cv2.rectangle(img, (x0, y0), (x0 + fill, y0 + bar_h), colour, -1)
    else:
        # MOVING – full cyan bar
        cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), colour, -1)

    # Text: "ST 92" or "MOVING"
    if score is not None:
        txt = f"{status[:3]} {score}"
    else:
        txt = status
    cv2.putText(img, txt, (x0, y0 + bar_h + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def draw(img, last, i):
    """Draw one row of results on a frame, in place."""
    h_, w_ = img.shape[:2]
    hands = (last or {}).get("hands", [])
    for hd in hands:                                   # object box first, so skeletons draw on top
        st = hd.get("state")
        if st and hd.get("object_bbox"):
            x1, y1, x2, y2 = [int(v) for v in hd["object_bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), STATE_COL[st], 2 if hd.get("bbox_est") else 3)
            cv2.putText(img, str(hd.get("object")), (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATE_COL[st], 2)
    for hd in hands:
        kp = [(int(x), int(y)) for x, y in hd["kp2d"]]
        st = hd.get("state")
        hot = st in ("grab", "hold", "release")
        c = COL[hd["side"]]
        if hd.get("interp") or hd.get("rescued") == "roi":
            c = tuple(int(v * 0.6) for v in c)         # dimmer = estimated from context, not directly detected

        # When stability is present, tint the skeleton by stability colour
        stab = hd.get("stability")
        if stab and stab.get("status") in STAB_COL:
            c = STAB_COL[stab["status"]]

        for a, b in CONN:
            cv2.line(img, kp[a], kp[b], c, 4 if hot else 2)
        for p in kp:
            cv2.circle(img, p, 3, (0, 255, 255) if hot else (255, 255, 255), -1)

        # Stability meter (bar + score)
        _draw_stability_meter(img, hd, kp)

        label = hd["side"].upper()
        if st:
            label += f" {st.replace('_', ' ').upper()} {hd.get('object') or ''}"
        cv2.putText(img, label, (kp[0][0], kp[0][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    STATE_COL[st] if st else c, 2)
    if not hands:
        cv2.rectangle(img, (0, 0), (w_ - 1, h_ - 1), (0, 0, 255), 6)   # red border = no hands found
    cv2.putText(img, f"frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def final_rows(out_dir):
    for key in ("final", "objects", "grip", "hands"):
        if (out_dir / F[key]).exists():
            return read_jsonl(out_dir / F[key])
    raise FileNotFoundError("No tracking output found. Run: python run.py track CLIP")


def render(clip, out_dir, vid_dir):
    rows = {r["frame"]: r for r in final_rows(out_dir)}
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    vid_dir.mkdir(parents=True, exist_ok=True)
    # Intra-only MJPG: no inter-frame prediction, so fast motion cannot smear/corrupt the overlay video.
    avi = vid_dir / "overlay_raw.avi"
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
    mp4 = vid_dir / "overlay.mp4"
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
