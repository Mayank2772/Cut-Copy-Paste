"""PS1 layer: MediaPipe hand tracking + context-aware recovery + One-Euro smoothing + stats.
Output: outputs/<clip>/hands.jsonl and hands_stats.json (detection share, FPS)."""
import json, math, os, time
from pathlib import Path
import cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODEL = "hand_landmarker.task"
MAX_GAP_S = 0.5      # reset smoothing if a hand was missing longer than this
ROI_MAX_GAP = float(os.environ.get("ROI_MAX_GAP", 0.8))  # seconds a lost hand is searched for near its predicted spot


class OneEuro:
    """One-Euro filter. Fast motion -> little smoothing (no lag); still hand -> strong smoothing.
    A jump bigger than `snap` pixels restarts the filter instead of dragging the point along."""
    def __init__(self, min_cutoff=1.5, beta=0.3, d_cutoff=5.0, snap=60.0):
        self.mc, self.beta, self.dc, self.snap = min_cutoff, beta, d_cutoff, snap
        self.x, self.dx, self.t = None, 0.0, None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        if self.x is None or abs(x - self.x) > self.snap:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-6); self.t = t
        self.dx += self._a(self.dc, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._a(self.mc + self.beta * abs(self.dx), dt) * (x - self.x)
        return self.x


def frames(path, step=1, width=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            if width:
                img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
            yield i, i / fps, img
        i += 1
    cap.release()


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _bright(img):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=40)


def _mpimg(img):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _update_track(track, side, kp, t, real):
    """Remember where a hand was and how fast it moved, to predict where to look when it is lost."""
    xs, ys = [p[0] for p in kp], [p[1] for p in kp]
    c = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    prev = track.get(side)
    v = (0.0, 0.0)
    if prev and 0 < t - prev["t"] <= 0.5:
        dt = t - prev["t"]
        v = tuple(max(-600.0, min(600.0, (c[i] - prev["c"][i]) / dt)) for i in (0, 1))
    track[side] = {"t": t, "c": c, "v": v, "size": size,
                   "real_t": t if real else (prev or {}).get("real_t", t)}


def _roi_search(lm2, img, tr, t, others):
    """Look for a lost hand only near where context says it should be: crop, enlarge, relaxed detector.
    The detection must be near the predicted wrist and not be the other hand."""
    h, w = img.shape[:2]
    dt = t - tr["t"]
    cx, cy = tr["c"][0] + tr["v"][0] * dt, tr["c"][1] + tr["v"][1] * dt
    sz = max(tr["size"] * 2.0, 160)
    x1, y1 = int(max(0, cx - sz / 2)), int(max(0, cy - sz / 2))
    x2, y2 = int(min(w, cx + sz / 2)), int(min(h, cy + sz / 2))
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    crop = img[y1:y2, x1:x2]
    cw, ch = x2 - x1, y2 - y1
    for fn in (None, _clahe, _bright):
        c2 = crop if fn is None else fn(crop)
        sc = max(1.0, 320.0 / max(cw, ch))
        if sc > 1:
            c2 = cv2.resize(c2, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        r = lm2.detect(_mpimg(c2))
        best = None
        for pts, hd, wpts in zip(r.hand_landmarks, r.handedness, r.hand_world_landmarks):
            wx, wy = x1 + pts[0].x * cw, y1 + pts[0].y * ch
            d = math.hypot(wx - cx, wy - cy)
            if d > sz * 0.5:
                continue                                    # too far from where the hand should be
            if any(math.hypot(wx - o[0], wy - o[1]) < 0.25 * sz for o in others):
                continue                                    # that is the other hand
            if best is None or d < best[0]:
                best = (d, pts, wpts, hd)
        if best:
            _, pts, wpts, hd = best
            return ([[x1 + p.x * cw, y1 + p.y * ch] for p in pts],
                    [[p.x, p.y, p.z] for p in wpts], hd[0].score)
    return None


def run_hands(clip, out_dir, step=2, swap_sides=False, preprocess=None):
    """MediaPipe labels were already correct on this footage (verified by the left-of-right check)."""
    clip, out_dir = Path(clip), Path(out_dir)
    if not clip.exists():
        raise FileNotFoundError(f"Clip not found: {clip}")
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=float(os.environ.get("HAND_TRACK_CONF", 0.6)))  # low value = tracker clings to ghosts
    opts_img = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.25, min_tracking_confidence=0.25)
    filters, last_seen, track = {}, {}, {}
    n_frames = n_any = n_raw = n_full = n_roi = 0
    t0 = time.time()

    def smooth(side, raw, t):
        if side not in filters or t - last_seen.get(side, -9) > MAX_GAP_S:
            filters[side] = [(OneEuro(), OneEuro()) for _ in range(21)]
        last_seen[side] = t
        return [[fx(x, t), fy(y, t)] for (x, y), (fx, fy) in zip(raw, filters[side])]

    def make(side, score, kp, kp3d, rescued):
        xs, ys = zip(*kp)
        return {"side": side, "score": score, "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "kp2d": kp, "kp3d": kp3d, "rescued": rescued}

    with vision.HandLandmarker.create_from_options(opts) as lm, \
         vision.HandLandmarker.create_from_options(opts_img) as lm2, \
         open(out_dir / "hands.jsonl", "w") as out:
        for i, t, img in frames(clip, step):
            if preprocess:
                img = preprocess(img)
            h, w = img.shape[:2]
            res = lm.detect_for_video(_mpimg(img), int(t * 1000))
            full = False
            if not res.hand_landmarks:   # whole-frame rescue: enhanced images, lower confidence
                for fn in (_clahe, _bright):
                    res2 = lm2.detect(_mpimg(fn(img)))
                    if res2.hand_landmarks:
                        res, n_full, full = res2, n_full + 1, True
                        break
            hands, seen = [], set()
            for pts, hd, wpts in zip(res.hand_landmarks, res.handedness, res.hand_world_landmarks):
                side = hd[0].category_name.lower()
                if swap_sides:
                    side = "right" if side == "left" else "left"
                if side in seen:  # two hands got the same label: skip duplicate
                    continue
                seen.add(side)
                kp = smooth(side, [[p.x * w, p.y * h] for p in pts], t)
                hands.append(make(side, hd[0].score, kp, [[p.x, p.y, p.z] for p in wpts],
                                  "full" if full else False))
                _update_track(track, side, kp, t, real=not full)
            # context recovery: a hand seen moments ago should still be near where it was heading
            others = [hh["kp2d"][0] for hh in hands]
            for side in ("left", "right"):
                tr = track.get(side)
                if side in seen or not tr or t - tr["real_t"] > ROI_MAX_GAP:
                    continue
                found = _roi_search(lm2, img, tr, t, others)
                if found:
                    raw, k3, score = found
                    kp = smooth(side, raw, t)
                    hands.append(make(side, score, kp, k3, "roi"))
                    _update_track(track, side, kp, t, real=False)
                    seen.add(side); others.append(kp[0]); n_roi += 1
            n_frames += 1
            n_any += bool(hands)
            n_raw += any(not hh["rescued"] for hh in hands)
            out.write(json.dumps({"frame": i, "t": t, "hands": hands}) + "\n")  # empty list if none
    secs = time.time() - t0
    stats = {"clip": clip.name, "frames_processed": n_frames,
             "detection_share": n_any / max(n_frames, 1),
             "detection_share_without_recovery": n_raw / max(n_frames, 1),
             "processing_fps": n_frames / max(secs, 1e-6), "step": step,
             "rescued_by_retry": n_full, "recovered_near_last_position": n_roi}
    json.dump(stats, open(out_dir / "hands_stats.json", "w"), indent=2)
    return stats
