"""PS1 layer: MediaPipe hand tracking + One-Euro smoothing + stats.
Output: outputs/<clip>/hands.jsonl and hands_stats.json (detection share, FPS)."""
import json, math, time
from pathlib import Path
import cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODEL = "hand_landmarker.task"
MAX_GAP_S = 0.5  # reset smoothing if a hand was missing longer than this


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.05):
        self.mc, self.beta, self.x, self.dx, self.t = min_cutoff, beta, None, 0.0, None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-6); self.t = t
        self.dx += self._a(1.0, dt) * ((x - self.x) / dt - self.dx)
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


def run_hands(clip, out_dir, step=2, swap_sides=True, preprocess=None):
    """swap_sides=True because MediaPipe assumes a mirrored selfie image.
    VERIFY on your output video and flip the default if labels look right."""
    clip, out_dir = Path(clip), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_tracking_confidence=0.4)
    filters, last_seen = {}, {}
    n_frames = n_with_hand = 0
    t0 = time.time()
    with vision.HandLandmarker.create_from_options(opts) as lm, open(out_dir / "hands.jsonl", "w") as out:
        for i, t, img in frames(clip, step):
            if preprocess:
                img = preprocess(img)
            h, w = img.shape[:2]
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(mp_img, int(t * 1000))
            hands, seen = [], set()
            for pts, hd, wpts in zip(res.hand_landmarks, res.handedness, res.hand_world_landmarks):
                side = hd[0].category_name.lower()
                if swap_sides:
                    side = "right" if side == "left" else "left"
                if side in seen:  # two hands got the same label: skip duplicate
                    continue
                seen.add(side)
                if side not in filters or t - last_seen.get(side, -9) > MAX_GAP_S:
                    filters[side] = [(OneEuro(), OneEuro()) for _ in range(21)]
                last_seen[side] = t
                kp = [[fx(p.x * w, t), fy(p.y * h, t)] for p, (fx, fy) in zip(pts, filters[side])]
                xs, ys = zip(*kp)
                hands.append({"side": side, "score": hd[0].score,
                              "bbox": [min(xs), min(ys), max(xs), max(ys)],
                              "kp2d": kp, "kp3d": [[p.x, p.y, p.z] for p in wpts]})
            n_frames += 1
            n_with_hand += bool(hands)
            out.write(json.dumps({"frame": i, "t": t, "hands": hands}) + "\n")  # empty list if none
    secs = time.time() - t0
    stats = {"clip": clip.name, "frames_processed": n_frames,
             "detection_share": n_with_hand / max(n_frames, 1),
             "processing_fps": n_frames / max(secs, 1e-6), "step": step}
    json.dump(stats, open(out_dir / "hands_stats.json", "w"), indent=2)
    return stats
