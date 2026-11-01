"""STEP 3 - objects: accurate hand-centred object detection.

Accuracy stack
--------------
1. Curated YOLO-World vocabulary (object_catalog.py) — no ultra-generic labels
2. Multi-crop: per-hand expanded crop + shared workspace crop
3. Multi-scale predict (imgsz 480 + 640) and merge
4. Cross-crop NMS with class-priority tie-break
5. Temporal smoothing (majority name + averaged conf over a short window)
6. Sticky nearest-object assignment (prefers last stable name per hand)
7. Soft shape/size metadata from the catalogue (never drops a detection)
8. Disk cache keyed by conf + vocab hash

In : outputs/<clip>/2_hands_grip.jsonl
Out: 3_hands_objects.jsonl  (+ 3_detections_*.json cache)

Per hand per frame:
  near     = best contact object {name, bbox, conf, tips, dist}
  nearby   = other portable objects within ~2.6× hand size
  surface  = FIXED surface with ≥2 fingertips inside
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict

import cv2
from src.common import F, fenv, hand_size, read_jsonl, write_jsonl
from src.object_catalog import (
    YOLO_PORTABLE_CLASSES as PORTABLE,
    FIXED,
    CLASS_PRIORITY,
    canonicalize_name,
    enrich_detection,
)

TIPS = [4, 8, 12, 16, 20]
MAX_AREA_FRAC = 0.38
NEARBY_HAND_MULT = 2.6
CROP_HAND_MULT = 3.0
WORKSPACE_PAD = 1.75
IOU_NMS = 0.48
TEMPORAL_WIN = 5          # majority vote over ±2 sampled frames
MIN_BOX_SIDE = 14         # px
IMGSZ_LIST = (480, 640)   # multi-scale


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / (area(a) + area(b) - inter + 1e-6)


def rect_dist(px, py, b):
    return math.hypot(max(b[0] - px, 0, px - b[2]), max(b[1] - py, 0, py - b[3]))


def tip_stats(hand, bbox, margin=0.28):
    pad = margin * max(hand_size(hand), 30)
    tips = []
    kp = hand.get("kp2d") or []
    for i in TIPS:
        if i < len(kp) and kp[i]:
            tips.append(kp[i])
    if not tips:
        return 0, 1e9
    n = sum(
        bbox[0] - pad < x < bbox[2] + pad and bbox[1] - pad < y < bbox[3] + pad
        for x, y in tips
    )
    d = min(rect_dist(x, y, bbox) for x, y in tips)
    return n, d


def hand_crop(hand, W, H, mult=CROP_HAND_MULT):
    hs = max(hand_size(hand), 40)
    cx = (hand["bbox"][0] + hand["bbox"][2]) / 2
    cy = (hand["bbox"][1] + hand["bbox"][3]) / 2
    r = hs * mult
    a = max(0, int(cx - r))
    b = max(0, int(cy - r))
    c = min(W, int(cx + r))
    d = min(H, int(cy + r))
    return a, b, c, d


def workspace_crop(hands, W, H, pad=WORKSPACE_PAD):
    xs, ys = [], []
    for h in hands:
        xs += [h["bbox"][0], h["bbox"][2]]
        ys += [h["bbox"][1], h["bbox"][3]]
        for p in h.get("kp2d") or []:
            if p and len(p) >= 2:
                xs.append(p[0]); ys.append(p[1])
    if not xs:
        return 0, 0, W, H
    span = max(max(xs) - min(xs), max(ys) - min(ys), 80) * pad
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    a = max(0, int(cx - span)); b = max(0, int(cy - span))
    c = min(W, int(cx + span)); d = min(H, int(cy + span))
    return a, b, c, d


# ---------------------------------------------------------------------------
# NMS + temporal
# ---------------------------------------------------------------------------
def nms(dets, iou_thr=IOU_NMS):
    if not dets:
        return []
    order = sorted(
        range(len(dets)),
        key=lambda i: dets[i]["conf"] + 0.03 * CLASS_PRIORITY.get(dets[i]["name"], 0),
        reverse=True,
    )
    keep, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(dets[i])
        bi = dets[i]["bbox"]
        for j in order:
            if j == i or j in suppressed:
                continue
            if iou(bi, dets[j]["bbox"]) >= iou_thr:
                suppressed.add(j)
    return keep


def temporal_smooth(frame_dets, win=TEMPORAL_WIN):
    """Majority-vote class label for boxes that match across a short window."""
    n = len(frame_dets)
    if n == 0:
        return frame_dets
    half = max(1, win // 2)
    out = []
    for i in range(n):
        cur = frame_dets[i] or []
        if not cur:
            out.append([])
            continue
        window = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            window.extend(frame_dets[j] or [])
        smoothed = []
        for d in cur:
            votes = Counter()
            confs = defaultdict(list)
            for w in window:
                if iou(d["bbox"], w["bbox"]) >= 0.35:
                    votes[w["name"]] += 1
                    confs[w["name"]].append(w["conf"])
            d = dict(d)
            if votes:
                name, cnt = votes.most_common(1)[0]
                if cnt >= 2 or len(window) < 3:
                    d["name"] = name
                    if confs[name]:
                        d["conf"] = float(sum(confs[name]) / len(confs[name]))
            smoothed.append(d)
        out.append(nms(smoothed))
    return out


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------
def pick_near(hand, objs, frame_area, prev_name=None):
    """Score portable objects for contact; soft preference for prev_name."""
    hs = max(hand_size(hand), 30)
    best, best_score = None, -1e9
    for o in objs:
        if o["name"] in FIXED:
            continue
        if area(o["bbox"]) > MAX_AREA_FRAC * frame_area:
            continue
        tips, dist = tip_stats(hand, o["bbox"])
        if tips == 0 and dist > hs * 1.35:
            continue
        score = (
            tips * 3.2
            + o["conf"] * 2.2
            + max(0.0, 1.6 - dist / hs)
            + 0.18 * CLASS_PRIORITY.get(o["name"], 0)
        )
        # Prefer tighter boxes slightly (tool vs whole table)
        score += max(0.0, 0.4 - area(o["bbox"]) / (frame_area * 0.15 + 1))
        if prev_name and o["name"] == prev_name:
            score += 0.65
        if score > best_score:
            best_score = score
            best = {
                "name": o["name"],
                "bbox": [round(x, 1) for x in o["bbox"]],
                "conf": round(float(o["conf"]), 3),
                "tips": int(tips),
                "dist": round(float(dist), 1),
            }
    return best


def nearby_list(hand, objs, frame_area, near_name=None, max_n=6):
    hs = max(hand_size(hand), 30)
    out = []
    for o in objs:
        if o["name"] in FIXED:
            continue
        if near_name and o["name"] == near_name:
            continue
        if area(o["bbox"]) > MAX_AREA_FRAC * frame_area:
            continue
        _, dist = tip_stats(hand, o["bbox"])
        if dist > hs * NEARBY_HAND_MULT:
            continue
        out.append({
            "name": o["name"],
            "bbox": [round(x, 1) for x in o["bbox"]],
            "conf": round(float(o["conf"]), 3),
            "dist": round(float(dist), 1),
        })
    out.sort(key=lambda x: (x["dist"], -x["conf"]))
    return out[:max_n]


def surface_touch(hand, objs):
    for o in objs:
        if o["name"] not in FIXED:
            continue
        tips, _ = tip_stats(hand, o["bbox"], margin=0.35)
        if tips >= 2:
            return {
                "name": o["name"],
                "bbox": [round(x, 1) for x in o["bbox"]],
                "conf": round(float(o["conf"]), 3),
            }
    return None


def attach(rows, dets, frame_area):
    prev = {"left": None, "right": None}
    for k, r in enumerate(rows):
        frame_objs = dets[k] if k < len(dets) else []
        for h in r.get("hands") or []:
            side = h.get("side") or "?"
            near = pick_near(h, frame_objs, frame_area, prev_name=prev.get(side))
            h["near"] = near
            h["nearby"] = nearby_list(
                h, frame_objs, frame_area,
                near_name=near["name"] if near else None,
            )
            h["surface"] = surface_touch(h, frame_objs)
            if near and near.get("conf", 0) >= 0.14:
                prev[side] = near["name"]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _predict_crop(model, crop, offset, conf, fill_frac, frame_area):
    """Run multi-scale YOLO on one crop; return list of absolute dets."""
    if crop is None or crop.size == 0:
        return []
    a, b = offset
    carea = crop.shape[0] * crop.shape[1]
    found = []
    for imgsz in IMGSZ_LIST:
        res = model.predict(crop, conf=conf, imgsz=imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        for bx, sc, cl in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
        ):
            bx = list(map(float, bx))
            if area(bx) > fill_frac * carea:
                continue
            abs_box = [bx[0] + a, bx[1] + b, bx[2] + a, bx[3] + b]
            if area(abs_box) > MAX_AREA_FRAC * frame_area:
                continue
            if (abs_box[2] - abs_box[0]) < MIN_BOX_SIDE or (abs_box[3] - abs_box[1]) < MIN_BOX_SIDE:
                continue
            name = canonicalize_name(res.names[int(cl)])
            found.append({"name": name, "conf": float(sc), "bbox": abs_box})
    return found


def run(clip, out_dir, conf=None, every=None):
    from ultralytics import YOLOWorld

    out_dir = __import__("pathlib").Path(out_dir)
    conf = float(conf if conf is not None else fenv("OBJ_CONF", 0.15))
    every = int(every if every is not None else fenv("OBJ_EVERY", 2))

    grip_path = out_dir / F["grip"]
    if not grip_path.exists():
        raise FileNotFoundError(f"Need {grip_path} — run: python run.py grip {clip}")

    rows = read_jsonl(grip_path)
    cap = cv2.VideoCapture(str(clip))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(W * H)

    vocab_sig = hashlib.md5(
        ("|".join(PORTABLE + FIXED) + f"|ms{IMGSZ_LIST}|v4").encode()
    ).hexdigest()[:8]
    cache = out_dir / f"3_detections_conf{conf}_{vocab_sig}.json"

    if cache.exists():
        print(f"loading detection cache {cache.name}")
        with open(cache, encoding="utf-8") as f:
            dets = json.load(f)
        # ensure length
        while len(dets) < len(rows):
            dets.append([])
        dets = dets[: len(rows)]
    else:
        print(f"YOLO-World multi-scale detect  conf={conf} every={every} imgsz={IMGSZ_LIST}")
        model = YOLOWorld(os.environ.get("OBJ_MODEL", "yolov8s-worldv2.pt"))
        model.set_classes(PORTABLE + FIXED)

        index = {r["frame"]: k for k, r in enumerate(rows)}
        raw_by_k = {}
        last = []
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            k = index.get(i)
            if k is not None:
                if k % every == 0:
                    last = []
                    hands = rows[k].get("hands") or []

                    for h in hands:
                        a, b, c, d = hand_crop(h, W, H)
                        last.extend(
                            _predict_crop(model, img[b:d, a:c], (a, b), conf, 0.72, frame_area)
                        )

                    if hands:
                        a, b, c, d = workspace_crop(hands, W, H)
                        last.extend(
                            _predict_crop(model, img[b:d, a:c], (a, b), conf, 0.68, frame_area)
                        )

                    last = nms(last)
                    last = [enrich_detection(o, W, H) for o in last]

                raw_by_k[k] = [dict(x) for x in last]
                if k % 200 == 0:
                    print(f"  row {k}/{len(rows)}", flush=True)
            i += 1
        cap.release()

        # temporal smooth on sampled keys, then carry-forward to all rows
        keys = sorted(raw_by_k.keys())
        sampled = temporal_smooth([raw_by_k[k] for k in keys], win=TEMPORAL_WIN)
        for k, dets_k in zip(keys, sampled):
            raw_by_k[k] = dets_k

        dets = []
        carry = []
        for k in range(len(rows)):
            if k in raw_by_k:
                carry = raw_by_k[k]
            dets.append([dict(x) for x in carry])

        with open(cache, "w", encoding="utf-8") as f:
            json.dump(dets, f)
        print(f"wrote cache {cache.name}")

    # enrich again (cheap) + attach
    dets = [[enrich_detection(o, W, H) for o in frame] for frame in dets]
    counts = Counter(o["name"] for d in dets[:: max(every, 1)] for o in d)
    print("detected objects (sampled):", dict(counts.most_common(12)) or "NOTHING")

    attach(rows, dets, frame_area)
    write_jsonl(out_dir / F["objects"], rows)

    n_hand = sum(len(r.get("hands") or []) for r in rows)
    n_near = sum(
        1 for r in rows for h in r.get("hands") or []
        if h.get("near") and h["near"].get("tips", 0) >= 1
    )
    n_nearby = sum(1 for r in rows for h in r.get("hands") or [] if h.get("nearby"))
    print(f"hand-frames with object under fingertips: {n_near}/{n_hand}")
    print(f"hand-frames with any nearby object:       {n_nearby}/{n_hand}")
    return rows


if __name__ == "__main__":
    import sys
    from src.common import dirs
    if len(sys.argv) < 2:
        print("Usage: python -m src.objects <clip.mp4>")
        raise SystemExit(1)
    clip = sys.argv[1]
    out, _ = dirs(clip)
    run(clip, out)
