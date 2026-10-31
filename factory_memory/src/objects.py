"""STEP 3 - objects: detect objects near hands (and in a wider local region).

Improvements over baseline:
  • Larger search region around each hand (not only a tight crop)
  • Optional second pass on a shared "workspace" crop covering both hands
  • Broader, more general class vocabulary so it works on varied factory footage
  • Attaches BOTH the nearest contact object AND a list of nearby objects
  • Lower confidence + area filters tuned for small tools / wood pieces
  • Results cached so re-runs are free

In : outputs/<clip>/2_hands_grip.jsonl
Out: 3_hands_objects.jsonl  (+ 3_detections_*.json cache)

Per hand per frame this adds:
  near     = nearest PORTABLE object under / closest to fingertips
             {name, bbox, conf, tips, dist}
  nearby   = list of portable objects within ~2.5× hand size
             [{name, bbox, conf, dist}, ...]
  surface  = FIXED surface touched by >=2 fingertips {name, bbox}
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter

import cv2
from src.common import F, fenv, hand_size, read_jsonl, write_jsonl

# ---------------------------------------------------------------------------
# Vocabulary – keep it general so the same pipeline works on many clips
# ---------------------------------------------------------------------------
PORTABLE = [
    # wood / material (prefer specific names — avoid ultra-generic labels)
    "wooden plank", "wooden board", "plywood sheet", "wooden panel",
    "wooden piece", "plywood piece", "timber", "wood block",
    # power tools
    "power drill", "cordless drill", "circular saw", "power sander",
    "orbital sander", "angle grinder", "jigsaw", "router", "nail gun",
    # hand tools
    "hammer", "screwdriver", "wrench", "pliers", "hand saw",
    "clamp", "tape measure",
    # small / distinct items
    "marker pen", "pencil", "glue bottle", "sandpaper", "sanding block",
    "paint can", "paint brush",
    "electric cable", "power cord",
    "cardboard box", "box",
    "glove",
]
FIXED = [
    "workbench", "table", "wooden frame", "shelf", "machine",
    "floor", "wall", "cabinet", "bench",
]

TIPS = [4, 8, 12, 16, 20]
MAX_AREA_FRAC = 0.40          # reject boxes that fill most of the frame
NEARBY_HAND_MULT = 2.8        # objects within this × hand-size count as "nearby"
CROP_HAND_MULT = 3.2          # how far around the hand we crop for detection
WORKSPACE_PAD = 1.8           # pad around both hands for the shared workspace crop


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def rect_dist(px, py, b):
    return math.hypot(max(b[0] - px, 0, px - b[2]), max(b[1] - py, 0, py - b[3]))


def tip_stats(hand, bbox, margin=0.30):
    """(fingertips inside padded box, closest fingertip distance in px)."""
    pad = margin * hand_size(hand)
    tips = [hand["kp2d"][i] for i in TIPS]
    n = sum(
        bbox[0] - pad < x < bbox[2] + pad and bbox[1] - pad < y < bbox[3] + pad
        for x, y in tips
    )
    d = min(rect_dist(x, y, bbox) for x, y in tips)
    return n, d


def hand_crop(hand, W, H, mult=CROP_HAND_MULT):
    """Expanded axis-aligned crop around a hand."""
    x1, y1, x2, y2 = hand["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 40) * mult
    a = max(0, int(cx - side / 2))
    b = max(0, int(cy - side / 2))
    c = min(W, int(cx + side / 2))
    d = min(H, int(cy + side / 2))
    return a, b, c, d


def workspace_crop(hands, W, H, pad=WORKSPACE_PAD):
    """Single crop covering all hands with extra padding."""
    if not hands:
        return 0, 0, W, H
    xs, ys = [], []
    for h in hands:
        x1, y1, x2, y2 = h["bbox"]
        xs += [x1, x2]
        ys += [y1, y2]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    w = (max(xs) - min(xs)) * pad + 80
    h = (max(ys) - min(ys)) * pad + 80
    a = max(0, int(cx - w / 2))
    b = max(0, int(cy - h / 2))
    c = min(W, int(cx + w / 2))
    d = min(H, int(cy + h / 2))
    return a, b, c, d


def nearest(hand, objs, frame_area):
    """Best portable object for contact (more tips → closer → smaller → higher conf)."""
    best = None
    for o in objs:
        if o["name"] in FIXED or area(o["bbox"]) > MAX_AREA_FRAC * frame_area:
            continue
        n, d = tip_stats(hand, o["bbox"])
        if n == 0 and d > 1.8 * hand_size(hand):
            continue
        key = (n, -round(d), -area(o["bbox"]), o["conf"])
        if best is None or key > best[0]:
            best = (key, {
                "name": o["name"],
                "bbox": o["bbox"],
                "conf": o["conf"],
                "tips": n,
                "dist": round(d, 1),
            })
    return best[1] if best else None


def nearby_list(hand, objs, frame_area, max_n=6):
    """All portable objects within a generous radius of the hand."""
    hs = hand_size(hand)
    cx = (hand["bbox"][0] + hand["bbox"][2]) / 2
    cy = (hand["bbox"][1] + hand["bbox"][3]) / 2
    out = []
    for o in objs:
        if o["name"] in FIXED or area(o["bbox"]) > MAX_AREA_FRAC * frame_area:
            continue
        ox = (o["bbox"][0] + o["bbox"][2]) / 2
        oy = (o["bbox"][1] + o["bbox"][3]) / 2
        d = math.hypot(ox - cx, oy - cy)
        if d > NEARBY_HAND_MULT * hs:
            continue
        out.append({
            "name": o["name"],
            "bbox": o["bbox"],
            "conf": o["conf"],
            "dist": round(d, 1),
        })
    out.sort(key=lambda x: (x["dist"], -x["conf"]))
    return out[:max_n]


def surface_touch(hand, objs):
    """Fixed surface touched by ≥2 fingertips."""
    best = None
    for o in objs:
        if o["name"] not in FIXED:
            continue
        n, d = tip_stats(hand, o["bbox"], margin=0.15)
        if n >= 2 and (best is None or n > best[0]):
            best = (n, {"name": o["name"], "bbox": o["bbox"]})
    return best[1] if best else None


def attach(rows, dets, frame_area):
    for r, objs in zip(rows, dets):
        for h in r["hands"]:
            h["near"] = nearest(h, objs, frame_area)
            h["nearby"] = nearby_list(h, objs, frame_area)
            h["surface"] = surface_touch(h, objs)


def _nms(boxes, iou_thr=0.55):
    """Simple greedy NMS on list of {bbox, conf, ...}."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda o: -o["conf"])
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [o for o in boxes if _iou(best["bbox"], o["bbox"]) < iou_thr]
    return keep


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = area(a) + area(b) - inter
    return inter / ua if ua > 0 else 0.0


def run(clip, out_dir, conf=None, every=None):
    conf = float(conf if conf is not None else fenv("OBJ_CONF", 0.08))
    every = int(every if every is not None else fenv("OBJ_EVERY", 2))
    rows = read_jsonl(out_dir / F["grip"])
    cap = cv2.VideoCapture(str(clip))
    W = int(cap.get(3))
    H = int(cap.get(4))
    frame_area = W * H

    # Cache key includes conf + class list hash so vocabulary changes invalidate it
    vocab_sig = hashlib.md5(("|".join(PORTABLE + FIXED)).encode()).hexdigest()[:6]
    sig = hashlib.md5(f"{conf}_{every}_{vocab_sig}".encode()).hexdigest()[:8]
    cache = out_dir / f"3_detections_conf{conf}_{sig}.json"
    dets = None
    if cache.exists():
        d = json.load(open(cache))
        if len(d) == len(rows):
            dets = d
            print(f"using cached detections ({cache.name})")

    if dets is None:
        from ultralytics import YOLOWorld
        index = {r["frame"]: k for k, r in enumerate(rows)}
        model = YOLOWorld(os.environ.get("OBJ_MODEL", "yolov8s-worldv2.pt"))
        model.set_classes(PORTABLE + FIXED)
        dets = [None] * len(rows)
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
                    hands = rows[k]["hands"]

                    # 1) Per-hand expanded crops
                    for h in hands:
                        a, b, c, d = hand_crop(h, W, H)
                        crop = img[b:d, a:c]
                        if crop.size == 0:
                            continue
                        res = model.predict(crop, conf=conf, imgsz=640, verbose=False)[0]
                        carea = max((c - a) * (d - b), 1)
                        for bx, sc, cl in zip(
                            res.boxes.xyxy.cpu().numpy(),
                            res.boxes.conf.cpu().numpy(),
                            res.boxes.cls.cpu().numpy(),
                        ):
                            if area(bx) > 0.75 * carea:
                                continue  # fills the crop → background
                            last.append({
                                "name": res.names[int(cl)],
                                "conf": float(sc),
                                "bbox": [
                                    float(bx[0]) + a, float(bx[1]) + b,
                                    float(bx[2]) + a, float(bx[3]) + b,
                                ],
                            })

                    # 2) Shared workspace crop (covers both hands + space between)
                    if hands:
                        a, b, c, d = workspace_crop(hands, W, H)
                        crop = img[b:d, a:c]
                        if crop.size > 0:
                            res = model.predict(crop, conf=conf, imgsz=640, verbose=False)[0]
                            carea = max((c - a) * (d - b), 1)
                            for bx, sc, cl in zip(
                                res.boxes.xyxy.cpu().numpy(),
                                res.boxes.conf.cpu().numpy(),
                                res.boxes.cls.cpu().numpy(),
                            ):
                                if area(bx) > 0.70 * carea:
                                    continue
                                last.append({
                                    "name": res.names[int(cl)],
                                    "conf": float(sc),
                                    "bbox": [
                                        float(bx[0]) + a, float(bx[1]) + b,
                                        float(bx[2]) + a, float(bx[3]) + b,
                                    ],
                                })

                    # Deduplicate heavily overlapping boxes (keep higher conf)
                    last = _nms(last, iou_thr=0.55)

                dets[k] = last
                if k % 200 == 0:
                    print(f"row {k}/{len(rows)}", flush=True)
            i += 1
        dets = [d or [] for d in dets]
        json.dump(dets, open(cache, "w"))
    cap.release()

    counts = Counter(o["name"] for d in dets[::every] for o in d)
    print("detected objects (count over sampled frames):",
          dict(counts.most_common(15)) or "NOTHING")
    attach(rows, dets, frame_area)
    write_jsonl(out_dir / F["objects"], rows)
    n_hand = sum(len(r["hands"]) for r in rows)
    n_near = sum(1 for r in rows for h in r["hands"] if h.get("near") and h["near"]["tips"] >= 1)
    n_nearby = sum(1 for r in rows for h in r["hands"] if h.get("nearby"))
    print(f"hand-frames with object under fingertips: {n_near}/{n_hand}")
    print(f"hand-frames with any nearby object:       {n_nearby}/{n_hand}")
