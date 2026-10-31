"""STEP 3 - general object detection and hand association.

The detector runs on the complete frame, so objects remain visible when no
hand is grabbing them.  A second, enlarged hand crop improves small-object
recall.  Detections are merged, matched between frames, and lightly smoothed
to reduce flicker without hiding newly appearing objects.
"""
import hashlib
import json
import math
import os
from collections import Counter

import cv2
from src.common import F, read_jsonl, write_jsonl, fenv, hand_size

TIPS = [4, 8, 12, 16, 20]
FIXED = {"workbench", "table", "wooden frame", "shelf", "machine"}
MAX_AREA_FRAC = 0.70
DEFAULT_CLASSES = (
    "person,hand,tool,hammer,screwdriver,wrench,pliers,saw,drill,nail gun,"
    "power tool,clamp,tape measure,scissors,knife,bottle,can,cup,box,"
    "piece of wood,wooden board,wooden block,plank,pipe,cable,wire,"
    "screw,nail,bolt,nut,part,container,bag,phone,keyboard"
)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(area(a) + area(b) - inter, 1e-6)


def rect_dist(px, py, b):
    return math.hypot(max(b[0] - px, 0, px - b[2]), max(b[1] - py, 0, py - b[3]))


def tip_stats(hand, bbox, margin=0.25):
    pad = margin * hand_size(hand)
    tips = [hand["kp2d"][i] for i in TIPS]
    count = sum(bbox[0] - pad < x < bbox[2] + pad and bbox[1] - pad < y < bbox[3] + pad
                for x, y in tips)
    return count, min(rect_dist(x, y, bbox) for x, y in tips)


def hand_crop(hand, width, height, scale=3.0, min_size=220):
    x1, y1, x2, y2 = hand["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    size = max(max(x2 - x1, y2 - y1) * scale, min_size)
    return (max(0, int(cx - size / 2)), max(0, int(cy - size / 2)),
            min(width, int(cx + size / 2)), min(height, int(cy + size / 2)))


def _detector():
    """Use COCO for reliable general labels; set OBJ_CLASSES for YOLO-World."""
    classes = [x.strip() for x in os.environ.get("OBJ_CLASSES", DEFAULT_CLASSES).split(",") if x.strip()]
    model_name = os.environ.get("OBJ_MODEL", "yolo11s.pt")
    if "world" in model_name.lower() or os.environ.get("OBJ_CLASSES"):
        from ultralytics import YOLOWorld
        model = YOLOWorld(model_name)
        model.set_classes(classes)
        return model
    from ultralytics import YOLO
    return YOLO(model_name)


def _predict(model, image, offset=(0, 0), conf=0.12):
    result = model.predict(image, conf=conf, imgsz=960, verbose=False)[0]
    names = result.names
    out = []
    for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                               result.boxes.conf.cpu().numpy(),
                               result.boxes.cls.cpu().numpy()):
        b = [float(box[0] + offset[0]), float(box[1] + offset[1]),
             float(box[2] + offset[0]), float(box[3] + offset[1])]
        if area(b) > 0:
            out.append({"name": str(names[int(cls)]).lower(), "conf": float(score), "bbox": b})
    return out


def _merge(detections):
    merged = []
    for item in sorted(detections, key=lambda x: x["conf"], reverse=True):
        duplicate = next((x for x in merged if x["name"] == item["name"] and iou(x["bbox"], item["bbox"]) >= 0.45), None)
        if duplicate:
            if item["conf"] > duplicate["conf"]:
                duplicate.update(item)
        else:
            merged.append(item)
    return merged


def _stable(current, previous, next_id):
    """Match same-label boxes and apply EMA only to an existing track."""
    used = set()
    result = []
    for obj in current:
        best = None
        for index, old in enumerate(previous):
            if index not in used and old["name"] == obj["name"]:
                score = iou(obj["bbox"], old["bbox"])
                if score >= 0.08 and (best is None or score > best[0]):
                    best = (score, index, old)
        if best:
            _, index, old = best
            used.add(index)
            alpha = 0.55 if obj["conf"] >= old["conf"] else 0.35
            obj["bbox"] = [alpha * a + (1 - alpha) * b for a, b in zip(obj["bbox"], old["bbox"])]
            obj["track_id"] = old["track_id"]
            obj["stable"] = True
        else:
            obj["track_id"] = next_id
            next_id += 1
            obj["stable"] = False
        result.append(obj)
    return result, next_id


def nearest(hand, objects, frame_area):
    best = None
    for obj in objects:
        if area(obj["bbox"]) > MAX_AREA_FRAC * frame_area:
            continue
        tips, distance = tip_stats(hand, obj["bbox"])
        if tips == 0 and distance > 1.8 * hand_size(hand):
            continue
        candidate = (tips, -distance, obj["conf"])
        if best is None or candidate > best[0]:
            best = (candidate, {"name": obj["name"], "bbox": obj["bbox"],
                                "conf": round(obj["conf"], 4), "tips": tips,
                                "dist": round(distance, 1), "track_id": obj.get("track_id")})
    return best[1] if best else None


def touched_surface(hand, objects):
    candidates = []
    for obj in objects:
        if obj["name"] in FIXED:
            tips, _ = tip_stats(hand, obj["bbox"])
            if tips >= 2:
                candidates.append(obj)
    if not candidates:
        return None
    obj = min(candidates, key=lambda x: area(x["bbox"]))
    return {"name": obj["name"], "bbox": obj["bbox"], "conf": obj["conf"]}


def run(clip, out_dir, every=1):
    conf = fenv("OBJ_CONF", 0.12)
    rows = read_jsonl(out_dir / F["grip"])
    cap = cv2.VideoCapture(str(clip))
    width, height = int(cap.get(3)), int(cap.get(4))
    signature = hashlib.md5(json.dumps([r["frame"] for r in rows]).encode()).hexdigest()[:8]
    cache = out_dir / f"3_detections_{signature}.json"
    detections = json.load(open(cache)) if cache.exists() else None
    if detections is not None and len(detections) != len(rows):
        detections = None

    if detections is None:
        model = _detector()
        frame_to_row = {r["frame"]: i for i, r in enumerate(rows)}
        detections, previous, next_id, frame_index = [[] for _ in rows], [], 1, 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            row_index = frame_to_row.get(frame_index)
            if row_index is not None:
                raw = _predict(model, image, conf=conf)
                for hand in rows[row_index]["hands"]:
                    x1, y1, x2, y2 = hand_crop(hand, width, height)
                    crop = image[y1:y2, x1:x2]
                    if crop.size:
                        raw.extend(_predict(model, crop, (x1, y1), conf=max(0.08, conf * 0.75)))
                current = _merge(raw)
                current, next_id = _stable(current, previous, next_id)
                previous = current
                detections[row_index] = current
                if row_index % 100 == 0:
                    print(f"row {row_index}/{len(rows)}", flush=True)
            frame_index += 1
        json.dump(detections, open(cache, "w"), indent=2)
    cap.release()

    for row, objects in zip(rows, detections):
        row["objects"] = objects
        for hand in row["hands"]:
            hand["near"] = nearest(hand, objects, width * height)
            hand["surface"] = touched_surface(hand, objects)
    write_jsonl(out_dir / F["objects"], rows)
    counts = Counter(obj["name"] for objects in detections for obj in objects)
    print("visible objects:", dict(counts.most_common(20)) or "none")
