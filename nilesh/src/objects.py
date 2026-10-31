"""STEP 3 - objects: find objects around each hand (detector runs on a zoomed crop of each hand) and
measure how close the fingertips are to them.
In : outputs/<clip>/2_hands_grip.jsonl      Out: 3_hands_objects.jsonl  (+ 3_detections_*.json cache)

Per hand per frame this adds:
  near    = nearest PORTABLE object {name, bbox, conf, tips, dist}   tips = fingertips inside/near its box
  surface = FIXED surface touched by >=2 fingertips {name, bbox}"""
import hashlib, json, math, os
from collections import Counter
import cv2
from src.common import F, read_jsonl, write_jsonl, fenv, hand_size

PORTABLE = ["wooden block", "wood block", "wooden piece", "plywood piece", "wooden plank",
            "wooden board", "plywood sheet", "wooden panel", "power drill", "red power tool",
            "power tool", "nail gun", "hammer", "screwdriver", "wrench", "pliers", "saw",
            "sandpaper", "sanding block", "tape measure", "paint can", "paint brush",
            "glue bottle", "clamp", "box", "circular saw", "power sander", "angle grinder",
            "jigsaw", "electric cable", "pencil", "marker pen"]
FIXED = ["workbench", "table", "wooden frame", "shelf", "machine"]
TIPS = [4, 8, 12, 16, 20]
MAX_AREA_FRAC = 0.35     # portable boxes bigger than this share of the frame are background


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def rect_dist(px, py, b):
    return math.hypot(max(b[0] - px, 0, px - b[2]), max(b[1] - py, 0, py - b[3]))


def tip_stats(hand, bbox, margin=0.25):
    """(fingertips inside the box padded by margin x hand size, closest fingertip distance in px)."""
    pad = margin * hand_size(hand)
    tips = [hand["kp2d"][i] for i in TIPS]
    n = sum(bbox[0] - pad < x < bbox[2] + pad and bbox[1] - pad < y < bbox[3] + pad for x, y in tips)
    return n, min(rect_dist(x, y, bbox) for x, y in tips)


def nearest(hand, objs, frame_area):
    """Portable object the fingers are on / closest to. More fingertips first, then closer, then smaller."""
    best = None
    for o in objs:
        if o["name"] in FIXED or area(o["bbox"]) > MAX_AREA_FRAC * frame_area:
            continue
        n, d = tip_stats(hand, o["bbox"])
        if n == 0 and d > 1.5 * hand_size(hand):
            continue
        key = (n, -round(d), -area(o["bbox"]), o["conf"])
        if best is None or key > best[0]:
            best = (key, {"name": o["name"], "bbox": o["bbox"], "conf": o["conf"], "tips": n, "dist": round(d, 1)})
    return best[1] if best else None


def touched_surface(hand, objs):
    best = None
    for o in objs:
        if o["name"] not in FIXED:
            continue
        n, _ = tip_stats(hand, o["bbox"])
        if n >= 2 and (best is None or area(o["bbox"]) < area(best["bbox"])):
            best = {"name": o["name"], "bbox": o["bbox"], "conf": o["conf"]}
    return best


def hand_crop(hand, W, H, scale=3.0, min_size=200):
    """Square-ish crop around a hand so a small held object becomes bigger for the detector."""
    x1, y1, x2, y2 = hand["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sz = max(max(x2 - x1, y2 - y1) * scale, min_size)
    return (max(0, int(cx - sz / 2)), max(0, int(cy - sz / 2)), min(W, int(cx + sz / 2)), min(H, int(cy + sz / 2)))


def attach(rows, dets, frame_area):
    for r, objs in zip(rows, dets):
        for h in r["hands"]:
            h["near"] = nearest(h, objs, frame_area)
            h["surface"] = touched_surface(h, objs)


def run(clip, out_dir, every=2):
    conf = fenv("OBJ_CONF", 0.08)
    rows = read_jsonl(out_dir / F["grip"])
    cap = cv2.VideoCapture(str(clip))
    W, H = int(cap.get(3)), int(cap.get(4))
    sig = hashlib.md5(json.dumps([[[round(c) for c in h["bbox"]] for h in r["hands"]] for r in rows]).encode()).hexdigest()[:8]
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
        dets, last, i = [None] * len(rows), [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            k = index.get(i)
            if k is not None:
                if k % every == 0:
                    last = []
                    for h in rows[k]["hands"]:           # detect on a zoomed crop around each hand
                        a, b, c, d = hand_crop(h, W, H)
                        crop = img[b:d, a:c]
                        if crop.size == 0:
                            continue
                        res = model.predict(crop, conf=conf, imgsz=640, verbose=False)[0]
                        carea = (c - a) * (d - b)
                        for bx, sc, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                              res.boxes.cls.cpu().numpy()):
                            if area(bx) > 0.7 * carea:    # fills the crop: background, not an object
                                continue
                            last.append({"name": res.names[int(cl)], "conf": float(sc),
                                         "bbox": [float(bx[0]) + a, float(bx[1]) + b, float(bx[2]) + a, float(bx[3]) + b]})
                dets[k] = last
                if k % 200 == 0:
                    print(f"row {k}/{len(rows)}", flush=True)
            i += 1
        dets = [d or [] for d in dets]
        json.dump(dets, open(cache, "w"))
    cap.release()
    counts = Counter(o["name"] for d in dets[::every] for o in d)
    print("detected objects (count over sampled frames):", dict(counts.most_common(12)) or "NOTHING")
    attach(rows, dets, W * H)
    write_jsonl(out_dir / F["objects"], rows)
    n_hand = sum(len(r["hands"]) for r in rows)
    n_near = sum(1 for r in rows for h in r["hands"] if h["near"] and h["near"]["tips"] >= 1)
    print(f"hand-frames with an object under the fingertips: {n_near}/{n_hand}")
