"""Object contact: YOLO-World (open-vocabulary) + fingertip rule.
Run after `hands` and `grab`:   python -m src.objects CLIP
Env: OBJ_CONF (default 0.15). Detections are cached per confidence value."""
import csv, json, sys
from collections import Counter
from pathlib import Path
import cv2

PORTABLE = ["wooden block", "wood block", "wooden piece", "plywood piece", "wooden plank",
            "wooden board", "plywood sheet", "wooden panel", "power drill", "red power tool",
            "power tool", "nail gun", "hammer", "screwdriver", "wrench", "pliers", "saw",
            "sandpaper", "sanding block", "tape measure", "paint can", "paint brush",
            "glue bottle", "clamp", "box", "circular saw", "power sander", "angle grinder",
            "jigsaw", "electric cable"]
FIXED = ["workbench", "table", "wooden frame", "shelf", "machine"]
TIPS = [4, 8, 12, 16, 20]
MIN_DUR, MAX_GAP = 0.25, 0.30
MAX_AREA_FRAC = 0.35   # portable boxes bigger than this share of the frame are background


def near_count(hand, bbox, margin=0.25):
    x1, y1, x2, y2 = hand["bbox"]
    pad = margin * max(x2 - x1, y2 - y1)
    ox1, oy1, ox2, oy2 = bbox
    return sum(ox1 - pad < hand["kp2d"][i][0] < ox2 + pad and
               oy1 - pad < hand["kp2d"][i][1] < oy2 + pad for i in TIPS)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def pick(hand, objs, margin=0.25, min_n=2, frame_area=None):
    """Object touched by >=min_n fingertips. Prefers portable, then more fingertips, then SMALLER box."""
    best = None
    for o in objs:
        fixed = o["name"] in FIXED
        if frame_area and not fixed and area(o["bbox"]) > MAX_AREA_FRAC * frame_area:
            continue
        n = near_count(hand, o["bbox"], margin)
        if n >= min_n:
            key = (not fixed, n, -area(o["bbox"]), o["conf"])
            if best is None or key > best[0]:
                best = (key, {**o, "_n": n})
    return best[1] if best else None


def label_rows(rows, dets, frame_area=None):
    """Sets hand['object'], hand['contact'] (portable|fixed|None), hand['object_bbox']."""
    for r, objs in zip(rows, dets):
        r["objects"] = objs
        for h in r["hands"]:
            o = pick(h, objs, frame_area=frame_area)
            if o and o["name"] not in FIXED and not h.get("grasp") and o["_n"] < 3:
                o = None                      # resting near an object is not holding it
            if o is None and h.get("grasp"):  # gripping: looser match, then unnamed fallback
                o = pick(h, [x for x in objs if x["name"] not in FIXED], margin=0.5, min_n=1,
                         frame_area=frame_area)
                if o is None:
                    o = {"name": "unknown object", "conf": 0.0, "bbox": None}
            h["object"] = o["name"] if o else None
            h["object_bbox"] = o.get("bbox") if o else None
            h["bbox_est"] = False
            h["contact"] = None if not o else ("fixed" if o["name"] in FIXED else "portable")


def _flush(run):
    """One name for the whole hold; every frame keeps a box (moved along with the wrist if estimated)."""
    names = Counter(h["object"] for h in run if h["object"] != "unknown object")
    if not names:
        return
    dom = names.most_common(1)[0][0]
    anchors = [(i, h) for i, h in enumerate(run) if h["object"] == dom and h.get("object_bbox")]
    for i, h in enumerate(run):
        if h["object"] == dom and h.get("object_bbox"):
            continue
        h["object"], h["object_bbox"] = dom, None
        if anchors:
            j, a = min(anchors, key=lambda q: abs(q[0] - i))
            if abs(j - i) <= 8:     # within ~0.5 s of a real detection
                dx = h["kp2d"][0][0] - a["kp2d"][0][0]
                dy = h["kp2d"][0][1] - a["kp2d"][0][1]
                b = a["object_bbox"]
                h["object_bbox"] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
                h["bbox_est"] = True


def hold_runs(rows, side, max_gap=0.4):
    """Consecutive stretches (list of (row_index, hand)) where this hand is 'portable'-in-contact."""
    runs, run, last_t = [], [], None
    for k, r in enumerate(rows):
        h = next((x for x in r["hands"] if x["side"] == side), None)
        if h and h.get("contact") == "portable":
            if run and r["t"] - last_t > max_gap:
                runs.append(run); run = []
            run.append((k, h)); last_t = r["t"]
        elif run and r["t"] - last_t > max_gap:
            runs.append(run); run = []
    if run:
        runs.append(run)
    return runs


def motion_coupling(rows, min_wrist_path=60.0, ratio=0.25):
    """A held object moves WITH the wrist; a plank on the table does not. If the hand travels clearly
    but the object's detected box hardly moves, it is being touched, not held -> contact='fixed'."""
    import math
    n = 0
    for side in ("left", "right"):
        for run in hold_runs(rows, side):
            pts = [(h["kp2d"][0], [(h["object_bbox"][0] + h["object_bbox"][2]) / 2,
                                   (h["object_bbox"][1] + h["object_bbox"][3]) / 2])
                   for _, h in run if h.get("object_bbox") and not h.get("bbox_est")]
            if len(pts) < 3:
                continue
            pw = sum(math.dist(a[0], b[0]) for a, b in zip(pts, pts[1:]))
            po = sum(math.dist(a[1], b[1]) for a, b in zip(pts, pts[1:]))
            if pw >= min_wrist_path and po < ratio * pw:
                for _, h in run:
                    h["contact"] = "fixed"
                n += len(run)
    return n


def relabel_runs(rows, max_gap=0.4):
    """Within one continuous hold, use the most common named object for every frame."""
    for side in ("left", "right"):
        run, last_t = [], None
        for r in rows:
            h = next((x for x in r["hands"] if x["side"] == side), None)
            if h and h.get("contact") == "portable":
                if run and r["t"] - last_t > max_gap:
                    _flush(run); run = []
                run.append(h); last_t = r["t"]
            elif run and r["t"] - last_t > max_gap:
                _flush(run); run = []
        _flush(run)


def segments(ts, labels):
    out, cur, start, last = [], None, None, None
    for t, l in zip(ts, labels):
        if l != cur:
            if cur is not None:
                out.append([cur, start, last])
            cur, start = l, t
        last = t
    if cur is not None:
        out.append([cur, start, last])
    merged = []
    for s in out:
        if merged and merged[-1][0] == s[0] and s[1] - merged[-1][2] <= MAX_GAP:
            merged[-1][2] = s[2]
        else:
            merged.append(s)
    return [s for s in merged if s[2] - s[1] >= MIN_DUR]


def write_contacts(rows, out_dir, clip_name):
    ts = [r["t"] for r in rows]
    result = []
    for side in ("left", "right"):
        labels = []
        for r in rows:
            h = next((x for x in r["hands"] if x["side"] == side), None)
            labels.append((h["contact"], h["object"]) if h and h.get("contact") else None)
        for (contact, name), s, e in segments(ts, labels):
            result.append({"clip": clip_name, "t_start": round(s, 2), "t_end": round(e, 2), "hand": side,
                           "state": "holding" if contact == "portable" else "touching_fixed",
                           "object_name": name})
    result.sort(key=lambda x: x["t_start"])
    with open(Path(out_dir) / "contacts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "t_start", "t_end", "hand", "state", "object_name"])
        w.writeheader(); w.writerows(result)
    return result


def hand_crop(hand, W, H, scale=3.0, min_size=200):
    """Square-ish crop around a hand so a small held object becomes bigger for the detector."""
    x1, y1, x2, y2 = hand["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sz = max(max(x2 - x1, y2 - y1) * scale, min_size)
    return (max(0, int(cx - sz / 2)), max(0, int(cy - sz / 2)),
            min(W, int(cx + sz / 2)), min(H, int(cy + sz / 2)))


def mask_to_box(mask, wrist_xy, frame_area):
    """Mask -> box, rejecting masks that cover the wrist (it is the hand/arm) or are background/specks."""
    import numpy as np
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    wx, wy = int(wrist_xy[0]), int(wrist_xy[1])
    if 0 <= wy < mask.shape[0] and 0 <= wx < mask.shape[1] and mask[wy, wx]:
        return None
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    if not (0.0005 * frame_area < area(box) < MAX_AREA_FRAC * frame_area):
        return None
    return box


def sam_boxes(clip, rows, frame_area, every=3):
    """Fallback: box for 'gripping but nothing detected' using MobileSAM point prompts."""
    import os
    try:
        from ultralytics import SAM
        model = SAM(os.environ.get("SAM_MODEL", "mobile_sam.pt"))
    except Exception as e:
        print("SAM unavailable, skipping fallback boxes:", e)
        return 0
    todo = {}
    for side in ("left", "right"):
        ks = [k for k, r in enumerate(rows) for h in r["hands"]
              if h["side"] == side and h.get("object") == "unknown object" and not h.get("object_bbox")]
        for k in ks[::every]:
            todo.setdefault(rows[k]["frame"], []).append((k, side))
    samples = {}
    cap = cv2.VideoCapture(str(clip))
    i = 0
    while todo and i <= max(todo):
        ok, img = cap.read()
        if not ok:
            break
        for k, side in todo.get(i, []):
            h = next(x for x in rows[k]["hands"] if x["side"] == side)
            kp = h["kp2d"]
            pos = [(kp[4][0] + kp[8][0]) / 2, (kp[4][1] + kp[8][1]) / 2]   # between thumb and index tip
            pts = [pos, kp[0], kp[9], kp[13]]                              # wrist + palm = not the object
            try:
                res = model(img, points=[pts], labels=[[1, 0, 0, 0]], verbose=False)[0]
                mask = res.masks.data[0].cpu().numpy() > 0.5
                samples[(k, side)] = mask_to_box(mask, kp[0], frame_area)
            except Exception:
                samples[(k, side)] = None
        i += 1
    cap.release()
    n = 0
    for side in ("left", "right"):
        ks = sorted(k for (k, s_), v in samples.items() if s_ == side and v)
        for k, r in enumerate(rows):
            h = next((x for x in r["hands"] if x["side"] == side), None)
            if h and h.get("object") == "unknown object" and not h.get("object_bbox") and ks:
                near = min(ks, key=lambda q: abs(q - k))
                if abs(near - k) <= 6:      # within ~0.4 s of a segmented frame
                    h["object_bbox"] = samples[(near, side)]
                    n += 1
    return n


def run(clip, out_dir, every=2):
    import os
    conf = float(os.environ.get("OBJ_CONF", "0.08"))
    clip, out_dir = Path(clip), Path(out_dir)
    src = out_dir / "hands_labeled.jsonl"
    if not src.exists():
        src = out_dir / "hands.jsonl"
    rows = [json.loads(l) for l in open(src)]
    cap = cv2.VideoCapture(str(clip))
    W, H = int(cap.get(3)), int(cap.get(4))
    frame_area = W * H
    import hashlib
    sig = hashlib.md5(json.dumps([[[round(c) for c in h["bbox"]] for h in r["hands"]] for r in rows]).encode()).hexdigest()[:8]
    cache = out_dir / f"dets_crop_conf{conf}_{sig}.json"
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
                    for h in rows[k]["hands"]:          # detect on a zoomed crop around each hand
                        a, b, c, d = hand_crop(h, W, H)
                        crop = img[b:d, a:c]
                        if crop.size == 0:
                            continue
                        res = model.predict(crop, conf=conf, imgsz=640, verbose=False)[0]
                        carea = (c - a) * (d - b)
                        for bx, sc, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                              res.boxes.cls.cpu().numpy()):
                            if area(bx) > 0.7 * carea:   # fills the crop: background, not an object
                                continue
                            last.append({"name": res.names[int(cl)], "conf": float(sc),
                                         "bbox": [float(bx[0]) + a, float(bx[1]) + b,
                                                  float(bx[2]) + a, float(bx[3]) + b]})
                dets[k] = last
                if k % 200 == 0:
                    print(f"row {k}/{len(rows)}", flush=True)
            i += 1
        dets = [d or [] for d in dets]
        json.dump(dets, open(cache, "w"))
    cap.release()
    counts = Counter(o["name"] for d in dets[::every] for o in d)
    print("detected objects (count over sampled frames):", dict(counts.most_common(12)) or "NOTHING")
    label_rows(rows, dets, frame_area)
    relabel_runs(rows)
    print("frames reclassified as touching (object does not move with hand):", motion_coupling(rows))
    if os.environ.get("USE_SAM"):
        print("segmented fallback boxes added:", sam_boxes(clip, rows, frame_area))
    with open(out_dir / "hands_labeled.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = write_contacts(rows, out_dir, clip.name)
    totals = {}
    for x in res:
        key = (x["state"], x["object_name"])
        totals[key] = totals.get(key, 0) + x["t_end"] - x["t_start"]
    print(f"{len(res)} contacts. seconds by (state, object):")
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k[0]:15s} {k[1]:15s} {v:6.1f}s")


if __name__ == "__main__":
    clip = sys.argv[1]
    run(clip, Path("outputs") / Path(clip).stem)
