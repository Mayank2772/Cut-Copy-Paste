"""STEP 4 - contacts: turn finger bends + object proximity (+ Gemini naming) into hand STATES.
In : outputs/<clip>/3_hands_objects.jsonl
Out: outputs/<clip>/contacts.csv  (the final timeline)  and  5_hands_final.jsonl (per-frame states for the video)

A GRIP exists only when BOTH hold:  (1) at least 3 fingers are bent (step 2)  and  (2) an object is there:
the detector sees one under the fingertips, or Gemini confirms from the crops that the hand holds something.
A bent hand with no object is not a grab, so there is no 'unknown object' any more.

States written per hand:
  approach        hand open, closing in on the object it will take (up to 1.5 s before the grip)
  grab            first moments of the grip (fingers close on the object)
  hold            grip continues, object moving with the hand
  release         last moments of the grip (fingers open, letting go)
  touching_fixed  hand on a fixed surface, or on an object that does not move with the hand"""
import csv, math, os
from bisect import bisect_left, bisect_right
from collections import Counter
from statistics import mean
from src.common import F, read_jsonl, write_jsonl, hand_of, hand_size

MIN_GRIP, MAX_GAP = 0.25, 0.30            # a grip needs >= 0.25 s; gaps shorter than 0.30 s are bridged
PHASE_W = 0.30                            # length of the grab / release windows (seconds)
APPROACH_MAX, APPROACH_DIST = 1.5, 1.0    # look back up to 1.5 s; 'close' = within 1 hand size
SURFACE_MIN, SURFACE_GAP = 0.5, 0.4
PRI = {"grab": 5, "release": 5, "hold": 4, "approach": 3, "touching_fixed": 2}


def segments(ts, flags, min_dur, max_gap):
    segs, start, last = [], None, None
    for t, f in zip(ts, flags):
        if f:
            start = t if start is None else start
            last = t
        elif start is not None:
            segs.append([start, last]); start = None
    if start is not None:
        segs.append([start, last])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [s for s in merged if s[1] - s[0] >= min_dur]


def grip_segments(rows, side):
    ts = [r["t"] for r in rows]
    flags = [bool((h := hand_of(r, side)) and h.get("grip_pose")) for r in rows]
    out = []
    for t0, t1 in segments(ts, flags, MIN_GRIP, MAX_GAP):
        k0, k1 = bisect_left(ts, t0), bisect_right(ts, t1) - 1
        out.append({"side": side, "k0": k0, "k1": k1, "t0": t0, "t1": t1})
    return out


def is_static(rows, side, anchors):
    """Wrist travels clearly but the object's box hardly moves -> the hand is touching it, not carrying it."""
    if len(anchors) < 3:
        return False
    w = [hand_of(rows[k], side)["kp2d"][0] for k, _ in anchors]
    c = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in anchors]
    pw = sum(math.dist(a, b) for a, b in zip(w, w[1:]))
    po = sum(math.dist(a, b) for a, b in zip(c, c[1:]))
    return pw >= 60 and po < 0.25 * pw


def evidence(rows, seg):
    side = seg["side"]
    hs = [(k, hand_of(rows[k], side)) for k in range(seg["k0"], seg["k1"] + 1)]
    hs = [(k, h) for k, h in hs if h]
    n = max(len(hs), 1)
    t1 = [(k, h) for k, h in hs if h.get("near") and h["near"]["tips"] >= 1]
    t2 = [(k, h) for k, h in hs if h.get("near") and h["near"]["tips"] >= 2]
    names = Counter()
    for _, h in t1:
        names[h["near"]["name"]] += h["near"]["tips"] * h["near"]["conf"]
    seg["det_name"] = names.most_common(1)[0][0] if names else None
    seg["strict_share"] = len(t2) / n
    seg["loose_share"] = len(t1) / n
    seg["bent_avg"] = mean(h.get("fingers_bent", 0) for _, h in hs) if hs else 0
    seg["anchors"] = [(k, h["near"]["bbox"]) for k, h in t1 if h["near"]["name"] == seg["det_name"]]
    cs = [h["near"]["conf"] for _, h in t1 if h["near"]["name"] == seg["det_name"]]
    seg["det_conf"] = mean(cs) if cs else 0.0
    seg["static"] = is_static(rows, side, seg["anchors"])
    seg["gem"], seg["gem_anchors"] = None, []


GENERIC = {"box", "wooden board", "wooden panel", "plywood sheet", "wooden plank", "wooden piece", "plywood piece"}


def needs_gemini(seg):
    """Only uncertain grips are sent to Gemini (saves requests). GEMINI_MODE=all asks about every grip."""
    if os.environ.get("GEMINI_MODE") == "all":
        return True
    backed = seg["strict_share"] >= 0.25 or seg["loose_share"] >= 0.5
    return (not backed or seg["static"] or seg["det_conf"] < float(os.environ.get("DET_CONF_OK", 0.2)) or seg["strict_share"] < 0.5
            or seg["det_name"] in GENERIC)


def decide(seg):
    """-> (kind, object name, source). kind: hold | touch | drop."""
    from src.naming import clean_name
    backed = seg["strict_share"] >= 0.25 or seg["loose_share"] >= 0.5
    g = seg.get("gem")
    if g:
        name = clean_name(g.get("object"))
        if g.get("holding", True) and name not in ("", "none", "nothing"):
            agree = seg["det_name"] and (seg["det_name"] in name or name in seg["det_name"])
            return "hold", name, "gemini+detector" if agree else "gemini"
        return ("touch", seg["det_name"], "gemini+detector") if seg["det_name"] else ("drop", None, "gemini")
    if backed and not seg["static"]:
        return "hold", seg["det_name"], "detector"
    if backed:
        return "touch", seg["det_name"], "detector"
    return "drop", None, "none"


def approach_item(rows, seg, name, source):
    side, k, start, misses = seg["side"], seg["k0"] - 1, None, 0
    while k >= 0 and rows[k]["t"] >= seg["t0"] - APPROACH_MAX:
        h = hand_of(rows[k], side)
        if h and not h.get("grip_pose") and h.get("near") and h["near"]["dist"] <= APPROACH_DIST * hand_size(h):
            start, misses = rows[k]["t"], 0
        else:
            misses += 1
            if misses > 2:
                break
        k -= 1
    if start is not None and seg["t0"] - start >= 0.2:
        return {"side": side, "state": "approach", "t0": start, "t1": seg["t0"], "name": name, "source": source,
                "seg": seg, "anchors": None}
    return None


def build_items(rows, seg, kind, name, source):
    side, t0, t1 = seg["side"], seg["t0"], seg["t1"]
    anchors = seg["gem_anchors"] if seg["gem_anchors"] else seg["anchors"]
    base = {"side": side, "name": name, "source": source, "seg": seg}
    if kind == "touch":
        return [{**base, "state": "touching_fixed", "t0": t0, "t1": t1, "anchors": None}]
    w = min(PHASE_W, (t1 - t0) / 3)
    items = [{**base, "state": "grab", "t0": t0, "t1": t0 + w, "anchors": anchors},
             {**base, "state": "hold", "t0": t0 + w, "t1": t1 - w, "anchors": anchors},
             {**base, "state": "release", "t0": t1 - w, "t1": t1, "anchors": anchors}]
    ap = approach_item(rows, seg, name, source)
    return items + ([ap] if ap else [])


def box_at(rows, side, k, anchors):
    """Box of the held object at row k. The object stays in the hand, so its box is kept at the same position
    relative to the wrist and blended between the nearest real boxes before and after (real boxes are
    detector hits or Gemini's boxes on the sampled frames). Returns (box, estimated?)."""
    if not anchors:
        return None, False
    anchors = sorted(anchors)
    hk = hand_of(rows[k], side)
    if not hk:
        return None, False
    for j, b in anchors:
        if j == k:
            return b, False

    def rel(j, b):                       # box relative to the wrist at row j
        w = hand_of(rows[j], side)["kp2d"][0]
        return [b[0] - w[0], b[1] - w[1], b[2] - w[0], b[3] - w[1]]
    before = [(j, b) for j, b in anchors if j < k]
    after = [(j, b) for j, b in anchors if j > k]
    if before and after:
        (j0, b0), (j1, b1) = before[-1], after[0]
        f = (k - j0) / (j1 - j0)
        r0, r1 = rel(j0, b0), rel(j1, b1)
        r = [r0[i] * (1 - f) + r1[i] * f for i in range(4)]
    else:
        r = rel(*(before[-1] if before else after[0]))
    w = hk["kp2d"][0]
    return [w[0] + r[0], w[1] + r[1], w[0] + r[2], w[1] + r[3]], True


def label_frames(rows, items):
    ts = [r["t"] for r in rows]
    for it in items:
        pri = PRI[it["state"]]
        for k in range(bisect_left(ts, it["t0"]), bisect_right(ts, it["t1"])):
            h = hand_of(rows[k], it["side"])
            if not h or pri <= h.get("_pri", 0):
                continue
            h["_pri"], h["state"], h["object"] = pri, it["state"], it["name"]
            box, est = box_at(rows, it["side"], k, it.get("anchors"))
            if box is None and it["state"] in ("approach", "touching_fixed"):
                src = h.get("surface") if it.get("surface") else h.get("near")
                if src and (src["name"] == it["name"] or it.get("surface")):
                    box = src["bbox"]
            h["object_bbox"], h["bbox_est"] = box, est


def surface_items(rows):
    ts = [r["t"] for r in rows]
    items = []
    for side in ("left", "right"):
        flags = [bool((h := hand_of(r, side)) and h.get("surface") and not h.get("state")) for r in rows]
        for t0, t1 in segments(ts, flags, SURFACE_MIN, SURFACE_GAP):
            ks = range(bisect_left(ts, t0), bisect_right(ts, t1))
            names = Counter(hand_of(rows[k], side)["surface"]["name"] for k in ks
                            if hand_of(rows[k], side) and hand_of(rows[k], side).get("surface"))
            if names:
                items.append({"side": side, "state": "touching_fixed", "t0": t0, "t1": t1, "name": names.most_common(1)[0][0],
                              "source": "detector", "seg": None, "anchors": None, "surface": True})
    return items


def run(clip, out_dir, use_gemini=None):
    rows = read_jsonl(out_dir / F["objects"])
    if use_gemini is None:
        use_gemini = bool(os.environ.get("GEMINI_API_KEY")) and not os.environ.get("NO_GEMINI")
    segs = [s for side in ("left", "right") for s in grip_segments(rows, side)]
    for s in segs:
        evidence(rows, s)
    gem = None
    if use_gemini and segs:
        from src.naming import name_segments
        ask = [s for s in segs if needs_gemini(s)]
        print(f"{len(ask)} of {len(segs)} grips are uncertain and go to Gemini; the other {len(segs) - len(ask)} keep the detector's name")
        gem = name_segments(clip, rows, ask, out_dir / F["names"])
        print("gemini:", gem)
    elif segs:
        print("Gemini naming OFF: names come from the detector only, grips with no detected object are dropped.")
    items, dropped = [], 0
    for s in segs:
        kind, name, source = decide(s)
        if kind == "drop":
            dropped += 1
            continue
        items += build_items(rows, s, kind, name or "unnamed object", source)
    label_frames(rows, items)
    surf = surface_items(rows)
    label_frames(rows, surf)
    items += surf
    for r in rows:
        for h in r["hands"]:
            h.pop("_pri", None)
    items.sort(key=lambda i: (i["t0"], i["side"]))
    with open(out_dir / F["contacts"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "t_start", "t_end", "hand", "state", "object_name", "source",
                    "fingers_bent_avg", "near_object_share"])
        for it in items:
            s = it["seg"]
            w.writerow([clip.name if hasattr(clip, "name") else os.path.basename(str(clip)),
                        round(it["t0"], 2), round(it["t1"], 2), it["side"], it["state"], it["name"], it["source"],
                        round(s["bent_avg"], 1) if s else "", round(s["strict_share"], 2) if s else ""])
    write_jsonl(out_dir / F["final"], rows)
    cnt = Counter((i["state"], i["name"]) for i in items if i["state"] == "hold")
    print(f"{len(items)} state rows -> {F['contacts']}")
    print("states:", dict(Counter(i["state"] for i in items)))
    print("held objects:", dict(cnt.most_common(10)) or "none")
    if dropped:
        print(f"{dropped} bent-finger moments dropped (no object found nearby and Gemini did not confirm one)")
    return {"state_rows": len(items), "dropped": dropped, "gemini": gem}
