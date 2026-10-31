"""Fix left/right labels, fill short tracking gaps, detect grab/release from 3D hand pose.
Reads outputs/<clip>/hands.jsonl; writes hands_labeled.jsonl and contacts.csv."""
import csv, json, os
from pathlib import Path
import numpy as np

TIPS, MCPS = [8, 12, 16, 20], [5, 9, 13, 17]
ENTER = float(os.environ.get("GRAB_ENTER", 1.20))   # grip starts below this curl ratio
EXIT = float(os.environ.get("GRAB_EXIT", 1.42))     # grip ends above this (hysteresis)
PINCH = 0.35
MIN_DUR, MAX_GAP = 0.25, 0.30                        # seconds: drop short blips, merge tiny gaps
FILL_GAP = float(os.environ.get("FILL_GAP", 0.4))   # seconds: interpolate hand over gaps up to this


def curl(kp):
    k = np.array(kp); w = k[0]
    tip = np.linalg.norm(k[TIPS] - w, axis=1)
    mcp = np.linalg.norm(k[MCPS] - w, axis=1)
    return float(np.mean(tip / np.maximum(mcp, 1e-6)))   # open hand ~1.8, fist ~1.0


def pinch(kp):
    k = np.array(kp)
    return float(np.linalg.norm(k[4] - k[8]) / max(np.linalg.norm(k[9] - k[0]), 1e-6))


def side_fix(rows):
    """Return (flip?, agreement). Left hand should sit left of right hand in head-cam view."""
    ok = tot = 0
    for r in rows:
        L = [h for h in r["hands"] if h["side"] == "left"]
        R = [h for h in r["hands"] if h["side"] == "right"]
        if L and R:
            tot += 1
            ok += L[0]["kp2d"][0][0] < R[0]["kp2d"][0][0]
    if tot < 20:
        return False, None
    return ok / tot < 0.5, ok / tot


def drop_unsupported_rescues(rows, max_dt=0.4, max_speed_frac=0.8):
    """A hand found only by the low-confidence rescue pass is kept only if a normally detected hand
    of the same side is close in time AND close in space (a hand cannot teleport). Otherwise it is
    treated as a false detection and removed."""
    W = max([h["kp2d"][0][0] for r in rows for h in r["hands"]] or [640]) or 640
    trusted = {"left": [], "right": []}
    for r in rows:
        for h in r["hands"]:
            if not h.get("rescued"):
                trusted[h["side"]].append((r["t"], h["kp2d"][0]))
    dropped = 0
    for r in rows:
        keep = []
        for h in r["hands"]:
            if h.get("rescued") in (True, "full"):
                wx, wy = h["kp2d"][0]
                ok = any(abs(t - r["t"]) <= max_dt and
                         ((px - wx) ** 2 + (py - wy) ** 2) ** 0.5 <= max(0.05 * W, max_speed_frac * W * abs(t - r["t"]))
                         for t, (px, py) in trusted[h["side"]])
                if not ok:
                    dropped += 1
                    continue
            keep.append(h)
        r["hands"] = keep
    return dropped


def fill_gaps(rows, max_gap=FILL_GAP):
    """Linearly interpolate a hand across short gaps; filled hands are marked interp=True."""
    n = 0
    for side in ("left", "right"):
        idx = [k for k, r in enumerate(rows) if any(h["side"] == side for h in r["hands"])]
        for a, b in zip(idx, idx[1:]):
            if b - a > 1 and rows[b]["t"] - rows[a]["t"] <= max_gap:
                ha = next(h for h in rows[a]["hands"] if h["side"] == side)
                hb = next(h for h in rows[b]["hands"] if h["side"] == side)
                for k in range(a + 1, b):
                    f = (rows[k]["t"] - rows[a]["t"]) / (rows[b]["t"] - rows[a]["t"])
                    lerp = lambda x, y: (np.array(x) * (1 - f) + np.array(y) * f).tolist()
                    new = {"side": side, "score": 0.0, "interp": True,
                           "bbox": lerp(ha["bbox"], hb["bbox"]), "kp2d": lerp(ha["kp2d"], hb["kp2d"])}
                    if ha.get("kp3d") and hb.get("kp3d"):
                        new["kp3d"] = lerp(ha["kp3d"], hb["kp3d"])
                    rows[k]["hands"].append(new)
                    n += 1
    return n


def smooth(vals, k=2):
    out = []
    for i, v in enumerate(vals):
        if v is None:
            out.append(None); continue
        w = [x for x in vals[max(0, i - k): i + k + 1] if x is not None]
        out.append(float(np.median(w)))
    return out


def segments(ts, flags):
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
        if merged and s[0] - merged[-1][1] <= MAX_GAP:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [s for s in merged if s[1] - s[0] >= MIN_DUR]


def detect(out_dir, clip_name):
    out_dir = Path(out_dir)
    rows = [json.loads(l) for l in open(out_dir / "hands.jsonl")]
    flip, agree = side_fix(rows)
    if flip:
        for r in rows:
            for h in r["hands"]:
                h["side"] = "right" if h["side"] == "left" else "left"
    n_dropped = drop_unsupported_rescues(rows)
    raw_share = sum(bool(r["hands"]) for r in rows) / max(len(rows), 1)
    n_fill = fill_gaps(rows)
    share = sum(bool(r["hands"]) for r in rows) / max(len(rows), 1)
    info = {"sides_flipped": flip, "left_of_right_agreement": agree,
            "hands_found_share_raw": round(raw_share, 3), "hands_found_share_after_gap_fill": round(share, 3),
            "interpolated_hand_frames": n_fill, "rejected_false_rescues": n_dropped}
    ts = [r["t"] for r in rows]
    result, allcurl = [], []
    for side in ("left", "right"):
        hs = [next((x for x in r["hands"] if x["side"] == side), None) for r in rows]
        cs = smooth([curl(h.get("kp3d") or h["kp2d"]) if h else None for h in hs])
        ps = smooth([pinch(h.get("kp3d") or h["kp2d"]) if h else None for h in hs])
        state, flags = False, []
        for h, c, p in zip(hs, cs, ps):
            if h is None:
                state = False
            else:
                state = (c < ENTER or p < PINCH) if not state else not (c > EXIT and p > PINCH * 1.3)
                h["grasp"] = state
            flags.append(state)
        allcurl += [c for c in cs if c is not None]
        for s, e in segments(ts, flags):
            result.append({"clip": clip_name, "t_start": round(s, 2), "t_end": round(e, 2),
                           "hand": side, "state": "grasp", "object_name": ""})
    with open(out_dir / "hands_labeled.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    result.sort(key=lambda x: x["t_start"])
    with open(out_dir / "contacts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "t_start", "t_end", "hand", "state", "object_name"])
        w.writeheader(); w.writerows(result)
    if allcurl:
        q = np.percentile(allcurl, [10, 50, 90])
        info["curl_p10_p50_p90"] = [round(float(v), 2) for v in q]
    info["grasps"] = {s: sum(1 for x in result if x["hand"] == s) for s in ("left", "right")}
    info["grasp_seconds"] = round(sum(x["t_end"] - x["t_start"] for x in result), 1)
    return info
