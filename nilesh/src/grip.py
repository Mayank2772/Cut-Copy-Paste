"""STEP 2 - grip: fix left/right, fill short tracking gaps, count BENT FINGERS per hand.
In : outputs/<clip>/1_hands.jsonl      Out: 2_hands_grip.jsonl, 2_grip_stats.json

A finger is 'bent' when its tip-to-wrist distance is short compared with knuckle-to-wrist (3D landmarks,
so camera angle does not matter). grip_pose = at least MIN_BENT of the 4 fingers bent (default 3)."""
import json, math
import numpy as np
from src.common import F, read_jsonl, write_jsonl, fenv, hand_of

FINGERS = [("index", 5, 8), ("middle", 9, 12), ("ring", 13, 16), ("pinky", 17, 20)]   # (name, knuckle, tip)
BEND_ON = fenv("BEND_ON", 1.30)      # counted as bent below this ratio
BEND_OFF = fenv("BEND_OFF", 1.50)    # counted as straight again above this (hysteresis stops flicker)
MIN_BENT = int(fenv("MIN_BENT", 3))  # fingers that must be bent for a grip pose
FILL_GAP = fenv("FILL_GAP", 0.4)     # seconds: interpolate a hand across gaps up to this


def finger_ratios(kp):
    k = np.array(kp); w = k[0]
    return [float(np.linalg.norm(k[t] - w) / max(np.linalg.norm(k[m] - w), 1e-6)) for _, m, t in FINGERS]


def side_fix(rows):
    """(flip?, agreement). In head-camera view the left hand should sit left of the right hand."""
    ok = tot = 0
    for r in rows:
        L, R = hand_of(r, "left"), hand_of(r, "right")
        if L and R:
            tot += 1
            ok += L["kp2d"][0][0] < R["kp2d"][0][0]
    if tot < 20:
        return False, None
    return ok / tot < 0.5, ok / tot


def drop_unsupported_rescues(rows, max_dt=0.4, max_speed_frac=0.8):
    """A hand found only by the whole-frame low-confidence retry is kept only if a normally detected hand
    of the same side is close in time AND space (hands cannot teleport). Otherwise it is a false detection."""
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
                         math.hypot(px - wx, py - wy) <= max(0.05 * W, max_speed_frac * W * abs(t - r["t"]))
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
        idx = [k for k, r in enumerate(rows) if hand_of(r, side)]
        for a, b in zip(idx, idx[1:]):
            if b - a > 1 and rows[b]["t"] - rows[a]["t"] <= max_gap:
                ha, hb = hand_of(rows[a], side), hand_of(rows[b], side)
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


def add_finger_state(rows):
    """Adds per hand: finger_ratios, bent_flags, fingers_bent (0-4), grip_pose, speed (px/s)."""
    allr = []
    for side in ("left", "right"):
        hs = [hand_of(r, side) for r in rows]
        R = [finger_ratios(h.get("kp3d") or h["kp2d"]) if h else None for h in hs]
        sm = [smooth([r[i] if r else None for r in R]) for i in range(len(FINGERS))]
        st, prev = [False] * len(FINGERS), None
        for k, h in enumerate(hs):
            if h is None:
                st, prev = [False] * len(FINGERS), None
                continue
            vals = [sm[i][k] for i in range(len(FINGERS))]
            for i, v in enumerate(vals):
                st[i] = (v < BEND_ON) if not st[i] else not (v > BEND_OFF)
            h["finger_ratios"] = [round(v, 2) for v in vals]
            h["bent_flags"] = list(st)
            h["fingers_bent"] = sum(st)
            h["grip_pose"] = sum(st) >= MIN_BENT
            h["speed"] = round(math.dist(h["kp2d"][0], prev[1]) / max(rows[k]["t"] - prev[0], 1e-6), 1) if prev else 0.0
            prev = (rows[k]["t"], h["kp2d"][0])
            allr += vals
    return allr


def process(out_dir):
    rows = read_jsonl(out_dir / F["hands"])
    flip, agree = side_fix(rows)
    if flip:
        for r in rows:
            for h in r["hands"]:
                h["side"] = "right" if h["side"] == "left" else "left"
    n_drop = drop_unsupported_rescues(rows)
    raw = sum(bool(r["hands"]) for r in rows) / max(len(rows), 1)
    n_fill = fill_gaps(rows)
    share = sum(bool(r["hands"]) for r in rows) / max(len(rows), 1)
    ratios = add_finger_state(rows)
    with_hand = [h for r in rows for h in r["hands"]]
    stats = {"sides_flipped": flip, "left_of_right_agreement": agree,
             "hands_found_share_raw": round(raw, 3), "hands_found_share_after_gap_fill": round(share, 3),
             "interpolated_hand_frames": n_fill, "rejected_false_rescues": n_drop,
             "finger_ratio_p10_p50_p90": [round(float(v), 2) for v in np.percentile(ratios, [10, 50, 90])] if ratios else None,
             "share_of_hand_frames_in_grip_pose": round(sum(h["grip_pose"] for h in with_hand) / max(len(with_hand), 1), 3)}
    write_jsonl(out_dir / F["grip"], rows)
    json.dump(stats, open(out_dir / F["grip_stats"], "w"), indent=2)
    return stats
