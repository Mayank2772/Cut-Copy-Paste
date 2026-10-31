"""
STEP - Hand Stability Meter
---------------------------
Measure small unwanted shaking / micro-movement during precision work.

Reads the best available hand landmarks (prefer 5_hands_final.jsonl,
then 2_hands_grip.jsonl / 1_hands.jsonl) and produces:

  stability.csv
  stability_stats.json

Also writes a stability field onto each hand record and saves an
augmented 5_hands_final.jsonl (or the source file) so the renderer
can colour hands by stability.

This is a motion-quality indicator for factory precision work,
not a medical tremor diagnosis.

Status colours (used by render.py):
  STABLE   -> green
  MODERATE -> yellow
  UNSTABLE -> red
  MOVING   -> cyan  (gross intentional movement; stability not scored)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.common import F, read_jsonl, write_jsonl

# ---------------------------------------------------------------------------
# Analysis configuration (engineering defaults – calibrate on your clips)
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 1.0          # rolling window length
MIN_SAMPLES = 8               # need at least this many samples in the window
BASELINE_TAU = 0.30           # low-pass time-constant for intentional motion
STABLE_JITTER = 0.003         # normalised residual below this -> STABLE
UNSTABLE_JITTER = 0.015       # normalised residual above this -> UNSTABLE
MAX_GROSS_SPEED = 1.5         # palm-widths / second; above = intentional move


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def extract_features(hand: dict) -> dict | None:
    """
    Wrist (lm 0), four fingertips relative to wrist, and palm width
    (distance between lm 5 and lm 17).
    """
    kp = np.asarray(hand.get("kp2d"), dtype=float)
    if kp.shape != (21, 2):
        return None
    palm_width = distance(kp[5], kp[17])
    if palm_width < 1.0:
        return None
    wrist = kp[0]
    fingertips = np.asarray([
        kp[8]  - wrist,   # index
        kp[12] - wrist,   # middle
        kp[16] - wrist,   # ring
        kp[20] - wrist,   # pinky
    ])
    return {
        "wrist": wrist,
        "fingers": fingertips,
        "palm_width": palm_width,
    }


def analyse_hand(samples: list) -> dict | None:
    """
    High-frequency residual after subtracting a slowly-changing EMA baseline.
    Normalised by instantaneous palm width so the score is scale-invariant.
    """
    if len(samples) < MIN_SAMPLES:
        return None

    times = np.asarray([s["t"] for s in samples], dtype=float)
    wrists = np.asarray([s["features"]["wrist"] for s in samples])
    fingers = np.asarray([s["features"]["fingers"] for s in samples])
    palm_widths = np.asarray([s["features"]["palm_width"] for s in samples])

    # Normalise by palm size
    norm_wrist = wrists / palm_widths[:, None]
    norm_fingers = fingers / palm_widths[:, None, None]

    # Slow intentional wrist motion (EMA)
    baseline = np.zeros_like(norm_wrist)
    baseline[0] = norm_wrist[0]
    for i in range(1, len(times)):
        dt = max(times[i] - times[i - 1], 1e-6)
        alpha = 1.0 - np.exp(-dt / BASELINE_TAU)
        baseline[i] = alpha * norm_wrist[i] + (1.0 - alpha) * baseline[i - 1]

    wrist_residual = norm_wrist - baseline
    wrist_jitter = float(np.sqrt(np.mean(np.sum(wrist_residual ** 2, axis=1))))

    # Finger residual relative to the same slow baseline
    finger_baseline = np.zeros_like(norm_fingers)
    finger_baseline[0] = norm_fingers[0]
    for i in range(1, len(times)):
        dt = max(times[i] - times[i - 1], 1e-6)
        alpha = 1.0 - np.exp(-dt / BASELINE_TAU)
        finger_baseline[i] = (
            alpha * norm_fingers[i] + (1.0 - alpha) * finger_baseline[i - 1]
        )
    finger_residual = norm_fingers - finger_baseline
    finger_jitter = float(np.sqrt(np.mean(np.sum(finger_residual ** 2, axis=2))))

    # Combined score (wrist dominates)
    jitter = 0.65 * wrist_jitter + 0.35 * finger_jitter

    # Gross speed – do not call intentional movement “unstable”
    dt = np.diff(times)
    if len(dt) == 0:
        return None
    speeds = np.linalg.norm(np.diff(norm_wrist, axis=0), axis=1)
    speeds = speeds / np.maximum(dt, 1e-6)
    gross_speed = float(np.median(speeds))

    if gross_speed > MAX_GROSS_SPEED:
        return {
            "score": None,
            "status": "MOVING",
            "jitter": round(jitter, 5),
            "wrist_jitter": round(wrist_jitter, 5),
            "finger_jitter": round(finger_jitter, 5),
            "gross_speed": round(gross_speed, 4),
        }

    ratio = np.clip(
        (jitter - STABLE_JITTER) / (UNSTABLE_JITTER - STABLE_JITTER),
        0.0, 1.0,
    )
    score = int(round(100 * (1.0 - ratio)))
    if score >= 85:
        status = "STABLE"
    elif score >= 60:
        status = "MODERATE"
    else:
        status = "UNSTABLE"

    return {
        "score": score,
        "status": status,
        "jitter": round(jitter, 5),
        "wrist_jitter": round(wrist_jitter, 5),
        "finger_jitter": round(finger_jitter, 5),
        "gross_speed": round(gross_speed, 4),
    }


def _pick_source(out_dir: Path) -> Path:
    """Prefer the richest hand file that already exists."""
    for key in ("final", "objects", "grip", "hands"):
        p = out_dir / F[key]
        if p.exists() and p.stat().st_size > 0:
            return p
    # Fallback names used in older runs
    for name in ("hands_labeled.jsonl", "hands.jsonl"):
        p = out_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(
        "No hand landmarks found. Run at least:  python run.py track CLIP"
    )


def run_stability(out_dir) -> dict:
    """
    Main entry point.
    out_dir : Path to outputs/<clip>/
    Returns the summary statistics dict.
    """
    out_dir = Path(out_dir)
    source = _pick_source(out_dir)
    rows = read_jsonl(source)

    history = {"left": [], "right": []}
    results = []          # per-frame summary for CSV
    for row in rows:
        t = float(row["t"])
        frame_result = {
            "frame": row["frame"],
            "t": t,
            "left": None,
            "right": None,
        }
        for hand in row.get("hands", []):
            side = hand.get("side")
            if side not in history:
                continue
            features = extract_features(hand)
            if features is None:
                continue
            history[side].append({"t": t, "features": features})
            # Keep only the rolling window
            history[side] = [
                s for s in history[side] if t - s["t"] <= WINDOW_SECONDS
            ]
            measurement = analyse_hand(history[side])
            if measurement is not None:
                frame_result[side] = measurement
                # Attach to the live hand record so render can use it
                hand["stability"] = measurement
        results.append(frame_result)

    # ---- CSV ---------------------------------------------------------------
    csv_path = out_dir / F["stability"]
    fields = [
        "frame", "t", "hand", "score", "status",
        "jitter", "wrist_jitter", "finger_jitter", "gross_speed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for side in ("left", "right"):
                m = result[side]
                if m is None:
                    continue
                writer.writerow({
                    "frame": result["frame"],
                    "t": round(result["t"], 3),
                    "hand": side,
                    **m,
                })

    # ---- summary stats -----------------------------------------------------
    summaries = {}
    for side in ("left", "right"):
        values = [
            r[side]["score"]
            for r in results
            if r[side] is not None and r[side]["score"] is not None
        ]
        summaries[side] = {
            "valid_windows": len(values),
            "average_stability": (
                round(float(np.mean(values)), 1) if values else None
            ),
            "minimum_stability": (int(min(values)) if values else None),
            "unstable_windows": sum(v < 60 for v in values),
        }

    stats = {
        "clip": out_dir.name,
        "window_seconds": WINDOW_SECONDS,
        "source": source.name,
        "left": summaries["left"],
        "right": summaries["right"],
    }
    with open(out_dir / F["stability_stats"], "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Write the augmented hand records back.
    # Prefer writing to the final file so the renderer always sees stability.
    dest = out_dir / F["final"]
    write_jsonl(dest, rows)

    print("Stability analysis complete.")
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.stability <outputs/clip_dir>")
        sys.exit(1)
    run_stability(Path(sys.argv[1]))
