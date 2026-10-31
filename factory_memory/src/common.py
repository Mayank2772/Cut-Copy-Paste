"""Shared paths, file names and tiny helpers. No other file hard-codes a path.

nilesh/
  outputs/<clip>/   json / jsonl / csv only   (the data each step produces)
  output_v/<clip>/  videos only               (what you show the judges)
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the nilesh/ folder
OUT = ROOT / "outputs"
VID = ROOT / "output_v"

# File names inside outputs/<clip>/ ; the number is the pipeline step that writes it.
F = {
    "hands": "1_hands.jsonl",              # step 1  track     : raw hand landmarks per frame
    "hands_stats": "1_hands_stats.json",
    "grip": "2_hands_grip.jsonl",          # step 2  grip      : + fixed left/right, gap fill, bent fingers
    "grip_stats": "2_grip_stats.json",
    "objects": "3_hands_objects.jsonl",    # step 3  objects   : + nearest object / surface per hand
    "names": "4_hold_names_gemini.json",   # step 4  contacts  : Gemini answers per grip (cache)
    "final": "5_hands_final.jsonl",        # step 4  contacts  : + state (approach/grab/hold/release) per frame
    "contacts": "contacts.csv",            # step 4  contacts  : FINAL timeline of states
    "actions_raw": "6_actions_gemini.json",  # PS3 timeline (Gemini answer, cached)
    "actions": "actions.csv",                # PS3 timeline
    "labels": "labels.csv",
    "events": "events.json",                  # Factory Memory searchable events
    "events_csv": "events.csv",                   # Factory Memory events (CSV view)
}


def dirs(clip):
    """(outputs/<clip>, output_v/<clip>) - created if missing."""
    stem = Path(clip).stem
    o, v = OUT / stem, VID / stem
    o.mkdir(parents=True, exist_ok=True)
    v.mkdir(parents=True, exist_ok=True)
    return o, v


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def fenv(name, default):
    return float(os.environ.get(name, default))


def hand_of(row, side):
    return next((h for h in row["hands"] if h["side"] == side), None)


def hand_size(h):
    x1, y1, x2, y2 = h["bbox"]
    return max(x2 - x1, y2 - y1, 1.0)
