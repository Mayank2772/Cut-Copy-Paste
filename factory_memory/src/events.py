"""
Factory Memory — Event Extraction
---------------------------------
Converts the contact / state timeline into searchable event records.

Reads:  outputs/<clip>/contacts.csv  (preferred)
        or 5_hands_final.jsonl as fallback

Writes: outputs/<clip>/events.json
        outputs/<clip>/events.csv

Event schema (Factory Memory):
{
  "event_id": "evt_001",
  "timestamp_start": 12.2,
  "timestamp_end": 16.0,
  "hand": "right",
  "object": "screwdriver",
  "event": "possible_pickup" | "contact" | "possible_release" | "hold" | "approach" | "tracking_uncertain",
  "confidence": 0.89,
  "confidence_label": "High" | "Medium" | "Low",
  "source": "detector" | "gemini" | "heuristic",
  "clip": "factory_001.mp4"
}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.common import F, read_jsonl

# Map internal state names → Factory Memory event types
STATE_TO_EVENT = {
    "grab": "possible_pickup",
    "hold": "hold",
    "release": "possible_release",
    "approach": "approach",
    "touching_fixed": "contact",
    "contact": "contact",
}


def _conf_label(score: float | None) -> str:
    if score is None:
        return "Medium"
    if score >= 0.80:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"


def _from_contacts_csv(csv_path: Path, clip_name: str) -> list[dict]:
    events = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            state = (row.get("state") or "contact").strip().lower()
            event_type = STATE_TO_EVENT.get(state, state or "contact")
            try:
                t0 = float(row.get("t_start") or row.get("start_s") or 0)
                t1 = float(row.get("t_end") or row.get("end_s") or t0)
            except ValueError:
                continue
            obj = (row.get("object_name") or row.get("object") or "unknown").strip()
            hand = (row.get("hand") or "unknown").strip().lower()
            source = (row.get("source") or "heuristic").strip()
            # Approximate confidence from near_object_share or fingers if present
            conf = 0.75
            try:
                share = row.get("near_object_share")
                if share not in (None, ""):
                    conf = max(0.4, min(0.98, float(share)))
            except ValueError:
                pass
            events.append({
                "event_id": f"evt_{i:03d}",
                "timestamp_start": round(t0, 2),
                "timestamp_end": round(t1, 2),
                "hand": hand,
                "object": obj,
                "event": event_type,
                "confidence": round(conf, 2),
                "confidence_label": _conf_label(conf),
                "source": source,
                "clip": clip_name,
            })
    return events


def _from_final_jsonl(jsonl_path: Path, clip_name: str) -> list[dict]:
    """Fallback: build contact segments from per-frame state labels."""
    rows = read_jsonl(jsonl_path)
    # Segment continuous (side, state, object) runs
    segments = []
    current = None  # {side, state, object, t0, t1, confs}

    def flush():
        nonlocal current
        if current and current["t1"] > current["t0"]:
            segments.append(current)
        current = None

    for r in rows:
        t = float(r.get("t", 0))
        for h in r.get("hands", []):
            side = h.get("side")
            state = h.get("state")
            if not side or not state:
                continue
            obj = h.get("object") or (h.get("near") or {}).get("name") or "unknown"
            conf = float(h.get("score") or (h.get("near") or {}).get("conf") or 0.7)
            key = (side, state, obj)
            if current and (current["side"], current["state"], current["object"]) == key:
                current["t1"] = t
                current["confs"].append(conf)
            else:
                flush()
                current = {
                    "side": side,
                    "state": state,
                    "object": obj,
                    "t0": t,
                    "t1": t,
                    "confs": [conf],
                }
    flush()

    events = []
    for i, s in enumerate(segments, start=1):
        conf = sum(s["confs"]) / max(len(s["confs"]), 1)
        events.append({
            "event_id": f"evt_{i:03d}",
            "timestamp_start": round(s["t0"], 2),
            "timestamp_end": round(s["t1"], 2),
            "hand": s["side"],
            "object": s["object"],
            "event": STATE_TO_EVENT.get(s["state"], s["state"]),
            "confidence": round(conf, 2),
            "confidence_label": _conf_label(conf),
            "source": "heuristic",
            "clip": clip_name,
        })
    return events


def build_events(out_dir, clip_name: str | None = None) -> list[dict]:
    """
    Build Factory Memory events for one clip folder.
    Returns the list of events and writes events.json + events.csv.
    """
    out_dir = Path(out_dir)
    clip_name = clip_name or out_dir.name

    contacts = out_dir / F["contacts"]
    final = out_dir / F["final"]

    if contacts.exists() and contacts.stat().st_size > 20:
        events = _from_contacts_csv(contacts, clip_name)
        print(f"events from contacts.csv → {len(events)} records")
    elif final.exists():
        events = _from_final_jsonl(final, clip_name)
        print(f"events from 5_hands_final.jsonl → {len(events)} records")
    else:
        raise FileNotFoundError(
            "Need contacts.csv or 5_hands_final.jsonl. "
            "Run: python run.py contacts CLIP"
        )

    # Sort by start time
    events.sort(key=lambda e: (e["timestamp_start"], e["hand"]))

    # Re-number ids after sort
    for i, e in enumerate(events, start=1):
        e["event_id"] = f"evt_{i:03d}"

    # JSON
    json_path = out_dir / F["events"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    # CSV (convenient for spreadsheets)
    csv_path = out_dir / F["events_csv"]
    fields = [
        "event_id", "timestamp_start", "timestamp_end",
        "hand", "object", "event", "confidence", "confidence_label",
        "source", "clip",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow({k: e.get(k) for k in fields})

    print(f"wrote {json_path.name} and {csv_path.name}")
    # Quick summary
    from collections import Counter
    print("event types:", dict(Counter(e["event"] for e in events)))
    print("objects:", dict(Counter(e["object"] for e in events).most_common(8)))
    return events


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.events <outputs/clip_dir> [clip_name]")
        sys.exit(1)
    build_events(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
