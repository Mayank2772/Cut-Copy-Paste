"""Name what each hold really is, using Gemini on small hand crops (one request per hold).
Run after `objects`:   python -m src.name_holds CLIP
Env: GEMINI_API_KEY (required), GEMINI_MODEL, MAX_HOLDS (default 100), GEMINI_DELAY seconds between calls (default 4).
Only tiny hand crops are sent (inline, nothing stored in Gemini file storage). Replies are cached on disk."""
import json, os, sys, time
from collections import Counter
from pathlib import Path
import cv2
from src.objects import hold_runs, write_contacts

PROMPT = """These {n} crops come from one moment in first-person factory video (head-mounted camera).
They show the worker's {side} hand. Name the ONE object the hand is gripping or using
(for example: pencil, power drill, nail gun, hammer, sander, angle grinder, saw, wooden block,
plywood sheet, tape measure, paint tin, cable).
Do NOT name the surface or work-piece the object is merely pressed against: a pencil writing on a plank is "pencil".
If the hand is only resting on or touching something and holds nothing, set holding to false.
Return JSON only: {{"object": "<short name>", "holding": true or false,
"boxes": [[ymin, xmin, ymax, xmax] or null, one per crop in order, each on a 0-1000 scale of that crop]}}"""


def pick_samples(run, k=3):
    """Up to k frames evenly spread through a hold."""
    if len(run) <= k:
        return list(range(len(run)))
    return sorted({round(i * (len(run) - 1) / (k - 1)) for i in range(k)})


def crop_region(hand, W, H, scale=2.2, min_size=160):
    x1, y1, x2, y2 = hand["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sz = max(max(x2 - x1, y2 - y1) * scale, min_size)
    return (max(0, int(cx - sz / 2)), max(0, int(cy - sz / 2)),
            min(W, int(cx + sz / 2)), min(H, int(cy + sz / 2)))


def box_to_frame(box, region):
    """Gemini box [ymin,xmin,ymax,xmax] on 0-1000 of the crop -> frame pixel [x1,y1,x2,y2], or None if invalid."""
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in box]
    except Exception:
        return None
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        return None
    x1, y1, x2, y2 = region
    cw, ch = x2 - x1, y2 - y1
    return [x1 + xmin / 1000 * cw, y1 + ymin / 1000 * ch, x1 + xmax / 1000 * cw, y1 + ymax / 1000 * ch]


def apply_result(run, picks, regions, result):
    """Write Gemini's answer into every frame of the hold."""
    holding = bool(result.get("holding", True))
    name = str(result.get("object", "")).strip().lower() or "unknown object"
    if not holding or name in ("none", "nothing"):
        for _, h in run:
            h["contact"], h["object"], h["object_bbox"] = None, None, None
        return "released"
    real = {}
    for j, (pi, box) in enumerate(zip(picks, (result.get("boxes") or [None] * len(picks)))):
        b = box_to_frame(box, regions[j]) if box else None
        if b:
            real[pi] = b
    for i, (_, h) in enumerate(run):
        h["object"], h["contact"], h["object_bbox"], h["bbox_est"] = name, "portable", None, False
        if i in real:
            h["object_bbox"] = real[i]
        elif real:
            j = min(real, key=lambda q: abs(q - i))
            if abs(j - i) <= 8:      # move the nearest real box along with the wrist
                a = run[j][1]
                dx, dy = h["kp2d"][0][0] - a["kp2d"][0][0], h["kp2d"][0][1] - a["kp2d"][0][1]
                b = real[j]
                h["object_bbox"], h["bbox_est"] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy], True
    return name


def ask_gemini(client, model, jpgs, side):
    from google.genai import types
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in jpgs]
    parts.append(PROMPT.format(n=len(jpgs), side=side))
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=model, contents=parts,
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            out = json.loads(resp.text)
            return out[0] if isinstance(out, list) and out else out
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(20 * (attempt + 1)); continue
            print("  Gemini error:", str(e)[:150]); return None
    return None


def main():
    from google import genai
    clip = sys.argv[1]
    out_dir = Path("outputs") / Path(clip).stem
    rows = [json.loads(l) for l in open(out_dir / "hands_labeled.jsonl")]
    cache_path = out_dir / "holds_gemini.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    delay, max_holds = float(os.environ.get("GEMINI_DELAY", 4)), int(os.environ.get("MAX_HOLDS", 100))
    client = genai.Client()
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(3)), int(cap.get(4))
    jobs = []
    for side in ("left", "right"):
        for run in hold_runs(rows, side):
            if run[-1][1] is not None and rows[run[-1][0]]["t"] - rows[run[0][0]]["t"] >= 0.25:
                jobs.append((side, run))
    jobs.sort(key=lambda j: -(rows[j[1][-1][0]]["t"] - rows[j[1][0][0]]["t"]))   # longest holds first
    jobs = jobs[:max_holds]
    print(f"{len(jobs)} holds to name with {model}")
    done = Counter()
    for n, (side, run) in enumerate(jobs, 1):
        key = f"{side}:{run[0][0]}-{run[-1][0]}"
        picks = pick_samples(run)
        regions = [crop_region(run[p][1], W, H) for p in picks]
        if key not in cache:
            jpgs = []
            for p, reg in zip(picks, regions):
                cap.set(cv2.CAP_PROP_POS_FRAMES, rows[run[p][0]]["frame"])
                ok, img = cap.read()
                if not ok:
                    continue
                crop = img[reg[1]:reg[3], reg[0]:reg[2]]
                jpgs.append(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
            if len(jpgs) != len(picks):
                continue
            cache[key] = ask_gemini(client, model, jpgs, side)
            json.dump(cache, open(cache_path, "w"), indent=1)
            time.sleep(delay)
        if cache[key]:
            done[apply_result(run, picks, regions, cache[key])] += 1
        if n % 10 == 0:
            print(f"  {n}/{len(jobs)}", flush=True)
    cap.release()
    with open(out_dir / "hands_labeled.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = write_contacts(rows, out_dir, Path(clip).name)
    print("named holds:", dict(done.most_common(15)))
    print(f"{len(res)} contacts written to contacts.csv")


if __name__ == "__main__":
    main()
