"""STEP 4a - naming: ask Gemini what each grip really is (one request per grip, up to 3 small hand crops).
Cached in outputs/<clip>/4_hold_names_gemini.json, so a re-run only asks for what is missing.
Env: GEMINI_API_KEY, GEMINI_MODEL, GEMINI_DELAY (s between calls, default 4), MAX_HOLDS (default 100).
Only small hand crops are sent, inline; nothing is stored in Gemini file storage."""
import json, os, time
from collections import Counter
import cv2
from src.common import hand_of

PROMPT = """{n} crops of the worker's {side} hand (first-person factory video), fingers bent as if gripping.{hint}
Name the ONE object the hand grips or uses (e.g. pencil, power drill, nail gun, hammer, sander, angle grinder,
saw, wooden block, plywood sheet, tape measure, paint tin, cable). Not the surface it presses on:
a pencil writing on a plank is "pencil". If the hand holds nothing, holding=false.
JSON only: {{"object": str, "holding": bool, "boxes": [[ymin,xmin,ymax,xmax] or null per crop, 0-1000 of that crop]}}"""

# Models are tried in this order. Override: GEMINI_MODELS="model-a,model-b".
DEFAULT_MODELS = "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.8-flash"
TRANSIENT = ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE",
             "overloaded", "high demand", "timed out", "Connection")
GONE = ("404", "NOT_FOUND", "no longer available", "not found")
FATAL = ("API key not valid", "API_KEY_INVALID", "PERMISSION_DENIED")


class Chain:
    """Which model to ask next; remembers models that are gone / busy, and stops after repeated failures."""
    def __init__(self, models=None, max_fail=4):
        env = os.environ.get("GEMINI_MODELS") or os.environ.get("GEMINI_MODEL") or DEFAULT_MODELS
        self.models = models or [m.strip().replace("models/", "") for m in env.split(",") if m.strip()]
        self.dead, self.bad_until, self.no_think = set(), {}, set()
        self.fails, self.max_fail, self.fatal = 0, max_fail, False

    def available(self):
        now = time.time()
        return [m for m in self.models if m not in self.dead and self.bad_until.get(m, 0) <= now]

    @property
    def stopped(self):
        return self.fatal or self.fails >= self.max_fail


def pick_samples(n, k=3):
    """Indices of up to k frames evenly spread through n frames."""
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def crop_region(hand, W, H, scale=2.2, min_size=160):
    x1, y1, x2, y2 = hand["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sz = max(max(x2 - x1, y2 - y1) * scale, min_size)
    return (max(0, int(cx - sz / 2)), max(0, int(cy - sz / 2)), min(W, int(cx + sz / 2)), min(H, int(cy + sz / 2)))


def box_to_frame(box, region):
    """Gemini box [ymin,xmin,ymax,xmax] on 0-1000 of the crop -> frame pixels [x1,y1,x2,y2]; None if invalid."""
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in box]
    except Exception:
        return None
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        return None
    x1, y1, x2, y2 = region
    cw, ch = x2 - x1, y2 - y1
    return [x1 + xmin / 1000 * cw, y1 + ymin / 1000 * ch, x1 + xmax / 1000 * cw, y1 + ymax / 1000 * ch]


def clean_name(s):
    s = str(s or "").strip().lower()
    for art in ("a ", "an ", "the "):
        if s.startswith(art):
            s = s[len(art):]
    return s


def make_config(types, model, think_off):
    kw = {"response_mime_type": "application/json", "max_output_tokens": 300, "temperature": 0}
    if think_off and "2.5" in model and hasattr(types, "ThinkingConfig"):
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)   # no hidden thinking tokens
    return types.GenerateContentConfig(**kw)


def ask_gemini(client, chain, jpgs, side, hint="", tries=3):
    """Ask the first usable model; on 'gone' skip it for good, on lasting overload rest it 5 min and use the next."""
    from google.genai import types
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in jpgs]
    parts.append(PROMPT.format(n=len(jpgs), side=side, hint=f" A detector guessed: {hint} (may be wrong)." if hint else ""))
    for model in chain.available():
        attempt = 0
        while attempt < tries:
            try:
                resp = client.models.generate_content(
                    model=model, contents=parts, config=make_config(types, model, model not in chain.no_think))
                out = json.loads(resp.text)
                chain.fails = 0
                return out[0] if isinstance(out, list) and out else out
            except Exception as e:
                msg = str(e)
                if "thinking" in msg.lower() and model not in chain.no_think:
                    chain.no_think.add(model); continue            # retry once without the thinking option
                if any(f in msg for f in FATAL):
                    print("  Gemini key problem:", msg[:120]); chain.fatal = True; return None
                if any(g in msg for g in GONE) and "429" not in msg and "503" not in msg:
                    print(f"  model {model} is not available for this key, skipping it"); chain.dead.add(model); break
                if any(t in msg for t in TRANSIENT):
                    attempt += 1
                    if attempt < tries:
                        wait = 10 * 2 ** (attempt - 1)
                        print(f"  {model} busy ({msg[:40]}...), retry in {wait}s", flush=True)
                        time.sleep(wait); continue
                    print(f"  {model} stays busy, resting it for 5 min and trying the next model")
                    chain.bad_until[model] = time.time() + 300
                    break
                print("  Gemini error:", msg[:150]); chain.fails += 1; return None
    chain.fails += 1
    return None


def small_jpg(img, max_side=320):
    h, w = img.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def name_segments(clip, rows, segs, cache_path):
    """Fills seg['gem'] (Gemini's answer or None) and seg['gem_anchors'] ([(row, frame_box)])."""
    from google import genai
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    cache = {k: v for k, v in cache.items() if v}                 # failures are never cached
    delay, max_n = float(os.environ.get("GEMINI_DELAY", 4)), int(os.environ.get("MAX_HOLDS", 40))
    k_samples = int(os.environ.get("GEMINI_SAMPLES", 2))
    chain = Chain()
    client = genai.Client()
    cap = cv2.VideoCapture(str(clip))
    W, H = int(cap.get(3)), int(cap.get(4))
    stats = Counter()
    order = sorted(segs, key=lambda s: -(s["t1"] - s["t0"]))[:max_n]
    print(f"asking Gemini about {len(order)} grips, models in order: {', '.join(chain.models)}")
    for n, seg in enumerate(order, 1):
        if chain.stopped:
            stats["skipped_after_failures"] = len(order) - n + 1
            print("Gemini keeps failing: stopping. The rest use detector names. Run `contacts` again later to retry only those.")
            break
        side = seg["side"]
        ks = [k for k in range(seg["k0"], seg["k1"] + 1) if hand_of(rows[k], side)]
        if not ks:
            continue
        picks = [ks[i] for i in pick_samples(len(ks), k_samples)]
        regions = [crop_region(hand_of(rows[k], side), W, H) for k in picks]
        w0 = hand_of(rows[ks[0]], side)["kp2d"][0]
        key = f"{side}:{seg['k0']}-{seg['k1']}:{round(w0[0])},{round(w0[1])}"
        res = cache.get(key)
        if res is None:
            jpgs = []
            for k, reg in zip(picks, regions):
                cap.set(cv2.CAP_PROP_POS_FRAMES, rows[k]["frame"])
                ok, img = cap.read()
                if ok:
                    jpgs.append(small_jpg(img[reg[1]:reg[3], reg[0]:reg[2]]))
            if len(jpgs) != len(picks):
                stats["failed"] += 1; continue
            res = ask_gemini(client, chain, jpgs, side, seg.get("det_name") or "")
            if res:
                cache[key] = res
                json.dump(cache, open(cache_path, "w"), indent=1)
                stats["asked"] += 1
            else:
                stats["failed"] += 1
            time.sleep(delay)
        else:
            stats["cached"] += 1
        if res:
            seg["gem"] = res
            seg["gem_anchors"] = [(k, b) for k, reg, box in zip(picks, regions, (res.get("boxes") or []) + [None] * 3)
                                  if box and (b := box_to_frame(box, reg))]
        if n % 10 == 0:
            print(f"  {n}/{len(order)}", flush=True)
    cap.release()
    return dict(stats)
