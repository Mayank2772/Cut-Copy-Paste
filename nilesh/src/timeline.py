"""PS3 core: Gemini video -> JSON action timeline. Responses are cached on disk."""
import json, os, time
from pathlib import Path
from google import genai
from google.genai import types

VERBS = ["pick", "place", "hold", "insert", "fasten", "tighten", "press", "pull",
         "inspect", "wipe", "carry", "align", "cut", "open", "close"]

PROMPT = f"""This is first-person video from a head-mounted camera in a factory.
List every distinct hand action in chronological order as a JSON array.
Each item: start_s (float seconds), end_s (float seconds), verb (one of {VERBS}),
object (short noun phrase for what is handled), hand ("left", "right" or "both").
Rules: actions must not overlap for the same hand; use only seconds that exist in the video;
skip moments where nothing is being done. Return only JSON."""


def get_timeline(clip, out_dir, fps=4, model=None):
    """fps: frames/sec Gemini samples (default is only ~1, too low for fast hand moves)."""
    clip, out_dir = Path(clip), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "actions_raw.json"
    if cache.exists():
        return json.load(open(cache))
    model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")  # check docs for current name
    client = genai.Client()  # reads GEMINI_API_KEY from env
    f = client.files.upload(file=str(clip))
    try:
        while f.state.name == "PROCESSING":
            time.sleep(5)
            f = client.files.get(name=f.name)
        part = types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                          video_metadata=types.VideoMetadata(fps=fps))
        resp = client.models.generate_content(
            model=model, contents=[part, PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        actions = json.loads(resp.text)
    finally:
        client.files.delete(name=f.name)  # DATA RULE: remove upload
    json.dump(actions, open(cache, "w"), indent=2)
    return actions


def clean(actions, duration):
    """Clamp to clip length, drop bad rows, sort."""
    out = []
    for a in actions:
        s, e = float(a["start_s"]), float(a["end_s"])
        s, e = max(0.0, s), min(duration, e)
        if e > s:
            out.append({**a, "start_s": s, "end_s": e})
    return sorted(out, key=lambda a: a["start_s"])
