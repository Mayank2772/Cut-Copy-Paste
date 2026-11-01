"""Factory Memory — one entry point.

  python run.py STEP CLIP

STEP
  track      step 1  MediaPipe hands + recovery + smoothing
  grip       step 2  left/right fix, gap fill, bent fingers
  objects    step 3  objects near hands (expanded search)
  contacts   step 4  approach / grab / hold / release  → contacts.csv
  events     step 5  Factory Memory searchable events → events.json
  render     step 6  draw overlay video
  all        track → grip → objects → contacts → events → render
  timeline   PS3 Gemini task-step timeline
  label      manual accuracy check on ~100 frames
  serve      open the Factory Memory UI (simple local server)

Options:
  --every N     process every Nth frame in track (default 2)
  --no-gemini   skip Gemini naming
  --n 100 / --report   for label
"""
import argparse
import json
import os
import sys
from pathlib import Path

from src.common import F, dirs
from dotenv import load_dotenv

load_dotenv()
ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument(
    "step",
    choices=[
        "track", "grip", "objects", "contacts", "events", "render",
        "all", "timeline", "label", "serve",
    ],
)
ap.add_argument("clip", nargs="?", default="")
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--no-gemini", action="store_true")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--report", action="store_true")
ap.add_argument("--port", type=int, default=8765)
a = ap.parse_args()

if a.step != "serve":
    if not a.clip:
        sys.exit("CLIP path is required for this step")
    if not Path(a.clip).exists():
        sys.exit(f"Clip not found: {a.clip}")

if a.no_gemini:
    os.environ["NO_GEMINI"] = "1"

if a.step == "serve":
    import http.server
    import mimetypes
    import re
    import threading
    import uuid
    import webbrowser
    from urllib.parse import urlparse, parse_qs, unquote
    from http.server import ThreadingHTTPServer

    root = Path(__file__).resolve().parent
    os.chdir(root)
    uploads = root / "uploads"
    uploads.mkdir(exist_ok=True)
    jobs = {}
    jobs_lock = threading.Lock()

    class FactoryHandler(http.server.SimpleHTTPRequestHandler):
        server_version = "FactoryMemory/1.0"

        def log_message(self, fmt, *args):
            print("[http] " + fmt % args)

        def _json(self, code, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/upload":
                return self._json(404, {"error": "Unknown endpoint"})
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2 * 1024 * 1024 * 1024:
                return self._json(413, {"error": "Video is empty or exceeds the 2 GB upload limit"})
            filename = unquote(parse_qs(parsed.query).get("filename", ["video.mp4"])[0])
            original = Path(filename).name
            ext = Path(original).suffix.lower()
            if ext not in {".mp4", ".mov", ".avi", ".webm", ".m4v"}:
                return self._json(400, {"error": "Supported formats: MP4, MOV, AVI, WebM, M4V"})
            job_id = uuid.uuid4().hex[:10]
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original).stem)[:70] or "video"
            target = uploads / f"{safe_name}_{job_id}{ext}"
            remaining = length
            try:
                with target.open("wb") as f:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise OSError("Upload ended unexpectedly")
                        f.write(chunk)
                        remaining -= len(chunk)
            except Exception as exc:
                target.unlink(missing_ok=True)
                return self._json(400, {"error": f"Upload failed: {exc}"})

            record = {"job_id": job_id, "filename": original, "stored_name": target.name,
                      "status": "queued", "progress": 0, "message": "Video uploaded; starting ML pipeline",
                      "log": "", "output_video": None, "events": None}
            with jobs_lock:
                jobs[job_id] = record

            def process_video():
                import subprocess
                record["status"] = "processing"
                record["message"] = "Running hand, object and event analysis"
                cmd = [sys.executable, str(root / "run.py"), "all", str(target)]
                try:
                    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
                    record["log"] = (proc.stdout or "") + "\n" + (proc.stderr or "")
                    if proc.returncode != 0:
                        record["status"] = "failed"
                        record["message"] = "ML pipeline failed. Open details to inspect the error."
                        return
                    stem = target.stem
                    event_file = root / "outputs" / stem / "events.json"
                    rendered = root / "output_v" / stem
                    candidates = [rendered / "overlay.mp4", rendered / "overlay_raw.avi"]
                    video_file = next((x for x in candidates if x.exists()), None)
                    record["events"] = f"/api/events/{job_id}"
                    record["output_video"] = f"/media/output/{stem}/{video_file.name}" if video_file else None
                    record["status"] = "completed"
                    record["progress"] = 100
                    record["message"] = "Analysis complete"
                    if not event_file.exists():
                        record["message"] = "Video processed, but no events file was generated"
                except Exception as exc:
                    record["status"] = "failed"
                    record["message"] = str(exc)

            threading.Thread(target=process_video, daemon=True).start()
            self._json(202, {"job_id": job_id, "filename": original, "stored_name": target.name, "status": "queued"})

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/status/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                with jobs_lock:
                    job = jobs.get(job_id)
                    if not job:
                        return self._json(404, {"error": "Job not found"})
                    safe = {k: v for k, v in job.items() if k != "log"}
                    safe["log_available"] = bool(job.get("log"))
                return self._json(200, safe)
            if parsed.path.startswith("/api/log/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                with jobs_lock:
                    job = jobs.get(job_id)
                    if not job:
                        return self._json(404, {"error": "Job not found"})
                    return self._json(200, {"log": job.get("log", "")})
            if parsed.path.startswith("/api/events/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                with jobs_lock:
                    job = jobs.get(job_id)
                if not job or job.get("status") != "completed":
                    return self._json(404, {"error": "Events are not ready"})
                stem = Path(job["stored_name"]).stem
                path = root / "outputs" / stem / "events.json"
                if not path.exists():
                    return self._json(200, [])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(path.read_bytes())
                return
            if parsed.path.startswith("/media/uploads/"):
                self.path = "/uploads/" + parsed.path.split("/media/uploads/", 1)[1]
            elif parsed.path.startswith("/media/output/"):
                parts = parsed.path.split("/")
                if len(parts) >= 5:
                    self.path = f"/output_v/{parts[3]}/{parts[4]}"
            return super().do_GET()

        def send_head(self):
            # Byte-range support is important for video seeking in Safari.
            path = self.translate_path(self.path)
            if not os.path.isfile(path):
                return super().send_head()
            f = open(path, "rb")
            size = os.fstat(f.fileno()).st_size
            rng = self.headers.get("Range")
            if rng:
                match = re.match(r"bytes=(\d*)-(\d*)$", rng)
                if match:
                    start = int(match.group(1) or 0)
                    end = int(match.group(2) or size - 1)
                    if start >= size or end < start:
                        f.close()
                        self.send_error(416, "Requested Range Not Satisfiable")
                        return None
                    end = min(end, size - 1)
                    self.send_response(206)
                    self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.end_headers()
                    f.seek(start)
                    self._range_remaining = end - start + 1
                    return f
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._range_remaining = None
            return f

        def copyfile(self, source, outputfile):
            remaining = getattr(self, "_range_remaining", None)
            if remaining is None:
                return super().copyfile(source, outputfile)
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)

    url = f"http://127.0.0.1:{a.port}/frontend/"
    with ThreadingHTTPServer(("127.0.0.1", a.port), FactoryHandler) as httpd:
        print(f"Factory Memory UI → {url}")
        print("Upload a video in the browser. Press Ctrl+C to stop.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()
    sys.exit(0)

out, vid = dirs(a.clip)
todo = (
    ["track", "grip", "objects", "contacts", "events", "render"]
    if a.step == "all"
    else [a.step]
)

for step in todo:
    print(f"\n=== {step} ===", flush=True)
    if step == "track":
        from src.tracking import run_hands
        print(json.dumps(run_hands(a.clip, out, step=a.every), indent=2))
    elif step == "grip":
        from src.grip import process
        print(json.dumps(process(out), indent=2))
    elif step == "objects":
        from src.objects import run
        run(a.clip, out)
    elif step == "contacts":
        from src.states import run
        run(Path(a.clip), out)
    elif step == "events":
        from src.events import build_events
        events = build_events(out, Path(a.clip).name)
        print(json.dumps({"n_events": len(events)}, indent=2))
    elif step == "render":
        from src.render import render
        print("saved", render(a.clip, out, vid))
    elif step == "label":
        from src.label import main
        main(a.clip, out, a.n, a.report)
    elif step == "timeline":
        import cv2
        import pandas as pd
        from src.timeline import clean, get_timeline
        cap = cv2.VideoCapture(a.clip)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        df = pd.DataFrame(clean(get_timeline(a.clip, out), dur)).assign(
            clip=Path(a.clip).name
        )
        df.to_csv(out / F["actions"], index=False)
        print(df)
