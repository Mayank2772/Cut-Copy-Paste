# Factory vision

The complete pipeline is maintained in the sibling `nilesh` folder. This
folder provides the requested `factory-vision` entry point and dependencies.

## Windows setup

```powershell
cd "B:\Cut_Copy_Paste_BIOS\Cut-Copy-Paste\factory-vision"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
curl.exe -L -o ..\nilesh\hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

Put a video in `nilesh\egohack-data\clips_480p\`, then run:

```powershell
python run.py all ..\nilesh\egohack-data\clips_480p\factory_001.mp4 --no-gemini
```

The same pipeline can be started with the original entry point:

```powershell
python src\main.py ..\nilesh\egohack-data\clips_480p\factory_001.mp4 --no-gemini
```

The `all` pipeline produces:

- `nilesh\outputs\<clip>\5_hands_final.jsonl`: hand, object, grab, and stability data
- `nilesh\outputs\<clip>\contacts.csv`: grab/hold/release timeline
- `nilesh\outputs\<clip>\3_hands_objects.jsonl`: all visible object boxes
- `nilesh\output_v\<clip>\overlay.mp4`: annotated video

The detector scans the full frame and enlarged hand crops. Therefore objects
are labelled whether or not a hand is grabbing them. `OBJ_MODEL`, `OBJ_CONF`,
and `OBJ_CLASSES` can be set as environment variables for a different
Ultralytics model or custom YOLO-World labels.
