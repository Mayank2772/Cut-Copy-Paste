import cv2
import json
import time
from pathlib import Path

from hand_tracker import HandTracker


DATA_DIR = Path("egohack-data")
OUTPUT_DIR = Path("output")

# Set to 1 for best tracking accuracy.
# Set to 2 for faster processing.
FRAME_STEP = 2

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".webm"
}


def process_video(video_path: Path):
    relative_path = video_path.relative_to(DATA_DIR)

    # Preserve the input folder structure in output/.
    output_video = (
        OUTPUT_DIR / relative_path.parent /
        f"{video_path.stem}_annotated.mp4"
    )

    output_jsonl = (
        OUTPUT_DIR / relative_path.parent /
        f"{video_path.stem}_hands.jsonl"
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {video_path}")
    print(f"Video output: {output_video}")
    print(f"JSON output:  {output_jsonl}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / FRAME_STEP,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_video}")

    # Create a fresh tracker per video so prior-video smoothing is not reused.
    tracker = HandTracker(smoothing=0.55)

    frame_index = 0
    processed_frames = 0
    frames_with_hands = 0
    start_time = time.perf_counter()

    try:
        with open(output_jsonl, "w", encoding="utf-8") as json_file:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                current_index = frame_index
                frame_index += 1

                if current_index % FRAME_STEP != 0:
                    continue

                timestamp = current_index / fps
                timestamp_ms = int(timestamp * 1000)

                hands = tracker.detect(frame, timestamp_ms)

                processed_frames += 1

                if hands:
                    frames_with_hands += 1

                json_file.write(
                    json.dumps({
                        "frame": current_index,
                        "t": timestamp,
                        "hands": hands
                    }) + "\n"
                )

                # Connected skeleton. Change to True to show hand boxes.
                frame = tracker.draw(
                    frame,
                    hands,
                    show_box=False
                )

                writer.write(frame)

                if processed_frames % 100 == 0:
                    print(
                        f"  Processed {processed_frames} "
                        f"of about {total_frames // FRAME_STEP} frames"
                    )

    finally:
        cap.release()
        writer.release()
        tracker.close()

    elapsed = time.perf_counter() - start_time
    coverage = (
        frames_with_hands / processed_frames
        if processed_frames else 0
    )

    print(f"Complete: {video_path.name}")
    print(f"  Processed frames: {processed_frames}")
    print(f"  Hand detection coverage: {coverage:.2%}")
    print(
        f"  Processing speed: "
        f"{processed_frames / elapsed:.2f} FPS"
        if elapsed > 0 else ""
    )

    return True


def main():
    if not DATA_DIR.exists():
        raise RuntimeError(
            f"Folder not found: {DATA_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    videos = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise RuntimeError(
            f"No videos found inside: {DATA_DIR.resolve()}"
        )

    print(f"Found {len(videos)} video(s).")

    successful = 0

    for video_path in videos:
        try:
            if process_video(video_path):
                successful += 1
        except Exception as error:
            print(f"\nFailed: {video_path}")
            print(f"Reason: {error}")

    print("\nAll processing finished.")
    print(f"Successful videos: {successful}/{len(videos)}")


if __name__ == "__main__":
    main()