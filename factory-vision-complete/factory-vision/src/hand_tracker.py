import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                 # palm
)

FINGERTIPS = {4, 8, 12, 16, 20}


class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", smoothing=0.55):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.90))
        self.previous_points = {}
        self.previous_boxes = {}

    def _smooth_points(self, side, points):
        current = np.asarray(points, dtype=np.float32)
        previous = self.previous_points.get(side)

        if previous is not None and previous.shape == current.shape:
            current = (
                self.smoothing * previous
                + (1.0 - self.smoothing) * current
            )

        self.previous_points[side] = current
        return current

    def detect(self, frame, timestamp_ms):
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect_for_video(image, timestamp_ms)
        hands = []

        for landmarks, handedness, world_landmarks in zip(
            result.hand_landmarks,
            result.handedness,
            result.hand_world_landmarks
        ):
            side = handedness[0].category_name.lower()

            raw_points = [
                (point.x * width, point.y * height)
                for point in landmarks
            ]

            points = self._smooth_points(side, raw_points)

            xs = points[:, 0]
            ys = points[:, 1]
            padding = 16

            bbox = np.array([
                max(0, xs.min() - padding),
                max(0, ys.min() - padding),
                min(width - 1, xs.max() + padding),
                min(height - 1, ys.max() + padding),
            ], dtype=np.float32)

            previous_box = self.previous_boxes.get(side)
            if previous_box is not None:
                bbox = (
                    self.smoothing * previous_box
                    + (1.0 - self.smoothing) * bbox
                )

            self.previous_boxes[side] = bbox

            hands.append({
                "side": side,
                "score": float(handedness[0].score),
                "bbox": bbox.tolist(),
                "kp2d": points.tolist(),
                "kp3d": [
                    [point.x, point.y, point.z]
                    for point in world_landmarks
                ],
            })

        return hands

    @staticmethod
    def _draw_label(frame, text, origin, color):
        x, y = origin

        (w, h), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )

        cv2.rectangle(
            frame,
            (x, y - h - 11),
            (x + w + 14, y + baseline - 3),
            (18, 18, 18),
            -1
        )

        cv2.rectangle(
            frame,
            (x, y - h - 11),
            (x + w + 14, y + baseline - 3),
            color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            text,
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA
        )

    def draw(self, frame, hands, show_box=False):
        for hand in hands:
            points = np.rint(
                np.asarray(hand["kp2d"])
            ).astype(int)

            # Different skeleton colour per hand
            accent = (
                (255, 210, 30)     # cyan / yellow-blue for right hand
                if hand["side"] == "right"
                else (255, 90, 220)  # pink for left hand
            )

            x1, y1, x2, y2 = map(int, hand["bbox"])

            if show_box:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    accent,
                    1,
                    cv2.LINE_AA
                )

            self._draw_label(
                frame,
                f"{hand['side'].upper()}  {hand['score']:.0%}",
                (x1, max(28, y1)),
                accent
            )

            # Black outline first: makes bones visible on busy backgrounds
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    tuple(points[start]),
                    tuple(points[end]),
                    (10, 10, 10),
                    6,
                    cv2.LINE_AA
                )

            # Actual connected skeleton
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    tuple(points[start]),
                    tuple(points[end]),
                    accent,
                    3,
                    cv2.LINE_AA
                )

            # Draw joints
            for index, point in enumerate(points):
                radius = 7 if index in FINGERTIPS else 5

                # Dark border
                cv2.circle(
                    frame,
                    tuple(point),
                    radius + 2,
                    (10, 10, 10),
                    -1,
                    cv2.LINE_AA
                )

                # White joint
                cv2.circle(
                    frame,
                    tuple(point),
                    radius,
                    (245, 245, 245),
                    -1,
                    cv2.LINE_AA
                )

                # Coloured center
                core_colour = (
                    (0, 150, 255)
                    if index in FINGERTIPS
                    else accent
                )

                cv2.circle(
                    frame,
                    tuple(point),
                    3 if index in FINGERTIPS else 2,
                    core_colour,
                    -1,
                    cv2.LINE_AA
                )

        return frame

    def close(self):
        self.landmarker.close()
