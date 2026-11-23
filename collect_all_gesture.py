import cv2
import csv
import time
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- SETTINGS ----------------

MODEL_PATH = "hand_landmarker.task"
OUTPUT_FILE = "isl_dataset.csv"

GESTURES = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "PLEASE",
    "HELP",
    "GOOD",
    "BAD",
    "LOVE",
    "STOP"
]

SAMPLES_PER_GESTURE = 60

# -------------------------------------------

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    landmarker.close()
    exit()

file_exists = os.path.exists(OUTPUT_FILE)

with open(OUTPUT_FILE, "a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:
        headers = ["gesture"]
        for hand in range(2):
            for point in range(21):
                headers.extend([
                    f"h{hand}_x{point}",
                    f"h{hand}_y{point}",
                    f"h{hand}_z{point}"
                ])
        writer.writerow(headers)

    last_timestamp = 0

    for gesture in GESTURES:

        print("\nPrepare gesture:", gesture)

        # Countdown
        for count in [3, 2, 1]:
            start = time.time()

            while time.time() - start < 1:
                ret, frame = camera.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)

                cv2.putText(
                    frame,
                    f"NEXT: {gesture} - {count}",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("ISL Data Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

        collected = 0

        while collected < SAMPLES_PER_GESTURE:

            ret, frame = camera.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            timestamp = time.monotonic_ns() // 1_000_000

            if timestamp <= last_timestamp:
                timestamp = last_timestamp + 1

            last_timestamp = timestamp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            # Two hands, 21 landmarks each
            hand_data = [[0.0] * 63 for _ in range(2)]

            for hand_index, hand_landmarks in enumerate(
                result.hand_landmarks
            ):

                if hand_index >= 2:
                    break

                values = []

                for point in hand_landmarks:
                    values.extend([
                        point.x,
                        point.y,
                        point.z
                    ])

                hand_data[hand_index] = values

            if result.hand_landmarks:

                row = [gesture]

                for hand in hand_data:
                    row.extend(hand)

                writer.writerow(row)
                file.flush()

                collected += 1

            cv2.putText(
                frame,
                f"GESTURE: {gesture}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {collected}/{SAMPLES_PER_GESTURE}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("ISL Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        print(f"{gesture}: {collected} samples saved!")

print("\nALL GESTURES COLLECTED!")

camera.release()
cv2.destroyAllWindows()
landmarker.close()