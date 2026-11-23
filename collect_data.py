import cv2
import mediapipe as mp
import csv
import time
from pathlib import Path

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

CSV_FILE = DATASET_DIR / "landmarks.csv"

# 2. All gesture labels
gestures = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
gesture_index = 0
label = gestures[gesture_index]

# 3. MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# 4. Hand connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# 5. CSV headers
headers = (
    [f"left_{axis}{i}" for i in range(21)
     for axis in ("x", "y", "z")]
    +
    [f"right_{axis}{i}" for i in range(21)
     for axis in ("x", "y", "z")]
    +
    ["label"]
)

new_file = not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0

saved_counts = {letter: 0 for letter in gestures}

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not detected!")

start_time = time.monotonic()
last_timestamp = -1

try:
    with open(CSV_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if new_file:
            writer.writerow(headers)

        with HandLandmarker.create_from_options(options) as detector:

            while True:

                success, frame = camera.read()

                if not success:
                    print("Camera frame unavailable.")
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )

                timestamp = max(timestamp, last_timestamp + 1)
                last_timestamp = timestamp

                result = detector.detect_for_video(
                    mp_image, timestamp
                )

                left_hand = [0.0] * 63
                right_hand = [0.0] * 63

                if result.hand_landmarks:

                    for index, hand in enumerate(result.hand_landmarks):

                        side = "unknown"

                        if (
                            result.handedness
                            and index < len(result.handedness)
                            and result.handedness[index]
                        ):
                            side = (
                                result.handedness[index][0]
                                .category_name.lower()
                            )

                        if "left" in side:
                            target = left_hand
                        elif "right" in side:
                            target = right_hand
                        else:
                            target = (
                                left_hand if index == 0
                                else right_hand
                            )

                        height, width, _ = frame.shape
                        points = []

                        for landmark in hand:

                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            points.append((x, y))

                            cv2.circle(
                                frame, (x, y), 5,
                                (0, 255, 0), -1
                            )

                        target[:] = [
                            value
                            for landmark in hand
                            for value in (
                                landmark.x,
                                landmark.y,
                                landmark.z
                            )
                        ]

                        for start, end in connections:
                            cv2.line(
                                frame,
                                points[start],
                                points[end],
                                (255, 0, 0),
                                2
                            )

                # 6. Display information

                cv2.putText(
                    frame,
                    f"GESTURE: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Progress: {gesture_index + 1}/26",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Samples: {saved_counts[label]}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "S=Save | N=Next | P=Previous | Q=Quit",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("ISL Gesture Collection", frame)

                # 7. Keyboard controls

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):

                    if not result.hand_landmarks:
                        print("No hand detected. Try again.")
                        continue

                    features = left_hand + right_hand

                    writer.writerow(features + [label])
                    file.flush()

                    saved_counts[label] += 1

                    print(
                        f"Saved {label}: "
                        f"{saved_counts[label]} samples"
                    )

                elif key == ord("n"):

                    if gesture_index < len(gestures) - 1:
                        gesture_index += 1
                        label = gestures[gesture_index]
                        print("Now collecting:", label)
                    else:
                        print("All 26 gestures completed!")

                elif key == ord("p"):

                    if gesture_index > 0:
                        gesture_index -= 1
                        label = gestures[gesture_index]
                        print("Back to:", label)

                elif key == ord("q"):
                    break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nData collection stopped.")
print("Samples collected:")

for letter, count in saved_counts.items():
    print(f"{letter}: {count}")

print("\nDataset saved at:", CSV_FILE)