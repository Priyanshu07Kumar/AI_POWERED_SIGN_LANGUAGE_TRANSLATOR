import cv2
import mediapipe as mp
import csv
import time
from pathlib import Path

# =====================================
# 1. PROJECT SETTINGS
# =====================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
CSV_FILE = BASE_DIR / "isl_test_dataset.csv"

gestures = [
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

# =====================================
# 2. FEATURE HEADERS (126 FEATURES)
# =====================================

headers = []

for hand in range(2):
    for i in range(21):
        for axis in ("x", "y", "z"):
            headers.append(f"h{hand}_{axis}{i}")

headers.append("gesture")

print("Total features:", len(headers) - 1)

# =====================================
# 3. MEDIAPIPE SETUP
# =====================================

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

# =====================================
# 4. INITIALIZE DATASET
# =====================================

new_file = (
    not CSV_FILE.exists()
    or CSV_FILE.stat().st_size == 0
)

saved_counts = {g: 0 for g in gestures}

gesture_index = 0
label = gestures[gesture_index]

# =====================================
# 5. START CAMERA
# =====================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not detected!")

start_time = time.monotonic()
last_timestamp = -1

print("Testing data collection started!")
print("Press S = Save | N = Next | P = Previous | Q = Quit")

# =====================================
# 6. COLLECT TEST DATA
# =====================================

try:

    with open(CSV_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if new_file:
            writer.writerow(headers)

        with HandLandmarker.create_from_options(options) as detector:

            while True:

                success, frame = camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )

                timestamp = max(
                    timestamp,
                    last_timestamp + 1
                )

                last_timestamp = timestamp

                result = detector.detect_for_video(
                    mp_image,
                    timestamp
                )

                # Two hands, 63 features each
                hand0 = [0.0] * 63
                hand1 = [0.0] * 63

                if result.hand_landmarks:

                    for index, hand in enumerate(
                        result.hand_landmarks[:2]
                    ):

                        features = [
                            value
                            for landmark in hand
                            for value in (
                                landmark.x,
                                landmark.y,
                                landmark.z
                            )
                        ]

                        if index == 0:
                            hand0 = features
                        else:
                            hand1 = features

                        height, width, _ = frame.shape

                        for landmark in hand:

                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            cv2.circle(
                                frame,
                                (x, y),
                                5,
                                (0, 255, 0),
                                -1
                            )

                # =================================
                # 7. DISPLAY INFORMATION
                # =================================

                cv2.putText(
                    frame,
                    f"GESTURE: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Progress: {gesture_index + 1}/10",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Test Samples: {saved_counts[label]}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "S=Save N=Next P=Previous Q=Quit",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "ISL Testing Data Collection",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                # SAVE SAMPLE
                if key == ord("s"):

                    if not result.hand_landmarks:
                        print("No hand detected!")
                        continue

                    features = hand0 + hand1

                    writer.writerow(features + [label])
                    file.flush()

                    saved_counts[label] += 1

                    print(
                        f"Saved {label}: "
                        f"{saved_counts[label]}"
                    )

                # NEXT GESTURE
                elif key == ord("n"):

                    if gesture_index < len(gestures) - 1:
                        gesture_index += 1
                        label = gestures[gesture_index]

                        print("Now collecting:", label)
                    else:
                        print("All gestures completed!")

                # PREVIOUS GESTURE
                elif key == ord("p"):

                    if gesture_index > 0:
                        gesture_index -= 1
                        label = gestures[gesture_index]

                        print("Back to:", label)

                # EXIT
                elif key == ord("q"):
                    break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nTesting data collection completed!")

for gesture, count in saved_counts.items():
    print(gesture, ":", count)

print("\nDataset saved at:", CSV_FILE)