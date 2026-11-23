import cv2
import mediapipe as mp
import csv
import time
from pathlib import Path

# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "hand_landmarker.task"
CSV_FILE = BASE_DIR / "isl_dataset.csv"

# ==========================================
# 2. GESTURE CLASSES
# ==========================================

ISL_GESTURES = [
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

ALPHABET_GESTURES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

gestures =ALPHABET_GESTURES + ["NO_GESTURE"]

gesture_index = 0
label = gestures[gesture_index]

SAMPLES_PER_CLASS = 120

saved_counts = {g: 0 for g in gestures}

# ==========================================
# 3. CSV FEATURE HEADERS
# ==========================================

feature_headers = [
    f"h{hand}_{axis}{point}"
    for hand in range(2)
    for point in range(21)
    for axis in ("x", "y", "z")
]


headers = ["gesture"] + feature_headers
# Create CSV only if it does not exist
new_file = not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0

if not new_file:
    with open(CSV_FILE, "r", newline="") as check_file:
        existing_headers = next(csv.reader(check_file), [])

    if existing_headers != headers:
        raise ValueError(
            "Your existing CSV columns do not match this code. "
            "Do not overwrite the existing dataset. "
            "Check the CSV header first."
        )

# ==========================================
# 4. MEDIAPIPE SETUP
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

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

# ==========================================
# 5. HAND CONNECTIONS
# ==========================================

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ==========================================
# 6. CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not detected!")

start_time = time.monotonic()
last_timestamp = -1

print("\nISL DATA COLLECTION STARTED")
print("Total gesture classes:", len(gestures))
print("Press S to save a sample")
print("Press N for next gesture")
print("Press P for previous gesture")
print("Press Q to quit\n")

# ==========================================
# 7. DATA COLLECTION
# ==========================================

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

                # Two hands, 63 values each
                hand0 = [0.0] * 63
                hand1 = [0.0] * 63

                height, width, _ = frame.shape

                if result.hand_landmarks:

                    for index, hand in enumerate(
                        result.hand_landmarks[:2]
                    ):

                        points = []

                        values = [
                            value
                            for landmark in hand
                            for value in (
                                landmark.x,
                                landmark.y,
                                landmark.z
                            )
                        ]

                        if index == 0:
                            hand0 = values
                        else:
                            hand1 = values

                        for landmark in hand:

                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            points.append((x, y))

                            cv2.circle(
                                frame,
                                (x, y),
                                5,
                                (0, 255, 0),
                                -1
                            )

                        for start, end in connections:

                            cv2.line(
                                frame,
                                points[start],
                                points[end],
                                (255, 0, 0),
                                2
                            )

                # ==================================
                # 8. DISPLAY INFORMATION
                # ==================================

                progress = (
                    f"Class: {gesture_index + 1}/{len(gestures)}"
                )

                cv2.putText(
                    frame,
                    f"GESTURE: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    progress,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Saved: {saved_counts[label]}/{SAMPLES_PER_CLASS}",
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
                    0.5,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "ISL Complete Dataset Collection",
                    frame
                )

                # ==================================
                # 9. KEYBOARD CONTROLS
                # ==================================

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):

                    if not result.hand_landmarks:
                        print("No hand detected. Try again.")
                        continue

                    if saved_counts[label] >= SAMPLES_PER_CLASS:
                        print("This class already has enough samples.")
                        continue

                    features = hand0 + hand1

                    writer.writerow( [label] + features )
                    file.flush()

                    saved_counts[label] += 1

                    print(
                        f"Saved {label}: "
                        f"{saved_counts[label]}/{SAMPLES_PER_CLASS}"
                    )

                elif key == ord("n"):

                    if gesture_index < len(gestures) - 1:

                        gesture_index += 1
                        label = gestures[gesture_index]

                        print("\nNow collecting:", label)

                    else:
                        print("All gesture classes completed!")

                elif key == ord("p"):

                    if gesture_index > 0:

                        gesture_index -= 1
                        label = gestures[gesture_index]

                        print("\nBack to:", label)

                elif key == ord("q"):
                    break

finally:

    camera.release()
    cv2.destroyAllWindows()

# ==========================================
# 10. COLLECTION SUMMARY
# ==========================================

print("\nDATA COLLECTION STOPPED")

for gesture, count in saved_counts.items():
    print(f"{gesture}: {count} samples")

print("\nDataset saved at:")
print(CSV_FILE)