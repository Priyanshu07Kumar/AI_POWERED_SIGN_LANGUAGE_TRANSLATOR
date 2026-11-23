import cv2
import mediapipe as mp
import joblib
import pandas as pd
import time

from pathlib import Path
from collections import deque, Counter
import numpy as np

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "hand_landmarker.task"

CLASSIFIER_PATH = BASE_DIR / "isl_normalized_model.pkl"

# =====================================
# 2. LOAD NORMALIZED AI MODEL
# =====================================

print("Loading normalized ISL AI model...")

model = joblib.load(CLASSIFIER_PATH)

feature_names = list(model.feature_names_in_)

if len(feature_names) != 126:
    raise ValueError("Expected 126 model features!")

print("AI model loaded successfully!")
print("Features:", len(feature_names))
print("Gestures:", model.classes_)

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
# 4. HAND CONNECTIONS
# =====================================

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# =====================================
# 5. NORMALIZATION FUNCTION
# =====================================

def normalize_hand(hand):

    points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand
        ],
        dtype=float
    )

    # Same normalization as training dataset
    relative = points - points[0]

    distances = np.linalg.norm(relative, axis=1)

    scale = np.max(distances)

    if scale > 1e-8:
        relative = relative / scale

    return relative.flatten().tolist()


# =====================================
# 6. START WEBCAM
# =====================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not detected!")

start_time = time.monotonic()
last_timestamp = -1

predictions = deque(maxlen=7)

print("Starting normalized ISL recognition...")
print("Press Q to exit.")

# =====================================
# 7. LIVE DETECTION
# =====================================

try:

    with HandLandmarker.create_from_options(options) as detector:

        while True:

            success, frame = camera.read()

            if not success:
                print("Camera frame unavailable!")
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

            # Monotonically increasing timestamp
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

            # =================================
            # 8. EXTRACT AND NORMALIZE FEATURES
            # =================================

            hand0 = [0.0] * 63
            hand1 = [0.0] * 63

            if result.hand_landmarks:

                for index, hand in enumerate(
                    result.hand_landmarks[:2]
                ):

                    normalized_features = normalize_hand(hand)

                    if index == 0:
                        hand0 = normalized_features
                    else:
                        hand1 = normalized_features

                    # Draw original hand landmarks
                    height, width, _ = frame.shape

                    points = []

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

            # =================================
            # 9. AI GESTURE PREDICTION
            # =================================

            if result.hand_landmarks:

                all_features = hand0 + hand1

                input_data = pd.DataFrame(
                    [all_features],
                    columns=feature_names
                )

                scores = model.predict_proba(
                    input_data
                )[0]

                best_index = scores.argmax()

                confidence = scores[best_index]

                predicted_gesture = model.classes_[best_index]

                if confidence >= 0.55:

                    predictions.append(predicted_gesture)

                    final_prediction = Counter(
                        predictions
                    ).most_common(1)[0][0]

                else:

                    predictions.clear()
                    final_prediction = "Uncertain"

            else:

                predictions.clear()
                final_prediction = "Show your hand"
                confidence = 0.0

            # =================================
            # 10. DISPLAY RESULT
            # =================================

            cv2.rectangle(
                frame,
                (10, 10),
                (frame.shape[1] - 10, 110),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                frame,
                "NORMALIZED ISL AI RECOGNITION",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Gesture: " + str(final_prediction),
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            if result.hand_landmarks:

                cv2.putText(
                    frame,
                    f"Model score: {confidence * 100:.1f}%",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                "AI Powered Indian Sign Language Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

print("Live ISL recognition stopped.")