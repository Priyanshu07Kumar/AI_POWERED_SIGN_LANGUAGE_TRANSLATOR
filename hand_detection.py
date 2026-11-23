import cv2
import mediapipe as mp
import time

# --------------------------------
# 1. Load MediaPipe Hand Landmarker
# --------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# --------------------------------
# 2. Hand Landmark Connections
# --------------------------------

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# --------------------------------
# 3. Start Webcam
# --------------------------------

print("Starting AI Hand Detection...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera not detected!")
    exit()

# --------------------------------
# 4. Start Detection
# --------------------------------

start_time = time.monotonic()
last_timestamp = -1

try:

    with HandLandmarker.create_from_options(options) as detector:

        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to capture camera frame")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate increasing timestamps
            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp = max(
                timestamp,
                last_timestamp + 1
            )

            last_timestamp = timestamp

            # Detect hand
            result = detector.detect_for_video(
                mp_image,
                timestamp
            )

            # --------------------------------
            # 5. Draw Hand Landmarks
            # --------------------------------

            if result.hand_landmarks:

                for hand in result.hand_landmarks:

                    height, width, _ = frame.shape
                    points = []

                    # Draw 21 landmark points
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

                    # Connect landmark points
                    for start, end in connections:

                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (255, 0, 0),
                            2
                        )

                cv2.putText(
                    frame,
                    "HAND DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "SHOW YOUR HAND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # Display webcam
            cv2.imshow(
                "AI Sign Language Detection",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

print("AI Hand Detection Stopped")
