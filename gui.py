import cv2
import mediapipe as mp
import joblib
import pandas as pd
import time
import threading
import tkinter as tk

from pathlib import Path
from collections import deque, Counter
from PIL import Image, ImageTk
from tkinter import messagebox
import pyttsx3


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "isl_model.pkl"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

model = joblib.load(MODEL_PATH)
feature_names = list(model.feature_names_in_)

if len(feature_names) != 126:
    raise ValueError("Model must contain 126 features!")


# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ==========================================
# 3. GUI WINDOW
# ==========================================

root = tk.Tk()
root.title("AI Powered Indian Sign Language Detection")
root.geometry("1100x720")
root.configure(bg="#101827")

camera = None
detector = None
running = False

start_time = 0
last_timestamp = -1

predictions = deque(maxlen=7)

current_gesture = "Show your hand"
current_confidence = 0.0
sentence = []


# ==========================================
# 4. GUI FUNCTIONS
# ==========================================

def start_detection():

    global camera, detector, running
    global start_time, last_timestamp

    if running:
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        camera = None
        messagebox.showerror(
            "Camera Error",
            "Camera could not be opened!"
        )
        return

    detector = HandLandmarker.create_from_options(options)

    start_time = time.monotonic()
    last_timestamp = -1

    predictions.clear()
    running = True

    status_label.config(
        text="● Detection Running",
        fg="#22c55e"
    )

    update_frame()


def stop_detection():

    global camera, detector, running

    running = False

    if camera is not None:
        camera.release()
        camera = None

    if detector is not None:
        detector.close()
        detector = None

    video_label.config(
        image="",
        text="Camera Stopped",
        fg="white"
    )

    video_label.image = None

    status_label.config(
        text="● Detection Stopped",
        fg="#ef4444"
    )


def update_frame():

    global last_timestamp
    global current_gesture, current_confidence

    if not running or camera is None:
        return

    success, frame = camera.read()

    if not success:
        root.after(20, update_frame)
        return

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
        mp_image,
        timestamp
    )

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

        all_features = hand0 + hand1

        input_data = pd.DataFrame(
            [all_features],
            columns=feature_names
        )

        probabilities = model.predict_proba(input_data)[0]

        best_index = probabilities.argmax()
        current_confidence = probabilities[best_index]

        predicted = model.classes_[best_index]

        if current_confidence >= 0.55:

            predictions.append(predicted)

            current_gesture = Counter(
                predictions
            ).most_common(1)[0][0]

        else:
            predictions.clear()
            current_gesture = "Uncertain"

    else:
        predictions.clear()
        current_gesture = "Show your hand"
        current_confidence = 0.0

    gesture_label.config(
        text=current_gesture
    )

    confidence_label.config(
        text=f"Model score: {current_confidence * 100:.1f}%"
    )

    # Convert webcam frame for Tkinter

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(frame)
    image.thumbnail((700, 450))

    photo = ImageTk.PhotoImage(image=image)

    video_label.config(
        image=photo,
        text=""
    )

    video_label.image = photo

    root.after(15, update_frame)


def add_gesture():

    if current_gesture in [
        "Show your hand",
        "Uncertain"
    ]:
        messagebox.showwarning(
            "No Gesture",
            "Please show a recognized gesture first."
        )
        return

    sentence.append(current_gesture)

    sentence_label.config(
        text=" ".join(sentence)
    )


def clear_sentence():

    sentence.clear()

    sentence_label.config(
        text="Your sentence will appear here..."
    )


def speak_sentence():

    text_to_speak = " ".join(sentence)

    if not text_to_speak:
        messagebox.showwarning(
            "Empty Sentence",
            "Add some gestures first!"
        )
        return

    def speak():
        try:
            engine = pyttsx3.init()
            engine.say(text_to_speak)
            engine.runAndWait()
            engine.stop()
        except Exception as error:
            print("Speech error:", error)

    threading.Thread(
        target=speak,
        daemon=True
    ).start()


def close_application():

    stop_detection()
    root.destroy()


# ==========================================
# 5. GUI DESIGN
# ==========================================

title = tk.Label(
    root,
    text="AI POWERED INDIAN SIGN LANGUAGE DETECTION",
    font=("Arial", 19, "bold"),
    bg="#101827",
    fg="#38bdf8"
)

title.pack(pady=18)

status_label = tk.Label(
    root,
    text="● Detection Stopped",
    font=("Arial", 11, "bold"),
    bg="#101827",
    fg="#ef4444"
)

status_label.pack()

video_label = tk.Label(
    root,
    text="Click START DETECTION",
    font=("Arial", 16),
    bg="#263449",
    fg="white",
    width=75,
    height=18
)

video_label.pack(pady=15)

gesture_label = tk.Label(
    root,
    text="Show your hand",
    font=("Arial", 25, "bold"),
    bg="#101827",
    fg="#22c55e"
)

gesture_label.pack(pady=5)

confidence_label = tk.Label(
    root,
    text="Model score: 0%",
    font=("Arial", 12),
    bg="#101827",
    fg="white"
)

confidence_label.pack(pady=5)


# ==========================================
# 6. BUTTONS
# ==========================================

button_frame = tk.Frame(
    root,
    bg="#101827"
)

button_frame.pack(pady=15)


def make_button(parent, text, command, color):

    return tk.Button(
        parent,
        text=text,
        command=command,
        font=("Arial", 11, "bold"),
        bg=color,
        fg="white",
        activebackground=color,
        activeforeground="white",
        width=17,
        pady=9,
        relief="flat",
        cursor="hand2"
    )


make_button(
    button_frame,
    "START DETECTION",
    start_detection,
    "#16a34a"
).grid(row=0, column=0, padx=7)

make_button(
    button_frame,
    "STOP DETECTION",
    stop_detection,
    "#dc2626"
).grid(row=0, column=1, padx=7)

make_button(
    button_frame,
    "ADD GESTURE",
    add_gesture,
    "#2563eb"
).grid(row=0, column=2, padx=7)


# ==========================================
# 7. SENTENCE SECTION
# ==========================================

sentence_title = tk.Label(
    root,
    text="RECOGNIZED SENTENCE",
    font=("Arial", 14, "bold"),
    bg="#101827",
    fg="#38bdf8"
)

sentence_title.pack(pady=8)

sentence_label = tk.Label(
    root,
    text="Your sentence will appear here...",
    font=("Arial", 15),
    bg="#263449",
    fg="white",
    wraplength=950,
    pady=12
)

sentence_label.pack(
    fill="x",
    padx=30
)

sentence_buttons = tk.Frame(
    root,
    bg="#101827"
)

sentence_buttons.pack(pady=15)

make_button(
    sentence_buttons,
    "SPEAK SENTENCE",
    speak_sentence,
    "#7c3aed"
).grid(row=0, column=0, padx=8)

make_button(
    sentence_buttons,
    "CLEAR SENTENCE",
    clear_sentence,
    "#475569"
).grid(row=0, column=1, padx=8)


# ==========================================
# 8. START APPLICATION
# ==========================================

root.protocol(
    "WM_DELETE_WINDOW",
    close_application
)

root.mainloop()