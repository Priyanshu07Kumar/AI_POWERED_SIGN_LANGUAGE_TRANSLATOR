import cv2
import mediapipe as mp
import joblib
import pandas as pd
import time
import threading
import queue
import customtkinter as ctk
import pyttsx3

from pathlib import Path
from collections import deque, Counter
from PIL import Image, ImageTk
from tkinter import messagebox


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "isl_model.pkl"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

GESTURES = [
    "HELLO", "YES", "NO", "THANK_YOU", "PLEASE",
    "HELP", "GOOD", "BAD", "LOVE", "STOP"
]

print("Starting SignAI application...")


# =====================================================
# 2. LOAD TRAINED AI MODEL
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError("isl_model.pkl is missing!")

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError("hand_landmarker.task is missing!")

model = joblib.load(MODEL_PATH)

FEATURE_NAMES = list(model.feature_names_in_)

if len(FEATURE_NAMES) != 126:
    raise ValueError("Expected exactly 126 model features.")

print("AI model loaded successfully!")


# =====================================================
# 3. MEDIAPIPE CONFIGURATION
# =====================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MEDIAPIPE_OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# =====================================================
# 4. APPLICATION CLASS
# =====================================================

class SignAI(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title("SignAI | Indian Sign Language Recognition")
        self.geometry("1250x800")
        self.minsize(1000, 680)

        self.running = False
        self.stop_event = threading.Event()

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()

        self.current_gesture = "WAITING"
        self.current_confidence = 0.0

        self.sentence = []

        self.photo = None

        self.create_interface()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

        self.after(30, self.update_interface)

        print("GUI initialized successfully!")

    # =================================================
    # 5. DESIGN HELPERS
    # =================================================

    def create_button(self, parent, text, command, color):

        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=42,
            corner_radius=9,
            fg_color=color,
            hover_color=color,
            font=("Arial", 12, "bold")
        )

    def create_card(self, parent, heading, value):

        card = ctk.CTkFrame(
            parent,
            fg_color="#19263A",
            corner_radius=14
        )

        ctk.CTkLabel(
            card,
            text=heading,
            font=("Arial", 12, "bold"),
            text_color="#94A3B8"
        ).pack(anchor="w", padx=15, pady=(12, 5))

        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=("Arial", 21, "bold"),
            text_color="#38BDF8"
        )

        value_label.pack(anchor="w", padx=15, pady=(0, 12))

        return card, value_label

    # =================================================
    # 6. MAIN INTERFACE
    # =================================================

    def create_interface(self):

        self.configure(fg_color="#0B1220")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # SIDEBAR

        sidebar = ctk.CTkFrame(
            self,
            width=220,
            corner_radius=0,
            fg_color="#111C2E"
        )

        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="SignAI",
            font=("Arial", 32, "bold"),
            text_color="#38BDF8"
        ).pack(pady=(35, 3))

        ctk.CTkLabel(
            sidebar,
            text="ISL RECOGNITION SYSTEM",
            font=("Arial", 10, "bold"),
            text_color="#94A3B8"
        ).pack(pady=(0, 35))

        self.create_button(
            sidebar, "DASHBOARD",
            self.show_dashboard, "#1E3A5F"
        ).pack(fill="x", padx=15, pady=7)

        self.create_button(
            sidebar, "LIVE DETECTION",
            self.start_detection, "#1E3A5F"
        ).pack(fill="x", padx=15, pady=7)

        self.create_button(
            sidebar, "STOP DETECTION",
            self.stop_detection, "#7F1D1D"
        ).pack(fill="x", padx=15, pady=7)

        self.create_button(
            sidebar, "ABOUT PROJECT",
            self.show_about, "#1E3A5F"
        ).pack(fill="x", padx=15, pady=7)

        ctk.CTkLabel(
            sidebar,
            text="BCA FINAL YEAR PROJECT\nAI AND COMPUTER VISION",
            font=("Arial", 10),
            text_color="#64748B"
        ).pack(side="bottom", pady=25)

        # MAIN CONTENT

        main = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        main.grid(
            row=0, column=1,
            sticky="nsew", padx=22, pady=18
        )

        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        # HEADER

        header = ctk.CTkFrame(
            main,
            fg_color="transparent"
        )

        header.grid(row=0, column=0, sticky="ew", pady=10)

        ctk.CTkLabel(
            header,
            text="AI SIGN LANGUAGE DASHBOARD",
            font=("Arial", 23, "bold"),
            text_color="white"
        ).pack(side="left")

        self.status_label = ctk.CTkLabel(
            header,
            text="● SYSTEM READY",
            text_color="#22C55E",
            font=("Arial", 12, "bold")
        )

        self.status_label.pack(side="right")

        # STATISTICS

        stats = ctk.CTkFrame(
            main,
            fg_color="transparent"
        )

        stats.grid(row=1, column=0, sticky="ew", pady=10)

        stats.grid_columnconfigure(
            (0, 1, 2),
            weight=1
        )

        card, self.gesture_value = self.create_card(
            stats, "DETECTED GESTURE", "WAITING"
        )
        card.grid(row=0, column=0, padx=5, sticky="ew")

        card, self.confidence_value = self.create_card(
            stats, "MODEL SCORE", "0%"
        )
        card.grid(row=0, column=1, padx=5, sticky="ew")

        card, self.model_value = self.create_card(
            stats, "AI MODEL", "LOADED"
        )
        card.grid(row=0, column=2, padx=5, sticky="ew")

        # CAMERA AND RECOGNITION

        center = ctk.CTkFrame(
            main,
            fg_color="transparent"
        )

        center.grid(
            row=2, column=0,
            sticky="nsew", pady=10
        )

        center.grid_columnconfigure(0, weight=3)
        center.grid_columnconfigure(1, weight=2)
        center.grid_rowconfigure(0, weight=1)

        # CAMERA PANEL

        camera_card = ctk.CTkFrame(
            center,
            fg_color="#19263A",
            corner_radius=14
        )

        camera_card.grid(
            row=0, column=0,
            sticky="nsew", padx=(0, 8)
        )

        ctk.CTkLabel(
            camera_card,
            text="LIVE CAMERA FEED",
            font=("Arial", 15, "bold")
        ).pack(anchor="w", padx=15, pady=15)

        self.camera_label = ctk.CTkLabel(
            camera_card,
            text="Camera is currently inactive",
            fg_color="#101827",
            corner_radius=10,
            font=("Arial", 16)
        )

        self.camera_label.pack(
            fill="both", expand=True,
            padx=15, pady=10
        )

        camera_buttons = ctk.CTkFrame(
            camera_card,
            fg_color="transparent"
        )

        camera_buttons.pack(pady=12)

        self.create_button(
            camera_buttons, "START CAMERA",
            self.start_detection, "#15803D"
        ).grid(row=0, column=0, padx=5)

        self.create_button(
            camera_buttons, "STOP CAMERA",
            self.stop_detection, "#B91C1C"
        ).grid(row=0, column=1, padx=5)

        # RECOGNITION PANEL

        result_card = ctk.CTkFrame(
            center,
            fg_color="#19263A",
            corner_radius=14
        )

        result_card.grid(
            row=0, column=1,
            sticky="nsew", padx=(8, 0)
        )

        ctk.CTkLabel(
            result_card,
            text="RECOGNITION RESULT",
            font=("Arial", 15, "bold")
        ).pack(pady=20)

        self.result_label = ctk.CTkLabel(
            result_card,
            text="WAITING",
            font=("Arial", 29, "bold"),
            text_color="#38BDF8",
            wraplength=230
        )

        self.result_label.pack(pady=25)

        self.progress = ctk.CTkProgressBar(
            result_card,
            progress_color="#0EA5E9"
        )

        self.progress.pack(fill="x", padx=20)
        self.progress.set(0)

        self.score_label = ctk.CTkLabel(
            result_card,
            text="Model score: 0%",
            text_color="#94A3B8"
        )

        self.score_label.pack(pady=10)

        ctk.CTkLabel(
            result_card,
            text="SUPPORTED GESTURES",
            font=("Arial", 12, "bold"),
            text_color="#94A3B8"
        ).pack(pady=(25, 10))

        ctk.CTkLabel(
            result_card,
            text="HELLO  |  YES  |  NO\n"
                 "THANK YOU  |  PLEASE\n"
                 "HELP  |  GOOD  |  BAD\n"
                 "LOVE  |  STOP",
            font=("Arial", 12),
            justify="center"
        ).pack(pady=5)

        # SENTENCE BUILDER

        sentence_card = ctk.CTkFrame(
            main,
            fg_color="#19263A",
            corner_radius=14
        )

        sentence_card.grid(
            row=3, column=0,
            sticky="ew", pady=10
        )

        ctk.CTkLabel(
            sentence_card,
            text="SENTENCE BUILDER",
            font=("Arial", 15, "bold")
        ).pack(anchor="w", padx=15, pady=12)

        self.sentence_box = ctk.CTkTextbox(
            sentence_card,
            height=65,
            font=("Arial", 14),
            fg_color="#101827"
        )

        self.sentence_box.pack(
            fill="x", padx=15, pady=5
        )

        sentence_buttons = ctk.CTkFrame(
            sentence_card,
            fg_color="transparent"
        )

        sentence_buttons.pack(pady=12)

        self.create_button(
            sentence_buttons, "ADD GESTURE",
            self.add_gesture, "#2563EB"
        ).grid(row=0, column=0, padx=5)

        self.create_button(
            sentence_buttons, "SPEAK SENTENCE",
            self.speak_sentence, "#7C3AED"
        ).grid(row=0, column=1, padx=5)

        self.create_button(
            sentence_buttons, "CLEAR",
            self.clear_sentence, "#475569"
        ).grid(row=0, column=2, padx=5)

    # =================================================
    # 7. CAMERA DETECTION THREAD
    # =================================================

    def start_detection(self):

        if self.running:
            return

        self.running = True
        self.stop_event.clear()

        self.status_label.configure(
            text="● STARTING CAMERA",
            text_color="#FACC15"
        )

        self.worker = threading.Thread(
            target=self.camera_worker,
            daemon=True
        )

        self.worker.start()

    def camera_worker(self):

        camera = None
        detector = None

        try:

            camera = cv2.VideoCapture(0)

            if not camera.isOpened():
                raise RuntimeError(
                    "Camera could not be opened."
                )

            detector = HandLandmarker.create_from_options(
                MEDIAPIPE_OPTIONS
            )

            start_time = time.monotonic()
            last_timestamp = -1

            predictions = deque(maxlen=7)

            self.result_queue.put(
                ("status", "RUNNING")
            )

            while not self.stop_event.is_set():

                success, frame = camera.read()

                if not success:
                    raise RuntimeError(
                        "Unable to read webcam frame."
                    )

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )

                timestamp = max(
                    timestamp, last_timestamp + 1
                )

                last_timestamp = timestamp

                result = detector.detect_for_video(
                    mp_image, timestamp
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
                                4,
                                (0, 255, 120),
                                -1
                            )

                        for start, end in CONNECTIONS:

                            cv2.line(
                                frame,
                                points[start],
                                points[end],
                                (0, 180, 255),
                                2
                            )

                    data = pd.DataFrame(
                        [hand0 + hand1],
                        columns=FEATURE_NAMES
                    )

                    probabilities = model.predict_proba(
                        data
                    )[0]

                    best_index = probabilities.argmax()
                    score = float(probabilities[best_index])

                    if score >= 0.55:

                        predictions.append(
                            str(model.classes_[best_index])
                        )

                        gesture = Counter(
                            predictions
                        ).most_common(1)[0][0]

                    else:

                        predictions.clear()
                        gesture = "Uncertain"

                else:

                    predictions.clear()
                    gesture = "Show your hand"
                    score = 0.0

                self.result_queue.put(
                    ("prediction", gesture, score)
                )

                # Keep only latest webcam frame

                if self.frame_queue.full():

                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put_nowait(
                        cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        )
                    )
                except queue.Full:
                    pass

        except Exception as error:

            self.result_queue.put(
                ("error", str(error))
            )

        finally:

            if detector is not None:
                detector.close()

            if camera is not None:
                camera.release()

            self.result_queue.put(
                ("status", "STOPPED")
            )

    # =================================================
    # 8. UPDATE GUI
    # =================================================

    def update_interface(self):

        try:

            while True:

                item = self.result_queue.get_nowait()

                if item[0] == "prediction":

                    gesture = item[1]
                    score = item[2]

                    self.current_gesture = gesture
                    self.current_confidence = score

                    percentage = score * 100

                    self.result_label.configure(
                        text=gesture
                    )

                    self.gesture_value.configure(
                        text=gesture
                    )

                    self.confidence_value.configure(
                        text=f"{percentage:.1f}%"
                    )

                    self.score_label.configure(
                        text=f"Model score: {percentage:.1f}%"
                    )

                    self.progress.set(score)

                elif item[0] == "status":

                    if item[1] == "RUNNING":

                        self.status_label.configure(
                            text="● CAMERA ACTIVE",
                            text_color="#22C55E"
                        )

                    else:

                        self.status_label.configure(
                            text="● SYSTEM READY",
                            text_color="#22C55E"
                        )

                elif item[0] == "error":

                    self.running = False
                    self.stop_event.set()

                    self.status_label.configure(
                        text="● DETECTION ERROR",
                        text_color="#EF4444"
                    )

                    messagebox.showerror(
                        "Detection Error",
                        item[1]
                    )

        except queue.Empty:
            pass

        try:

            frame = self.frame_queue.get_nowait()

            image = Image.fromarray(frame)
            image.thumbnail((720, 430))

            self.photo = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=image.size
            )

            self.camera_label.configure(
                image=self.photo,
                text=""
            )

        except queue.Empty:
            pass

        if self.winfo_exists():
            self.after(30, self.update_interface)

    # =================================================
    # 9. STOP DETECTION
    # =================================================

    def stop_detection(self):

        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        self.status_label.configure(
            text="● STOPPING CAMERA",
            text_color="#FACC15"
        )

        self.result_label.configure(text="WAITING")
        self.gesture_value.configure(text="WAITING")
        self.confidence_value.configure(text="0%")
        self.score_label.configure(text="Model score: 0%")
        self.progress.set(0)

        self.camera_label.configure(
            image=None,
            text="Camera is currently inactive"
        )

        self.photo = None

    # =================================================
    # 10. SENTENCE BUILDER
    # =================================================

    def add_gesture(self):

        if self.current_gesture not in GESTURES:
            messagebox.showwarning(
                "No Gesture",
                "Show a recognized gesture first."
            )
            return

        self.sentence.append(
            self.current_gesture
        )

        self.update_sentence()

    def update_sentence(self):

        self.sentence_box.delete("1.0", "end")

        self.sentence_box.insert(
            "1.0",
            " ".join(self.sentence)
        )

    def clear_sentence(self):

        self.sentence.clear()
        self.update_sentence()

    # =================================================
    # 11. TEXT TO SPEECH
    # =================================================

    def speak_sentence(self):

        text = " ".join(self.sentence)

        if not text:
            messagebox.showwarning(
                "Empty Sentence",
                "Add gestures first."
            )
            return

        def speak():

            try:

                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()

            except Exception as error:
                print("Speech error:", error)

        threading.Thread(
            target=speak,
            daemon=True
        ).start()

    # =================================================
    # 12. NAVIGATION
    # =================================================

    def show_dashboard(self):

        messagebox.showinfo(
            "SignAI Dashboard",
            "Welcome to the SignAI dashboard."
        )

    def show_about(self):

        messagebox.showinfo(
            "About Project",
            "AI-Powered Indian Sign Language Detection\n\n"
            "Technologies:\n"
            "Python, OpenCV, MediaPipe,\n"
            "Random Forest and Tkinter.\n\n"
            "This prototype recognizes 10 trained gestures."
        )

    # =================================================
    # 13. CLOSE APPLICATION
    # =================================================

    def close_application(self):

        self.running = False
        self.stop_event.set()

        self.destroy()


# =====================================================
# 14. RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    try:

        print("Initializing GUI...")

        app = SignAI()

        print("SignAI started successfully!")

        app.mainloop()

    except Exception as error:

        print("\nAPPLICATION ERROR:")
        print(repr(error))

        raise
    