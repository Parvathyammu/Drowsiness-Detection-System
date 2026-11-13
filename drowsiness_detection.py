
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading
from collections import deque

# --------------------------------
# MODEL PATH
# --------------------------------

MODEL_PATH = "models/face_landmarker.task"

# --------------------------------
# FACIAL LANDMARK INDICES
# --------------------------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MOUTH_POINTS = [13, 14, 78, 308, 82, 87, 312, 317]

# --------------------------------
# DETECTION THRESHOLDS
# --------------------------------

EAR_THRESHOLD = 0.22
DROWSY_SECONDS = 2.0

MAR_THRESHOLD = 0.40
MAR_RELEASE_THRESHOLD = 0.35
YAWN_SECONDS = 1.5
YAWN_RELEASE_DELAY = 0.5

# --------------------------------
# BLINK RATE DETECTION
# --------------------------------

blink_count = 0
eye_was_closed = False
blink_times = deque()
blink_rate = 0

# --------------------------------
# PERCLOS DETECTION
# --------------------------------

PERCLOS_WINDOW = 60
eye_state_history = deque()

perclos = 0.0
closed_eye_time = 0.0
observed_eye_time = 0.0

MAX_SAMPLE_GAP = 0.2

# --------------------------------
# DEEP DOUBLE-TONE ALARM
# --------------------------------

def deep_alarm():

    for _ in range(4):

        winsound.Beep(350, 400)
        winsound.Beep(250, 500)

        time.sleep(0.15)


# --------------------------------
# CALCULATE EAR
# --------------------------------

def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for index in eye_points:

        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)

        points.append((x, y))

    A = np.linalg.norm(
        np.array(points[1]) - np.array(points[5])
    )

    B = np.linalg.norm(
        np.array(points[2]) - np.array(points[4])
    )

    C = np.linalg.norm(
        np.array(points[0]) - np.array(points[3])
    )

    if C == 0:
        return 0.0, points

    ear = (A + B) / (2.0 * C)

    return ear, points


# --------------------------------
# CALCULATE MAR
# --------------------------------

def calculate_mar(landmarks, width, height):

    points = []

    for index in MOUTH_POINTS:

        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)

        points.append((x, y))

    A = np.linalg.norm(
        np.array(points[0]) - np.array(points[1])
    )

    B = np.linalg.norm(
        np.array(points[4]) - np.array(points[5])
    )

    C = np.linalg.norm(
        np.array(points[6]) - np.array(points[7])
    )

    D = np.linalg.norm(
        np.array(points[2]) - np.array(points[3])
    )

    if D == 0:
        return 0.0, points

    mar = (A + B + C) / (2.0 * D)

    return mar, points


# --------------------------------
# CALCULATE PERCLOS
# --------------------------------

def calculate_perclos(current_time):

    global closed_eye_time, observed_eye_time

    cutoff_time = current_time - PERCLOS_WINDOW

    while (
        len(eye_state_history) > 2
        and eye_state_history[1][0] <= cutoff_time
    ):

        eye_state_history.popleft()

    closed_eye_time = 0.0
    observed_eye_time = 0.0

    for i in range(1, len(eye_state_history)):

        previous_time, previous_state = eye_state_history[i - 1]

        current_sample_time, current_state = eye_state_history[i]

        duration = current_sample_time - previous_time

        if duration <= 0 or duration > MAX_SAMPLE_GAP:
            continue

        interval_start = max(previous_time, cutoff_time)

        valid_duration = current_sample_time - interval_start

        if valid_duration <= 0:
            continue

        observed_eye_time += valid_duration

        if previous_state:
            closed_eye_time += valid_duration

    if observed_eye_time > 0:

        perclos = (
            closed_eye_time / observed_eye_time
        ) * 100

    else:

        perclos = 0.0

    return perclos


# --------------------------------
# MEDIAPIPE FACE LANDMARKER
# --------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=1
)


# --------------------------------
# START WEBCAM
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Webcam not found!")
    exit()


start_time = time.monotonic()

closed_start = None
yawn_start = None
yawn_below_threshold_start = None

eye_alarm_triggered = False
yawn_alarm_triggered = False

eye_status = "EYES NORMAL"
yawn_status = "NO YAWNING"

print("Drowsiness Detection Started. Press Q to quit.")


# --------------------------------
# MAIN DETECTION LOOP
# --------------------------------

try:

    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            height, width, _ = frame.shape

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

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            current_time = time.monotonic()

            # --------------------------------
            # FACE DETECTED
            # --------------------------------

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                # --------------------------------
                # CALCULATE EAR
                # --------------------------------

                left_ear, left_points = calculate_ear(
                    landmarks,
                    LEFT_EYE,
                    width,
                    height
                )

                right_ear, right_points = calculate_ear(
                    landmarks,
                    RIGHT_EYE,
                    width,
                    height
                )

                ear = (left_ear + right_ear) / 2

                # --------------------------------
                # BLINK RATE DETECTION
                # --------------------------------

                if ear < EAR_THRESHOLD:

                    if not eye_was_closed:
                        eye_was_closed = True

                else:

                    if eye_was_closed:

                        blink_count += 1
                        blink_times.append(current_time)

                        eye_was_closed = False

                while (
                    blink_times
                    and current_time - blink_times[0] > 60
                ):

                    blink_times.popleft()

                blink_rate = len(blink_times)

                # --------------------------------
                # PERCLOS DETECTION
                # --------------------------------

                eyes_closed = ear < EAR_THRESHOLD

                eye_state_history.append(
                    (current_time, eyes_closed)
                )

                perclos = calculate_perclos(current_time)

                # --------------------------------
                # CALCULATE MAR
                # --------------------------------

                mar, mouth_points = calculate_mar(
                    landmarks,
                    width,
                    height
                )

                # --------------------------------
                # DRAW EYE LANDMARKS
                # --------------------------------

                for point in left_points + right_points:

                    cv2.circle(
                        frame,
                        point,
                        2,
                        (0, 255, 0),
                        -1
                    )

                # --------------------------------
                # DRAW MOUTH LANDMARKS
                # --------------------------------

                for point in mouth_points:

                    cv2.circle(
                        frame,
                        point,
                        2,
                        (255, 0, 255),
                        -1
                    )

                # --------------------------------
                # EYE CLOSURE DETECTION
                # --------------------------------

                if ear < EAR_THRESHOLD:

                    if closed_start is None:
                        closed_start = current_time

                    closed_duration = current_time - closed_start

                    if closed_duration >= DROWSY_SECONDS:

                        eye_status = "DROWSINESS ALERT!"

                        if not eye_alarm_triggered:

                            threading.Thread(
                                target=deep_alarm,
                                daemon=True
                            ).start()

                            eye_alarm_triggered = True

                    else:

                        eye_status = "EYES CLOSING..."

                else:

                    closed_start = None
                    eye_alarm_triggered = False
                    eye_status = "EYES NORMAL"

                # --------------------------------
                # YAWNING DETECTION
                # --------------------------------

                if yawn_start is None:

                    if mar >= MAR_THRESHOLD:

                        yawn_start = current_time
                        yawn_below_threshold_start = None

                else:

                    if mar < MAR_RELEASE_THRESHOLD:

                        if yawn_below_threshold_start is None:

                            yawn_below_threshold_start = current_time

                        elif (
                            current_time - yawn_below_threshold_start
                            >= YAWN_RELEASE_DELAY
                        ):

                            yawn_start = None
                            yawn_below_threshold_start = None
                            yawn_alarm_triggered = False

                    else:

                        yawn_below_threshold_start = None

                if yawn_start is not None:

                    yawn_duration = current_time - yawn_start

                    if yawn_duration >= YAWN_SECONDS:

                        yawn_status = "YAWNING ALERT!"

                        if not yawn_alarm_triggered:

                            threading.Thread(
                                target=deep_alarm,
                                daemon=True
                            ).start()

                            yawn_alarm_triggered = True

                    else:

                        yawn_status = "MOUTH OPENING..."

                else:

                    yawn_status = "NO YAWNING"

                # --------------------------------
                # DISPLAY EAR
                # --------------------------------

                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # --------------------------------
                # DISPLAY MAR
                # --------------------------------

                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f} (start: {MAR_THRESHOLD:.2f})",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # --------------------------------
                # DISPLAY BLINK INFORMATION
                # --------------------------------

                cv2.putText(
                    frame,
                    f"Blink Count: {blink_count}",
                    (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Blink Rate: {blink_rate} / min",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # --------------------------------
                # DISPLAY PERCLOS
                # --------------------------------

                cv2.putText(
                    frame,
                    f"PERCLOS: {perclos:.1f}%",
                    (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if perclos >= 40
                    else (0, 255, 255),
                    2
                )

                # --------------------------------
                # DISPLAY EYE STATUS
                # --------------------------------

                cv2.putText(
                    frame,
                    eye_status,
                    (30, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if "ALERT" in eye_status
                    else (0, 255, 0),
                    2
                )

                # --------------------------------
                # DISPLAY YAWN STATUS
                # --------------------------------

                cv2.putText(
                    frame,
                    yawn_status,
                    (30, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if "ALERT" in yawn_status
                    else (0, 255, 0),
                    2
                )

                # --------------------------------
                # EYE CLOSURE TIMER
                # --------------------------------

                if closed_start is not None:

                    duration = current_time - closed_start

                    cv2.putText(
                        frame,
                        f"Eyes Closed: {duration:.1f}s",
                        (30, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                # --------------------------------
                # YAWNING TIMER
                # --------------------------------

                if yawn_start is not None:

                    duration = current_time - yawn_start

                    cv2.putText(
                        frame,
                        f"Yawning: {duration:.1f}s",
                        (30, 335),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2
                    )

            # --------------------------------
            # NO FACE DETECTED
            # --------------------------------

            else:

                closed_start = None
                yawn_start = None
                yawn_below_threshold_start = None

                eye_alarm_triggered = False
                yawn_alarm_triggered = False

                eye_was_closed = False

                eye_state_history.clear()
                closed_eye_time = 0.0
                observed_eye_time = 0.0
                perclos = 0.0

                eye_status = "EYES NORMAL"
                yawn_status = "NO YAWNING"

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # --------------------------------
            # DISPLAY WEBCAM
            # --------------------------------

            cv2.imshow(
                "Drowsiness Detection System",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:

    # --------------------------------
    # RELEASE RESOURCES
    # --------------------------------

    cap.release()
    cv2.destroyAllWindows()

    print("Drowsiness Detection Stopped.")
