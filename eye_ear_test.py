
import cv2
import mediapipe as mp
import numpy as np
import time

MODEL_PATH = "models/face_landmarker.task"

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22

def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for index in eye_points:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append((x, y))

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (A + B) / (2.0 * C)

    return ear, points


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

cap = cv2.VideoCapture(0)
start_time = time.monotonic()

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.monotonic() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            left_ear, left_points = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )

            right_ear, right_points = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            ear = (left_ear + right_ear) / 2.0

            for point in left_points + right_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            if ear < EAR_THRESHOLD:
                status = "EYES CLOSED"
                color = (0, 0, 255)
            else:
                status = "EYES OPEN"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                status,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        else:
            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Eye Aspect Ratio Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
