import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("AI Eye Fatigue Detection System")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20

blink_count = 0
blink_detected = False


def calculate_EAR(eye_points, landmarks, w, h):

    points = []

    for point in eye_points:
        landmark = landmarks[point]

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        points.append((x, y))

    # Vertical distances
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

    # Horizontal distance
    h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (v1 + v2) / (2.0 * h1)

    return ear


while run:

    ret, frame = cap.read()

    if not ret:
        st.write("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    fatigue_status = "Normal"
    fatigue_level = "Low"

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            left_ear = calculate_EAR(LEFT_EYE, landmarks, w, h)
            right_ear = calculate_EAR(RIGHT_EYE, landmarks, w, h)

            avg_ear = (left_ear + right_ear) / 2

            # Fatigue Level Logic
            if avg_ear > 0.30:
                fatigue_level = "Low"

            elif avg_ear > 0.22:
                fatigue_level = "Medium"

            else:
                fatigue_level = "High"

            # Blink Detection
            if avg_ear < EAR_THRESHOLD:

                fatigue_status = "Fatigue Detected"

                if not blink_detected:
                    blink_count += 1
                    blink_detected = True

            else:
                fatigue_status = "Normal"
                blink_detected = False

            # Fatigue Status
            cv2.putText(
                frame,
                fatigue_status,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Blink Counter
            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            # Fatigue Level
            cv2.putText(
                frame,
                f"Fatigue Level: {fatigue_level}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()