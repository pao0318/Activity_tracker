import cv2
import mediapipe as mp

# importing the requests library
import requests
import numpy as np
from IPython.display import display
import time
from calculateAngle import *
from tolerance import *

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)


global cap, frame, switch, counter, capture
switch = 1
cap = cv2.VideoCapture(0)
capture = 0
counter = 0
params = {"counter": counter}


def generate_frames1():
    tol_angle = get_tolerance("mild")
    stage = None
    starttime = time.time()
    lasttime = starttime
    flag = None
    counter = 0

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            else:
                laptime = round((time.time() - lasttime), 2)
                # Recolor feed and making it immutable array
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False

                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                #         Extracting landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [
                        landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y,
                    ]
                    elbow = [
                        landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y,
                    ]
                    wrist = [
                        landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y,
                    ]

                    # Calculate angle
                    angle = calculateAngle(shoulder, elbow, wrist)

                    # Visualize angle
                    cv2.putText(
                        image,
                        str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    #         Angles measurement
                    if angle > 160 - (tol_angle):
                        stage = "down"
                        lapperdown = time.time()

                    if angle < 30 + (tol_angle) and stage == "down":
                        stage = "up"
                        counter += 1
                        lapperup = time.time()

                    #          Error Analysis
                    if angle > 120 and (time.time() - lapperup > 10):
                        flag = "Raise your arms"
                    elif angle < 40 and (time.time() - lapperdown > 10):
                        flag = "Lower your arms"
                    else:
                        flag = None

                except:
                    pass

                #         Setup status box
                cv2.rectangle(image, (0, 0), (250, 73), (245, 117, 16), -1)

                # Rep data
                cv2.putText(
                    image,
                    "REPS ",
                    (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(counter) + " ",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Stage
                cv2.putText(
                    image,
                    " Timer ",
                    (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(int(laptime)),
                    (85, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Warning box
                cv2.rectangle(image, (500, 0), (600, 73), (245, 117, 16), -1)
                cv2.putText(
                    image,
                    "Error",
                    (500, 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    flag,
                    (500, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Pose Detection
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                )

                ret, buffer = cv2.imencode(".jpg", image)
                image = buffer.tobytes()

                if counter >= 1:
                    params["counter"] = counter
                    r = requests.get(url="http://127.0.0.1:5000/score", params=params)

                if capture:
                    params["counter"] = counter
                    # url = "http://127.0.0.1:5000/score?counter=" + string(counter)
                    r = requests.get(url="http://127.0.0.1:5000/score", params=params)
                    break
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + image + b"\r\n")
