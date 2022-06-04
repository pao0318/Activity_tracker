from multiprocessing.dummy import active_children
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
from playsound import playsound
import io
from PIL import Image
import base64
import cv2
from matplotlib.style import use
import numpy as np
from flask_cors import CORS, cross_origin
import imutils
import json
from engineio.payload import Payload

import mediapipe as mp

import requests
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def get_tolerance(str):
    if str == "low":
        tol_angle = 10
    else:
        tol_angle = 20
    return tol_angle


class Var:
    def __init__(self, severeity='low', threshtime=3):
        self.counter = 0
        self.stage = None
        self.t1 = self.t2 = time.time()
        self.curr_timer = time.time()
        self.start_time = time.time()
        self.timer = 0
        self.times = [0] * 4
        self.threshtime = threshtime
        self.feedback = None
        self.feedback_raise_time=0
        self.feedback_lower_time=0
        self.rep_time = None
        self.tol_angle = get_tolerance(severeity)
        self.error = 0
        self.params = {"counter": 0, "timer": 0, "error": 0}
        self.min_angle = 360
        self.max_angle = 0
        self.exercise = None

    def reset(self, severeity='low', threshtime=3):
        self.counter = 0
        self.stage = None
        self.t1 = self.t2 = time.time()
        self.curr_timer = time.time()
        self.start_time = time.time()
        self.timer = 0
        self.times = [0] * 4
        self.threshtime = threshtime
        self.feedback = None
        self.rep_time = None
        self.tol_angle = get_tolerance(severeity)
        self.error = 0
        self.params = {"counter": 0, "timer": 0, "error": 0}
        self.min_angle = 360
        self.max_angle = 0
        self.exercise = None


global vars
global userid
global exercise
global severity
global expertime
global exerciseid
global activityid
vars = Var()


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def calculate_angle_lateral(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    return angle


Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')



@app.route("/elbowFlexion", methods=['POST', 'GET'])
def elbowFlexion():
    vars.reset()
    return render_template("elbowFlexion.html")


@app.route("/lateralFlexion", methods=['POST', 'GET'])
def lateralFlexion():
    vars.reset()
    return render_template("lateralFlexion.html")


@app.route("/kneeChest", methods=['POST', 'GET'])
def kneeChest():
    vars.reset()
    return render_template("kneeChest.html")


@app.route("/lunges", methods=['POST', 'GET'])
def lunges():
    vars.reset()
    return render_template("lunges.html")


@app.route("/squats", methods=['POST', 'GET'])
def squats():
    vars.reset()
    return render_template("squats.html")

@app.route("/stop", methods=['POST', 'GET'])
def stop():
    vars.reset()
    return render_template("index.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    vars.timer = time.time() - vars.start_time
    if vars.exercise == 'LATERAL FLEXION':
        return render_template("aroms_result.html", obj=vars)
    else:
        return render_template("results.html", obj=vars)


@app.route("/complete", methods=['POST', 'GET'])
def complete():
    global userid

    obj1 = {
        "name": vars.exercise,
        "min": vars.min_angle,
        "max": vars.max_angle
    }

    headers = {"Content-Type": "application/json"}
    dictToSend = json.dumps(obj1)
    response = requests.post(
        "https://exelligence-backend.herokuapp.com/api/excercise/aromsrecord",
        data=dictToSend,
        headers=headers,
    )

    if vars.exercise == 'LATERAL FLEXION':
        return render_template("aroms_result.html", obj=vars)
    else:
        return render_template("results.html", obj=vars)
    # return request("post", "http://localhost:5000/api/exercise/report", data=arr)


@app.route("/squats_lunges_complete", methods=['POST', 'GET'])
def squats_lunges_complete():
    global userid
    global exerciseid
    global activityid

    obj1 = {
        "counter": vars.counter,
        "timer": vars.timer,
      
        "exercisename": vars.exercise,

    }

    headers = {"Content-Type": "application/json"}
    dictToSend = json.dumps(obj1)
    response = requests.post(
        "https://exelligence-backend.herokuapp.com/api/excercise/report",
        data=dictToSend,
        headers=headers,
    )

    return render_template("results.html", obj=vars)
    # return request("post", "http://localhost:5000/api/exercise/report", data=arr)


@app.route("/aroms/<url>")
def exercise1(url):
    arr = url.split(",", 2)

    global userid
    global exercisename

    global obj

    userid = arr[1]
    exercisename = 'Lateral Flexion'

    print('USER ID --->', userid)

    obj = {
        "userid": userid,
        "exercisename": exercisename,
    }

    print('HELLO WORLD')

    return render_template("lateralFlexion.html")


def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx+7:]
    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)


# ELBOW FLEXION
@socketio.on('image')
def image(data_image):
    # Assing Exercisse
    vars.exercise = 'ELBOW FLEXION'

    image = (readb64(data_image))
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Storing curr time
    vars.curr_timer = time.time()

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate angle
        angle = calculate_angle(shoulder, elbow, wrist)
        vars.min_angle = min(vars.min_angle, angle)
        vars.max_angle = max(vars.max_angle, angle)

        # Visualize angle
        cv2.putText(image, str(angle)[:5],
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                        255, 255, 255), 2, cv2.LINE_AA
                    )

        # Curl counter logic
        if angle > 160 - vars.tol_angle and (vars.stage is None or vars.stage == 'up'):
            if vars.stage == 'up':
                vars.t2 = time.time()  # curr rep time
                # storing to track average time per rep
                vars.times[(vars.counter-1) % 4] = abs(vars.t2-vars.t1)
                # storing it to print later
                vars.rep_time = abs(vars.t2-vars.t1)

            vars.t1 = time.time()  # previous rep time

        if angle > 160 - vars.tol_angle:
            vars.stage = "down"

        if angle < 35 + vars.tol_angle and vars.stage == 'down':
            vars.stage = "up"
            vars.counter += 1
            # print(counter)

    except:
        pass

    # Render curl counter
    # Setup status box
    cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

    # Rep data
    cv2.putText(image, 'REPS', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(vars.counter),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(image, 'STAGE', (140, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, vars.stage,
                (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Time Data
    cv2.putText(image, 'REP TIME', (320, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image, str(vars.rep_time)[0:4],
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Feedback
    cv2.putText(image, 'FEEDBACK', (500, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    if abs(vars.curr_timer-vars.t1) > 2 * vars.threshtime:  # if curr time - prev rep > 3 we say
        if vars.stage == 'up':
            vars.feedback = 'Lower your arms'
            vars.feedback_lower_time=time.time()

        else:
            vars.feedback = 'Raise your arms'
            vars.feedback_raise_time=time.time() 

    else:
        vars.feedback = None
    





    cv2.putText(image, vars.feedback,
                (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=1)
                              )

    if vars.counter >= 1:
        vars.params["counter"] = vars.counter
        tim = time.time()
        vars.params["timer"] = np.round(tim - vars.start_time, 2)
        vars.params["error"] = vars.error

    imgencode = cv2.imencode(
        '.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


# SQUATS
@socketio.on('image1')
def image1(data_image):
    # Assing Exercisse
    vars.exercise = 'SQUATS'

    image = (readb64(data_image))
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Storing curr time
    vars.curr_timer = time.time()

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Get Coordinates
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate angle
        angle = calculate_angle(left_hip, left_knee, left_ankle)
        vars.min_angle = min(vars.min_angle, angle)
        vars.max_angle = max(vars.max_angle, angle)

        # Visualize angle
        cv2.putText(image, str(angle),
                    tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                    255, 255), 2, cv2.LINE_AA
                    )

        # Curl counter logic
        if angle > 160 - vars.tol_angle and (vars.stage is None or vars.stage == 'down'):
            if vars.stage == 'down':
                vars.t2 = time.time()  # curr rep time
                # storing to track average time per rep
                vars.times[(vars.counter-1) % 4] = abs(vars.t2-vars.t1)
                # storing it to print later
                vars.rep_time = abs(vars.t2-vars.t1)

            vars.t1 = time.time()  # previous rep time

        if angle > 160 - vars.tol_angle:
            vars.stage = "up"

        if angle < 45 + vars.tol_angle and vars.stage == 'up':
            vars.stage = "down"
            vars.counter += 1
            # print(counter)

    except:
        pass

    # Render curl counter
    # Setup status box
    cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

    # Rep data
    cv2.putText(image, 'REPS', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(vars.counter),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(image, 'STAGE', (140, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, vars.stage,
                (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Time Data
    cv2.putText(image, 'REP TIME', (320, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image, str(vars.rep_time)[0:4],
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Feedback
    cv2.putText(image, 'FEEDBACK', (500, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if vars.counter % 4 == 0 and vars.counter != 0:
        if (np.mean(vars.times) - vars.threshtime) > vars.threshtime/4:
            vars.feedback = 'Do Fast'
            vars.error += 1
        elif (vars.threshtime - np.mean(vars.times)) > vars.threshtime/4:
            vars.feedback = 'Do slow'
            vars.error += 1
        else:
            vars.feedback = 'Doing good'

    elif abs(vars.curr_timer-vars.t1) > 2 * vars.threshtime:  # if curr time - prev rep > 3 we say
        if vars.stage == 'up':
            vars.feedback = 'Go down'
        else:
            vars.feedback = 'Go up'

    else:
        vars.feedback = None

    cv2.putText(image, vars.feedback,
                (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=1)
                              )

    if vars.counter >= 1:
        vars.params["counter"] = vars.counter
        tim = time.time()
        vars.params["timer"] = np.round(tim - vars.start_time, 2)
        vars.params["error"] = vars.error

    imgencode = cv2.imencode(
        '.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

# emit the frame back
    emit('response_back', stringData)


# Lateral Flexion
@socketio.on('image2')
def image(data_image):
    # Assing Exercisse
    vars.exercise = 'LATERAL FLEXION'

    image = (readb64(data_image))
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Storing curr time
    vars.curr_timer = time.time()

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Get Coordinates
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        # Calculate angle
        angle = calculate_angle_lateral(left_knee, left_hip, left_shoulder)
        vars.min_angle = min(vars.min_angle, angle)
        vars.max_angle = max(vars.max_angle, angle)

        # Visualize angle
        cv2.putText(image, str(angle)[0:5],
                    tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                    255, 255), 2, cv2.LINE_AA
                    )

        # Curl counter logic
        if angle < 170 + vars.tol_angle and (vars.stage is None or vars.stage == 'right'):
            if vars.stage == 'right':
                vars.t2 = time.time()  # curr rep time
                # storing to track average time per rep
                vars.times[(vars.counter-1) % 4] = abs(vars.t2-vars.t1)
                # storing it to print later
                vars.rep_time = abs(vars.t2-vars.t1)

            vars.t1 = time.time()  # previous rep time

        if angle > 170 + vars.tol_angle:
            vars.stage = "left"

        if angle > 200 - vars.tol_angle and vars.stage == 'left':
            vars.stage = "right"
            vars.counter += 1
            # print(counter)

    except:
        pass

    # Render curl counter
    # Setup status box
    cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

    # Rep data
    # cv2.putText(image, 'REPS', (15, 12),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(image, str(vars.counter),
    #             (10, 60),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(image, 'STAGE', (140, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, vars.stage,
                (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Time Data
    cv2.putText(image, 'REP TIME', (320, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image, str(vars.rep_time)[0:4],
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Feedback
    cv2.putText(image, 'FEEDBACK', (500, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if vars.counter % 4 == 0 and vars.counter != 0:
        if (np.mean(vars.times) - vars.threshtime) > vars.threshtime/4:
            vars.feedback = 'Do Fast'
            vars.error += 1
        elif (vars.threshtime - np.mean(vars.times)) > vars.threshtime/4:
            vars.feedback = 'Do slow'
            vars.error += 1
        else:
            vars.feedback = 'Doing good'

    elif abs(vars.curr_timer-vars.t1) > 2 * vars.threshtime:  # if curr time - prev rep > 3 we say
        if vars.stage == 'left':
            vars.feedback = 'Bend Rightwards'
        else:
            vars.feedback = 'Bend Leftwards'

    else:
        vars.feedback = None

    cv2.putText(image, vars.feedback,
                (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=1)
                              )

    if vars.counter >= 1:
        vars.params["counter"] = vars.counter
        tim = time.time()
        vars.params["timer"] = np.round(tim - vars.start_time, 2)
        vars.params["error"] = vars.error

    imgencode = cv2.imencode(
        '.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


# Knee to Chest
@socketio.on('image3')
def image(data_image):
    # Assing Exercisse
    vars.exercise = 'KNEE TO CHEST'

    image = (readb64(data_image))
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Storing curr time
    vars.curr_timer = time.time()

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Get Coordinates
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        # Calculate angle
        angle = calculate_angle(left_knee, left_hip, left_shoulder)
        vars.min_angle = min(vars.min_angle, angle)
        vars.max_angle = max(vars.max_angle, angle)

        # Visualize angle
        cv2.putText(image, str(angle),
                    tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                    255, 255), 2, cv2.LINE_AA
                    )

        # Curl counter logic
        if angle > 160 - vars.tol_angle and (vars.stage is None or vars.stage == 'up'):
            if vars.stage == 'up':
                vars.t2 = time.time()  # curr rep time
                # storing to track average time per rep
                vars.times[(vars.counter-1) % 4] = abs(vars.t2-vars.t1)
                # storing it to print later
                vars.rep_time = abs(vars.t2-vars.t1)

            vars.t1 = time.time()  # previous rep time

        if angle > 160 - vars.tol_angle:
            vars.stage = "down"

        if angle < 70 + vars.tol_angle and vars.stage == 'down':
            vars.stage = "up"
            vars.counter += 1
            # print(counter)

    except:
        pass

    # Render curl counter
    # Setup status box
    cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

    # Rep data
    cv2.putText(image, 'REPS', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(vars.counter),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(image, 'STAGE', (140, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, vars.stage,
                (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Time Data
    cv2.putText(image, 'REP TIME', (320, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image, str(vars.rep_time)[0:4],
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Feedback
    cv2.putText(image, 'FEEDBACK', (500, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if vars.counter % 4 == 0 and vars.counter != 0:
        if (np.mean(vars.times) - vars.threshtime) > vars.threshtime/4:
            vars.feedback = 'Do Fast'
            vars.error += 1
        elif (vars.threshtime - np.mean(vars.times)) > vars.threshtime/4:
            vars.feedback = 'Do slow'
            vars.error += 1
        else:
            vars.feedback = 'Doing good'

    elif abs(vars.curr_timer-vars.t1) > 2 * vars.threshtime:  # if curr time - prev rep > 3 we say
        if vars.stage == 'up':
            vars.feedback = 'Lower your knees'
        else:
            vars.feedback = 'Raise your knees'

    else:
        vars.feedback = None

    cv2.putText(image, vars.feedback,
                (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=1)
                              )

    if vars.counter >= 1:
        vars.params["counter"] = vars.counter
        tim = time.time()
        vars.params["timer"] = np.round(tim - vars.start_time, 2)
        vars.params["error"] = vars.error

    imgencode = cv2.imencode(
        '.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


# LUNGES
@socketio.on('image4')
def image(data_image):
    # Assing Exercisse
    vars.exercise = 'LUNGES'

    image = (readb64(data_image))
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Storing curr time
    vars.curr_timer = time.time()

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Get Coordinates
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate angle
        angle = calculate_angle(left_knee, left_hip, left_ankle)
        vars.min_angle = min(vars.min_angle, angle)
        vars.max_angle = max(vars.max_angle, angle)

        # Visualize angle
        cv2.putText(image, str(angle),
                    tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                    255, 255), 2, cv2.LINE_AA
                    )

        # Curl counter logic
        if angle >  45 - vars.tol_angle and (vars.stage is None or vars.stage == 'up'):
            if vars.stage == 'up':
                vars.t2 = time.time()  # curr rep time
                # storing to track average time per rep
                vars.times[(vars.counter-1) % 4] = abs(vars.t2-vars.t1)
                # storing it to print later
                vars.rep_time = abs(vars.t2-vars.t1)

            vars.t1 = time.time()  # previous rep time

        if angle > 45 - vars.tol_angle:
            vars.stage = "down"

        if angle < 5 + vars.tol_angle and vars.stage == 'down':
            vars.stage = "up"
            vars.counter += 1
            # print(counter)

    except:
        pass

    # Render curl counter
    # Setup status box
    cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

    # Rep data
    cv2.putText(image, 'REPS', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(vars.counter),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(image, 'STAGE', (140, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, vars.stage,
                (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Time Data
    cv2.putText(image, 'REP TIME', (320, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image, str(vars.rep_time)[0:4],
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Feedback
    cv2.putText(image, 'FEEDBACK', (500, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if vars.counter % 4 == 0 and vars.counter != 0:
        if (np.mean(vars.times) - vars.threshtime) > vars.threshtime/4:
            vars.feedback = 'Do Fast'
            vars.error += 1
        elif (vars.threshtime - np.mean(vars.times)) > vars.threshtime/4:
            vars.feedback = 'Do slow'
            vars.error += 1
        else:
            vars.feedback = 'Doing good'

    elif abs(vars.curr_timer-vars.t1) > 2 * vars.threshtime:  # if curr time - prev rep > 3 we say
        if vars.stage == 'up':
            vars.feedback = 'Lower your knees'
        else:
            vars.feedback = 'Raise your knees'

    else:
        vars.feedback = None

    cv2.putText(image, vars.feedback,
                (450, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=1)
                              )

    if vars.counter >= 1:
        vars.params["counter"] = vars.counter
        tim = time.time()
        vars.params["timer"] = np.round(tim - vars.start_time, 2)
        vars.params["error"] = vars.error

    imgencode = cv2.imencode(
        '.jpeg', image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


if __name__ == '__main__':
    socketio.run(app, port=9990, debug=True)
