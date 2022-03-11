import cv2
import mediapipe as mp
import numpy as np
import time
import requests
from IPython.display import display

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
capture = 0

def get_tolerance(str):
    if str == "low":
        tol_angle = 10
    else:
        tol_angle = 20
    return tol_angle


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



def elbowFlexion(severity='low', threshtime=2):
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    stage = None
    t1 = t2 = 0
    times = [0] * 4
    feedback = None
    tol_angle = get_tolerance(severity)
    error = 0
    params = {"counter": counter, "timer": 0, "error": error}

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

                # Visualize angle
                cv2.putText(image, str(angle)[:5],
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle > 160 - tol_angle and (stage is None or stage == 'up'):
                    if stage == 'up':
                        times[(counter-1) % 4] = abs(t2-t1)

                    t1 = time.time()
                    t2 = t1

                if angle > 160 - tol_angle:
                    stage = "down"

                if angle < 35 + tol_angle and stage == 'down':
                    stage = "up"
                    counter += 1
                    # print(counter)

                if angle < 35 + tol_angle:
                    t2 = time.time()

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (140, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Time Data
            cv2.putText(image, 'REP TIME', (320, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(image, str(t2-t1)[0:4],
                        (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Feedback
            cv2.putText(image, 'FEEDBACK', (500, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if counter % 4 == 0 and counter != 0:
                if (np.mean(times) - threshtime) > threshtime/4:
                    feedback = 'Do Fast'
                    error += 1
                elif (threshtime - np.mean(times)) > threshtime/4:
                    feedback = 'Do slow'
                    error += 1
                else:
                    feedback = 'Doing good'

            cv2.putText(image, feedback,
                        (450, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=1)
                                      )

            # cv2.imshow("images1",image)
            ret, buffer = cv2.imencode(".jpg", image)
            image = buffer.tobytes()
            
            if counter >= 1:
                params["counter"] = counter
                tim = time.time()
                params["timer"] = np.round(tim, 2)
                params["error"] = error
                r = requests.get(
                    url="http://127.0.0.1:5000/score", params=params)
            # if capture:
            #         params["counter"] = counter
            #         r = requests.get(url="http://127.0.0.1:5000/score", params=params)
            #         break

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + image + b"\r\n")



def lateralFlexion(threshtime=5):
    cap = cv2.VideoCapture(0)

    # curl counter variables
    counter = 0
    stage = None
    t1 = t2 = 0
    times = [0] * 4
    feedback = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Coloring BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
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
                angle = calculate_angle_lateral(
                    left_knee, left_hip, left_shoulder)

                # Visualize angle
                cv2.putText(image, str(angle)[0:5],
                            tuple(np.multiply(
                                left_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                255, 255, 255), 2, cv2.LINE_AA
                            )

                # print(angle)

                # Curl counter logic
                if angle < 170 and (stage == None or stage == 'right'):
                    if stage == 'right':
                        times[(counter-1) % 4] = abs(t2-t1)

                    t1 = time.time()
                    t2 = t1

                if angle < 170:
                    stage = 'left'

                if angle > 200 and stage == 'left':
                    stage = 'right'
                    counter += 1

                if angle > 200:
                    t2 = time.time()

                # print(counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (140, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Time Data
            cv2.putText(image, 'REP TIME', (300, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(image, str(abs(t2-t1))[0:4],
                        (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Feedback
            cv2.putText(image, 'FEEDBACK', (500, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if (counter) % 4 == 0 and counter != 0:
                if (np.mean(times) - threshtime) > threshtime/4:
                    feedback = 'Do Fast'
                elif (threshtime - np.mean(times)) > threshtime/4:
                    feedback = 'Do slow'
                else:
                    feedback = 'Doing good'

            cv2.putText(image, feedback,
                        (450, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=1)
                                      )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def kneeChest(threshtime=4):
    cap = cv2.VideoCapture(0)

    # curl counter variables
    counter = 0
    stage = None
    t1 = t2 = 0
    times = [0] * 4
    feedback = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Coloring BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
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

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(
                                left_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                255, 255, 255), 2, cv2.LINE_AA
                            )

                # print(angle)

            # Curl counter logic
                if angle > 160 and (stage is None or stage == 'up'):
                    if stage == 'up':
                        times[(counter-1) % 4] = abs(t2-t1)

                    t1 = time.time()
                    t2 = t1

                if angle > 160:
                    stage = "down"

                if angle < 70 and stage == 'down':
                    stage = "up"
                    counter += 1
                    # print(counter)

                if angle < 70:
                    t2 = time.time()

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (140, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Time Data
            cv2.putText(image, 'REP TIME', (300, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(image, str(t2-t1)[0:4],
                        (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Feedback
            cv2.putText(image, 'FEEDBACK', (500, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if counter % 4 == 0 and counter != 0:
                if (np.mean(times) - threshtime) > threshtime/4:
                    feedback = 'Do Fast'
                elif (threshtime - np.mean(times)) > threshtime/4:
                    feedback = 'Do slow'
                else:
                    feedback = 'Doing good'

            cv2.putText(image, feedback,
                        (450, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=1, circle_radius=1)
                                      )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def lunges(threshtime=2):
    cap = cv2.VideoCapture(0)

    # curl counter variables
    counter = 0
    stage = None
    t1 = t2 = 0
    times = [0] * 4
    feedback = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Coloring BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
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

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(left_knee, [
                                  640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                255, 255, 255), 2, cv2.LINE_AA
                            )

    #             print(angle)

            # Curl counter logic
                if angle > 85 and (stage is None or stage == 'up'):
                    if stage == 'up':
                        times[(counter-1) % 4] = abs(t2-t1)

                    t1 = time.time()
                    t2 = t1

                if angle > 85:
                    stage = "down"

                if angle < 5 and stage == 'down':
                    stage = "up"
                    counter += 1
                    # print(counter)

                if angle < 5:
                    t2 = time.time()

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (640, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (140, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Time Data
            cv2.putText(image, 'REP TIME', (300, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(image, str(abs(t2-t1))[0:4],
                        (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Feedback
            cv2.putText(image, 'FEEDBACK', (500, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if (counter) % 4 == 0 and counter != 0:
                if (np.mean(times) - threshtime) > threshtime/4:
                    feedback = 'Do Fast'
                elif (threshtime - np.mean(times)) > threshtime/4:
                    feedback = 'Do slow'
                else:
                    feedback = 'Doing good'

            cv2.putText(image, feedback,
                        (450, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=1, circle_radius=1)
                                      )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()