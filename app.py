from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
from IPython.display import display
from PIL import Image
import time

from generate_frames1 import *
from generate_frames2 import *
from generate_frames3 import *

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)

app = Flask(__name__)


cap = cv2.VideoCapture(0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video1")
def video1():
    return Response(
        generate_frames1(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video2")
def video2():
    return Response(
        generate_frames2(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video3")
def video3():
    return Response(
        generate_frames3(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/exercise1")
def exercise1():
    return render_template("exercise1.html")


@app.route("/exercise2")
def exercise2():
    return render_template("exercise2.html")


@app.route("/exercise3")
def exercise3():
    return render_template("exercise3.html")


# @app.route("/requests", methods=["GET", "POST"])
# def results():
#     global switch, camera
#     if request.method == "POST":
#         if request.form.get("click") == "Capture":
#             global capture
#             capture = 1
#         elif request.form.get("stop") == "Stop/Start":
#             if switch == 1:
#                 switch = 0
#                 camera.release()
#                 cv2.destroyAllWindows()
#             else:
#                 camera = cv2.VideoCapture(0)
#                 switch = 1
#     elif request.method == "GET":
#         return render_template("index.html")
#     return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
