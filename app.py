from flask import Flask, render_template, Response, request
import json

import cv2
import datetime, time
import os, sys
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

global capture, switch, counter
capture = 0
counter = 0

switch = 1
cap = cv2.VideoCapture(0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video1")
def video1():
    global capture
    capture = 1
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
    counter = 0
    return render_template("exercise1.html")


@app.route("/exercise2")
def exercise2():
    return render_template("exercise2.html")


@app.route("/exercise3")
def exercise3():
    return render_template("exercise3.html")


@app.route("/score", methods=["GET", "POST"])
def score():
    global counter
    if request.method == "GET":
        counter = request.args.get("counter")
    else:
        # counter = request.args.get("counter")
        print("final is", counter)
        return render_template("results.html", res=counter)

    return render_template("index.html", res=counter)


@app.route("/requests", methods=["POST", "GET"])
def tasks():
    global switch, cap
    if request.method == "GET":
        if request.form.get("pause") == "Pause":
            global capture
            capture = 1

    elif request.method == "POST":
        return render_template("index.html", res=request.counter)

    return render_template("index.html", res=counter)


if __name__ == "__main__":
    app.run(debug=True)
