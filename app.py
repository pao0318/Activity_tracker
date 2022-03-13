from flask import Flask, render_template, Response, request
import json
import cv2
import datetime
import os, sys
import numpy as np

import time

from exercises import *

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video1")
def video1():
    return Response(
        elbowFlexion(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/exercise1")
def exercise1():
    counter = 0
    return render_template("exercise1.html")



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
