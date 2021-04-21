from threading import Thread
from queue import Queue

import numpy as np
import cv2
import requests
import json
import os, time


with open('config.json', 'r') as f:
    config = json.load(f)

sd_thresh = config['detection']['threshold']
upload_url = config['upload']['url']
uid = config['upload']['uid']
skip_frames = config['detection']['skip_frames']

frame_queue = Queue()


def dist_map(frame1, frame2):
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:, :, 0] ** 2 + diff32[:, :, 1] ** 2 + diff32[:, :, 2] ** 2) / np.sqrt(
        255 ** 2 + 255 ** 2 + 255 ** 2)
    dist = np.uint8(norm32 * 255)
    return dist


def send_frame():
    while True:
        frame = frame_queue.get()
        print("Sending")
        _, image = cv2.imencode('.jpg', frame)

        requests.post(upload_url, files={'img': ('frame.jpg',  image, 'image/jpeg')}, data={'uid': uid})


worker = Thread(target=send_frame)
worker.setDaemon(True)
worker.start()

print("Trying to connect to the camera")

while True:
    cap = cv2.VideoCapture(0)

    if cap is None or not cap.isOpened():
        print("Retrying to connect")
        time.sleep(5)
    else:
        print("Connected")
        break

cap.set(cv2.CAP_PROP_FPS, config['capture']['fps'])

_, frame1 = cap.read()
_, frame2 = cap.read()

skip_count = 0

while True:
    try:
        _, frame3 = cap.read()
        rows, cols, _ = np.shape(frame3)
        dist = dist_map(frame1, frame3)

        frame1 = frame2
        frame2 = frame3
    except ValueError:
        print("Restarting Script")
        os.system("python motion_detector.py")

    mod = cv2.GaussianBlur(dist, (9, 9), 0)

    _, thresh = cv2.threshold(mod, 100, 255, 0)

    _, stDev = cv2.meanStdDev(mod)

    if stDev > sd_thresh:
        if skip_count == skip_frames:
            skip_count = 0
            frame_queue.put(frame2)
        else:
            skip_count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
