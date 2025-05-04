import cv2
from ultralytics import YOLO
import numpy as np
import pygame
import mediapipe as mp
import cvzone
import serial
import time

import os
from utils.tracker import Tracker
from utils.log_manager import start_new_session, update_detection, end_session, update_location
import geocoder 

# Initialize serial connection to Arduino
arduino = serial.Serial('COM5', 9600)  # Replace with your actual port
time.sleep(2)  # Allow time for Arduino to initialize

pygame.init()
pygame.mixer.music.load("utils/alarm.wav")

# Load models
fire_smoke_model = YOLO("models/fire_smoke_yolo.pt")
human_model = YOLO("models/yolov8n.pt")

with open("coco.txt", "r") as f:
    classes = f.read().split("\n")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

fall_threshold = 1
tracker = Tracker()

cap = cv2.VideoCapture(r"C:\Users\harsh\Project_final\Fire-Detection-Arduino-by-Camera\test.mp4")
if not cap.isOpened():
    print("Error: Video not found or can't be opened.")
    exit()

# Logging
session_id = start_new_session()
detected_once = {}

# Fire logic control flags
fire_detected_once = False
fire_count = 0
servo_started = False
led8_triggered = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 700))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FIRE/SMOKE Detection
    fire_results = fire_smoke_model(frame)[0]
    frame = fire_results.plot()
    for box in fire_results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # Fire
            fire_count += 1
            detected_once["fire"] = detected_once.get("fire", 0) + 1

            if not fire_detected_once:
                fire_detected_once = True
                arduino.write(b'F')  # Turn on LED on pin 7 (fire detected once)
                time.sleep(1)
                arduino.write(b'S')  # Start continuous servo rotation
                servo_started = True

            elif fire_count >= 3 and not led8_triggered:
                arduino.write(b'T')  # Turn on LED on pin 8 (fire detected 3 times)
                led8_triggered = True

        elif cls == 2:  # Smoke
            detected_once["smoke"] = detected_once.get("smoke", 0) + 1

    # POSE + FALL Detection
    result_pose = pose.process(rgb_frame)
    if result_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

    person_results = human_model(frame)[0]
    lis = []
    for box in person_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if int(box.cls[0]) < len(classes):
            name = classes[int(box.cls[0])]
            if "person" in name and box.conf[0] > 0.5:
                lis.append([x1, y1, x2, y2])

    bbox_id = tracker.update(lis)
    for bb in bbox_id:
        x1, y1, x2, y2, pid = bb
        width, height = x2 - x1, y2 - y1
        fall_ratio = width / height

        cvzone.cornerRect(frame, (x1, y1, width, height), l=20, t=10)

        if fall_ratio > fall_threshold:
            detected_once["lying_person"] = detected_once.get("lying_person", 0) + 1
            cvzone.putTextRect(frame, "Fall Detected!", (x1, y1 - 40), 2, 3, colorR=(0, 0, 255))
            cvzone.cornerRect(frame, (x1, y1, width, height), l=20, t=10, colorR=(255, 0, 0))
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
            arduino.write(b'L')  # Send 'L' for fall LED

    cv2.imshow("Multi Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
arduino.write(b'O')  # Send 'O' to turn off all hardware
arduino.close()
cap.release()
cv2.destroyAllWindows()

# Logging and geolocation
for label, count in detected_once.items():
    if count > 0:
        update_detection(session_id, label)

end_session(session_id)

g = geocoder.ip('me')
lat, lon = g.latlng if g.ok else (None, None)
location = g.city or "Unknown"

import json
from datetime import datetime

def update_accident_db(labels, severity):
    accident_entry = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": location,
        "severity": severity,
        "labels": labels,
        "ambulance_enroute": False,
        "lat": lat,
        "lon": lon,
    }
    with open("logs/accident_db.json", "w") as f:
        json.dump(accident_entry, f, indent=2)

update_location(session_id, lat, lon, location)

# You must define final_labels and severity_score in your main flow
# If not, replace with dummy or log only `detected_once`
final_labels = list(detected_once.keys())
severity_score = sum(detected_once.values())

update_accident_db(final_labels, severity_score)
