# Driver Drowsiness Detection 🚗😴

Real-time driver fatigue detection using **Python**, **MediaPipe**, and **OpenCV**.

## Features
- Eye-closure detection (Eye Aspect Ratio)
- Yawn detection
- Head nod detection
- Alarm alert system
- Event logging

## Requirements
- Python 3.9–3.10
- OpenCV
- MediaPipe

## How it Works
The system uses MediaPipe Face Mesh to detect facial landmarks.
Eye Aspect Ratio (EYE_AR) is used to determine eye closure.
If eyes remain closed for a set duration, an alarm is triggered.

## Setup
1. Install Python 3.10
2. Install dependencies:
   pip install opencv-python mediapipe
3. Run:
   python driver_drowsiness_detection.py

## Alarm
The alarm sound is triggered when drowsiness is detected.
Make sure `alarm.wav` is in the same directory as the script.
