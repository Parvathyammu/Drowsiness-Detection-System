The **Drowsiness Detection System Using Computer Vision** is a real-time driver fatigue monitoring application developed using Python, OpenCV, and MediaPipe. The project aims to improve road safety by identifying potential signs of driver drowsiness through continuous facial behaviour analysis using a webcam.

The system uses MediaPipe Face Landmarker to detect facial landmarks and analyse eye and mouth movements. It implements **Eye Aspect Ratio (EAR)** for eye closure detection and **Mouth Aspect Ratio (MAR)** for yawning detection. It also monitors blink patterns, calculates blink rate over a rolling 60-second window, and measures **PERCLOS (Percentage of Eye Closure)** to estimate eye closure over time.

An OpenCV-based dashboard provides real-time monitoring by displaying important parameters such as EAR, MAR, blink count, blink rate, PERCLOS, eye status, yawning status, and drowsiness alert status. When prolonged eye closure is detected, the system generates an audio warning using the Windows Winsound module to alert the user.

### Key Features
- Real-time face detection and facial landmark tracking.
- Eye closure monitoring using EAR.
- Yawning detection using MAR.
- Automatic blink counting and blink rate calculation.
- PERCLOS measurement using a rolling observation window.
- Audio alerts for prolonged eye closure.
- Live OpenCV dashboard for monitoring detection results.

### Technologies Used
Python, OpenCV, MediaPipe, NumPy, Threading, Winsound, and Collections (Deque).

### Project Domain
Artificial Intelligence | Computer Vision | Driver Safety | Real-Time Monitoring

This project demonstrates the practical application of computer vision in fatigue monitoring and intelligent road safety systems. It can be further enhanced with head pose estimation, machine learning-based fatigue classification, event logging, and integration with intelligent vehicle assistance systems.
