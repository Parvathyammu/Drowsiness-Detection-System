# Drowsiness Detection System Using Computer Vision

## 1. Project Overview

The **Drowsiness Detection System** is a real-time driver fatigue monitoring application developed using Python, OpenCV, and MediaPipe.

The system monitors a person's facial movements through a webcam and detects signs of drowsiness, such as prolonged eye closure and yawning. When drowsiness is detected, an alert is generated to warn the user.

The main objective of this project is to improve driver safety by identifying early signs of fatigue and providing timely warnings.

## 2. Problem Statement

Driver fatigue is one of the major causes of road accidents. Drivers may become sleepy during long journeys, late-night driving, or due to insufficient rest.

Traditional methods of detecting fatigue often depend on manual observation. This project uses computer vision techniques to automatically monitor facial features and identify possible signs of drowsiness.

## 3. Objectives

- Detect a face and facial landmarks in real time.
- Monitor eye movements to identify prolonged eye closure.
- Detect yawning using mouth movements.
- Calculate the blink rate.
- Measure eye closure over time using PERCLOS.
- Generate an audio alert when drowsiness is detected.
- Display real-time monitoring information through an OpenCV dashboard.

## 4. Features

### 4.1 Eye Closure Detection

The system uses the **Eye Aspect Ratio (EAR)** to monitor eye movements.

EAR is calculated using facial landmark coordinates. When the EAR value falls below a predefined threshold, the system considers the eyes closed.

If the eyes remain closed for a specified duration, the system identifies possible drowsiness.

### 4.2 Yawning Detection

The system uses the **Mouth Aspect Ratio (MAR)** to detect yawning.

When the mouth opens beyond a predefined threshold for a specified duration, the system identifies a possible yawn.

### 4.3 Blink Rate Detection

The system counts the number of eye blinks and calculates the blink rate over a 60-second time window.

Blink rate helps monitor changes in blinking behaviour that may be associated with fatigue.

### 4.4 PERCLOS Calculation

PERCLOS stands for **Percentage of Eye Closure**.

It measures the percentage of time that the eyes remain closed during a specific observation period.

The system uses a rolling 60-second window to monitor eye closure and estimate fatigue-related behaviour.

### 4.5 Audio Alert System

When prolonged eye closure indicates possible drowsiness, the system generates an audio warning to alert the user.

### 4.6 OpenCV Dashboard

An OpenCV-based dashboard displays real-time monitoring information, including:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Blink count
- Blink rate
- PERCLOS
- Eye status
- Yawning status
- Drowsiness alert status

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Face detection and facial landmark tracking |
| NumPy | Numerical calculations |
| Time | Duration and time-window calculations |
| Winsound | Audio alerts on Windows |
| Threading | Running alert operations separately |
| Collections (deque) | Maintaining rolling time-window data |

## 6. System Workflow

1. Start the application.
2. Access the webcam.
3. Capture video frames continuously.
4. Detect facial landmarks using MediaPipe.
5. Calculate EAR to monitor eye closure.
6. Calculate MAR to detect yawning.
7. Count blinks and calculate blink rate.
8. Calculate PERCLOS over the observation window.
9. Display the monitoring values on the OpenCV dashboard.
10. Generate an audio alert when prolonged eye closure indicates possible drowsiness.

## 7. Project Structure

```text
Drowsiness-Detection-System/
│
├── models/
│   └── face_landmarker.task
│
├── drowsiness.py
│
├── requirements.txt
│
├── README.md
│
└── screenshots/
```

*The filenames should match your actual project files.*

## 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Drowsiness-Detection-System.git
```

### Step 2: Open the Project Folder

```bash
cd Drowsiness-Detection-System
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python drowsiness.py
```

Make sure the MediaPipe model file is available at:

```text
models/face_landmarker.task
```

## 9. Requirements

- Python 3
- A working webcam
- Windows operating system for the current `winsound` audio-alert implementation
- Required Python libraries listed in `requirements.txt`
- MediaPipe Face Landmarker model file

## 10. Expected Output

When the application runs:

- The webcam feed is displayed.
- Facial landmarks are tracked.
- Eye closure and yawning are monitored.
- Blink count and blink rate are updated.
- PERCLOS is calculated.
- Monitoring information is displayed on the dashboard.
- An audio warning is generated when prolonged eye closure indicates possible drowsiness.

## 11. Applications

- Driver fatigue monitoring
- Road safety systems
- Driver assistance applications
- Fatigue monitoring research
- Computer vision-based safety applications

## 12. Future Enhancements

- Head pose estimation for detecting head nodding.
- Integration with vehicle safety systems.
- Mobile application support.
- Improved fatigue classification using machine learning.
- Cloud-based monitoring and reporting.
- Event logging with timestamps.
- Support for different lighting conditions.

## 13. Limitations

- Detection accuracy can be affected by poor lighting.
- Face tracking may be affected by camera position and facial obstruction.
- The current implementation uses predefined EAR and MAR thresholds.
- Audio alerts using `winsound` are designed for Windows.
- This system is a fatigue-monitoring aid and should not replace proper rest or safe driving practices.

## 14. Conclusion

The Drowsiness Detection System demonstrates how computer vision can be used to monitor facial behaviour and identify possible signs of driver fatigue.

By combining eye closure detection, yawning detection, blink rate calculation, PERCLOS measurement, and audio alerts, the system provides a real-time approach to fatigue monitoring.

The project can be further extended with head pose estimation, machine learning-based fatigue classification, and integration with intelligent vehicle safety systems.

---

**Project Type:** Computer Vision / Python Application  
**Domain:** Artificial Intelligence and Road Safety  
**Developed Using:** Python, OpenCV, and MediaPipe
