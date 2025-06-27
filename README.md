# Weapon-Detection-Model - Mobile SSDNet based AI model to detect weapons from a connected webcamera and send Alert notifications

# Project Overview

Weapon-Detection-Model is an AI-powered system designed to detect weapons (specifically knives and guns) from images or a live camera feed. It uses a Mobile SSDNet-based model and provides both a web-based frontend and a backend model server. The system can run locally or in Docker, and is optimized for deployment on devices like the Jetson Nano.

# How the Project Works

**1. Model Server (model_server.py)**
Purpose : Runs the weapon detection model on a Jetson Nano (or similar device) and exposes HTTP endpoints for detection requests.

Key Features:
- Loads a TensorFlow model for weapon detection.
- Accepts image uploads via HTTP POST requests for inference.
- Can capture frames directly from a connected camera.
- Provides endpoints for: Starting/stopping the camera stream, Fetching the latest frame, Running detection on uploaded images.
- Uses a queue to buffer camera frames for efficient processing.
- Logs all activity for debugging and monitoring.

**2. Frontend (frontend/ folder)**
Purpose : Provides a user-friendly web interface for interacting with the weapon detection system.

Key Components: 

app.py:
- Flask web app that serves the frontend.
- Handles user authentication (simple admin login).
- Streams detection results to the browser in real-time.
- Integrates with the backend model server via HTTP.

weapon_detection_system.py:
- Orchestrates the detection workflow.
- Operates in Live Mode (Connects to the model server for real-time detection from a camera)
- Draws bounding boxes on detected weapons and prepares data for the frontend.

detection_api.py:
- Exposes REST API endpoints for detection.
- Handles image uploads, calls the model server, and returns detection results.
- Implements rate limiting and security checks.

Templates (templates/):
- HTML files for login, detection dashboard, and results display.
- Includes a live feed, processed images, and detection statistics.

Static Folders: For storing uploaded and processed images, and alarm sounds.

**3. User Workflow**

Setup:
- Install dependencies (pip install -r requirements.txt).
- Run the model server (python model_server.py) on the Jetson Nano or server.
- Start the frontend (python app.py or via Docker).

Usage:
- Log in to the web interface.
- Upload images or view the live camera feed.
- The system detects weapons, draws bounding boxes, and displays results in real-time.
- If a weapon is detected, an alarm can be triggered and notifications can be sent.

# Key Technologies Used
Backend: TensorFlow, Flask, OpenCV, NumPy, threading, REST APIs.
Frontend: Flask, HTML/CSS (Bootstrap), JavaScript.
Deployment: Docker support, Jetson Nano optimization.
