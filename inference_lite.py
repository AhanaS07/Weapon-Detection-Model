#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Weapon Detection System for Nvidia Jetson Nano
Created on Fri Dec  4 02:44:25 2020
Modified on Sun Apr 27 2025

@author: vihan, with improvements by [Your Name]
"""
# Import packages for Tensorflow, OpenCV, and threading
import cv2
import time
import tensorflow as tf
import threading
import numpy as np
import os

# Import packages for email notifications
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json  # For reading configuration file

# Configure TensorFlow to use GPU memory efficiently on Jetson Nano
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled")
    except Exception as e:
        print(f"Error configuring GPU: {e}")

# Load TFLite model and allocate tensors
model_path = "../models/tflite_model/model.tflite"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

interpreter = tf.lite.Interpreter(model_path=model_path)
print("Model loaded successfully")

# Email Configuration
# Create a config directory if it doesn't exist
config_dir = os.path.expanduser("~/.config/weapon_detection")
config_file = os.path.join(config_dir, "email_config.json")

# Default email configuration
email_config = {
    "sender_email": "jetsondetector@gmail.com",
    "receiver_email": "vihanmurthy@gmail.com",
    "app_password": "",  # Empty by default, must be configured
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 465
}

# Create config directory and file if they don't exist
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

# Load existing config or create a new one
if os.path.exists(config_file):
    try:
        with open(config_file, 'r') as f:
            email_config.update(json.load(f))
    except Exception as e:
        print(f"Error loading email configuration: {e}")
else:
    # Create a new config file with default values
    try:
        with open(config_file, 'w') as f:
            json.dump(email_config, f, indent=4)
        print(f"Created new email configuration file at {config_file}")
        print("Please edit this file to add your app password before running detection.")
    except Exception as e:
        print(f"Error creating email configuration file: {e}")

# Check if app password is configured
if not email_config["app_password"]:
    print("\nWARNING: Email app password not configured!")
    print(f"Please edit {config_file} and add your app password.")
    print("For Gmail, you can generate an app password at https://myaccount.google.com/apppasswords")
    print("Detection will continue, but email notifications will be disabled.\n")

# Zebronics ZEB-Ultimate Pro camera configuration (1920x1080)
def initialize_camera(attempts=5):
    for attempt in range(attempts):
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            print(f"Camera initialization attempt {attempt+1}/{attempts} failed, retrying...")
            time.sleep(1)
            continue
        
        # Configure camera for Zebronics ZEB-Ultimate Pro (1920x1080)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify camera configuration
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {actual_width}x{actual_height} at {actual_fps} FPS")
        return cap
    
    print("Failed to initialize camera after multiple attempts")
    exit(1)

cap = initialize_camera()
object_map = ["knife", "gun"]
detect_timer = 0
alert_active = False
alert_end_time = 0
frame_count = 0
skip_frames = 2  # Process every 3rd frame (0, 3, 6, etc.)

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

def send_email_thread(image):
    thread = threading.Thread(target=send_email, args=(image,))
    thread.daemon = True
    thread.start()

def send_email(image=None):
    # Skip if app password is not configured
    if not email_config["app_password"]:
        print("Email notification skipped: App password not configured")
        return
    
    try:
        subject = "ALERT: Weapon Detected"
        body = f"Weapon detection system detected a potential threat at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        sender_email = email_config["sender_email"]
        receiver_email = email_config["receiver_email"]
        app_password = email_config["app_password"]
        smtp_server = email_config["smtp_server"]
        smtp_port = email_config["smtp_port"]
        
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        
        # Add body to email
        message.attach(MIMEText(body, "plain"))
        
        if image is not None:
            # Create alerts directory if it doesn't exist
            alerts_dir = os.path.expanduser("~/weapon_detection_alerts")
            if not os.path.exists(alerts_dir):
                os.makedirs(alerts_dir)
            
            # Save image with timestamp
            filename = os.path.join(alerts_dir, f"weapon_detected_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(filename, image)
            print(f"Alert image saved to {filename}")
            
            # Attach image to email
            with open(filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {os.path.basename(filename)}",
            )
            message.attach(part)
        
        text = message.as_string()
        
        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, text)
        print("Email alert sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

# Create alert window
cv2.namedWindow("ALERT", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ALERT", 800, 200)

# Infinite loop to receive frames(images) from camera source
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            time.sleep(0.5)
            continue
        
        frame_count += 1
        if frame_count % (skip_frames + 1) != 0:
            # Display the frame but skip inference
            display_frame = frame.copy()
            
            # Show alert banner if active
            if time.time() < alert_end_time:
                # Display alert window with red background
                alert_window = np.zeros((200, 800, 3), dtype=np.uint8)
                alert_window[:, :] = (0, 0, 255)  # BGR format, red color
                
                # Add text
                cv2.putText(alert_window, "WEAPON DETECTED!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.imshow("ALERT", alert_window)
                
                # Add red banner to main display
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 80), (0, 0, 255), -1)
                cv2.putText(display_frame, "WEAPON DETECTED", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            else:
                # Only try to destroy window if it exists
                try:
                    cv2.getWindowProperty("ALERT", cv2.WND_PROP_VISIBLE)
                    cv2.destroyWindow("ALERT")
                except:
                    pass
            
            cv2.imshow("Weapon Detection System", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Resize to 16:9 aspect ratio (640x360) for inference
        start_time = time.time()
        resized_img = cv2.resize(frame, (640, 360))
        
        # Prepare input tensor
        interpreter.set_tensor(input_details[0]['index'], [resized_img])
        
        # Run inference
        interpreter.invoke()
        
        # Get results
        rects = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        detections = interpreter.get_tensor(output_details[3]['index'])
        
        # Create copy for display
        display_frame = frame.copy()
        
        # Process detections
        weapon_detected = False
        for i in range(int(detections[0])):
            # Using 75% detection threshold
            if scores[0][i] * 100 > 75:
                weapon_detected = True
                per_box = rects[0][i]
                class_name = object_map[int(classes[0][i])]
                confidence = scores[0][i] * 100
                
                print(f"Detected {class_name} with {confidence:.2f}% confidence")
                
                # Convert coordinates from the resized image back to original
                y1 = int(per_box[0] * 360 * (frame.shape[0] / 360))
                x1 = int(per_box[1] * 640 * (frame.shape[1] / 640))
                y2 = int(per_box[2] * 360 * (frame.shape[0] / 360))
                x2 = int(per_box[3] * 640 * (frame.shape[1] / 640))
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}%"
                cv2.putText(display_frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Set alert for 10 seconds
                current_time = time.time()
                if current_time - detect_timer > 10:
                    print("Sending alert notification")
                    detect_timer = current_time
                    alert_end_time = current_time + 10  # 10-second alert duration
                    
                    # Capture full resolution image for notification
                    send_email_thread(frame)
        
        # Show alert if active
        if time.time() < alert_end_time:
            # Display alert window with red background
            alert_window = np.zeros((200, 800, 3), dtype=np.uint8)
            alert_window[:, :] = (0, 0, 255)  # BGR format, red color
            
            # Add text
            cv2.putText(alert_window, "WEAPON DETECTED!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow("ALERT", alert_window)
            
            # Add red banner to main display
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 80), (0, 0, 255), -1)
            cv2.putText(display_frame, "WEAPON DETECTED", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        else:
            # Only try to destroy window if it exists
            try:
                cv2.getWindowProperty("ALERT", cv2.WND_PROP_VISIBLE)
                cv2.destroyWindow("ALERT")
            except:
                pass
        
        # Show processing time
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Weapon Detection System", display_frame)
        
        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Application terminated")
