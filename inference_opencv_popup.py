#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized inference script for weapon detection with visual alerts

@author: ahana
"""
import cv2
import time
import threading
import tensorflow.compat.v1 as tf
import os
import numpy as np

# Variable declarations
detect_timer = 0
object_map = ["knife", "gun"]

# Path to saved model
model_path = os.path.expanduser("~/models/research/weapon_detection/models/saved_model")
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    print("Please update model_path variable")
    exit(1)

# Alert configuration
notification_interval = 10  # Seconds between notifications
skip_frames = 2  # Process every nth frame to reduce CPU load

# Disabling TF version 2 behavior
tf.disable_v2_behavior()

# OpenCV Popup Alert class with threading
class CVPopupAlert(threading.Thread):
    def __init__(self, detect_obj, score, image):
        threading.Thread.__init__(self)
        self.detect_obj = detect_obj
        self.score = score
        self.image = image.copy()  # Create a copy of the image
        
    # Function to show a popup alert with OpenCV
    def run(self):
        # Create a window name with detection details
        window_name = f"ALERT: {self.detect_obj.upper()} DETECTED!"
        
        # Add alert text to the image
        h, w = self.image.shape[:2]
        alert_text = f"ALERT: {self.detect_obj.upper()} DETECTED!"
        confidence_text = f"Confidence score: {self.score:.2f}%"
        
        # Create a slightly larger image to add text
        border = 60  # Space for text
        display_img = np.zeros((h + border, w, 3), dtype=np.uint8)
        display_img[border:, :] = self.image
        
        # Add red background for the alert
        cv2.rectangle(display_img, (0, 0), (w, border), (0, 0, 200), -1)
        
        # Add text with better visibility
        cv2.putText(display_img, alert_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_img, confidence_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create a named window that can be resized
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w, h + border)
        cv2.imshow(window_name, display_img)
        
        # Save the detection image
        cv2.imwrite("detected_weapon.jpg", self.image)
        
        print(f"OpenCV popup alert displayed for: {self.detect_obj}")
        
        # Wait for key press or timeout (10 seconds)
        cv2.waitKey(10000)
        cv2.destroyWindow(window_name)

def main():
    global detect_timer
   
    print("Starting weapon detection system...")
    
    # Configure TensorFlow for better performance
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Prevent TF from grabbing all GPU memory
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # Limit GPU memory usage
   
    # Build TF graph and load model into session for inference
    try:
        print(f"Loading model from {model_path}")
        session = tf.Session(graph=tf.Graph(), config=config)
        tf.saved_model.loader.load(session, ['serve'], model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
   
    # Attach primary camera source to the opencv VideoCapture object
    try:
        print("Connecting to camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        print("Camera connected successfully")
        
        # Set resolution for Zebronics ZEB-Ultimate Pro (1920x1080)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    except Exception as e:
        print(f"Error accessing camera: {e}")
        return
   
    # Initialize a display window
    cv2.namedWindow("Weapon Detection", cv2.WINDOW_NORMAL)
   
    print("System initialized and running. Press 'q' to quit.")
    
    # Variables for frame skipping
    frame_count = 0
   
    # Infinite loop to receive frames from camera source
    try:
        while True:
            # Reading frames from camera (ret is True if read is successful)
            ret, image = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                time.sleep(1)
                continue
            
            # Skip frames to improve performance
            frame_count += 1
            if frame_count % skip_frames != 0:
                # Still display the frame but skip detection
                cv2.putText(image, "Monitoring...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Weapon Detection", image)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting application...")
                    break
                continue
           
            # Resize image for faster processing - use smaller size for inference
            height, width, ch = image.shape
            image_resized = cv2.resize(image, (640, 360), cv2.INTER_LINEAR)  # 16:9 aspect ratio preserved
           
            start_time = time.time()
           
            # Convert image to bytes for TensorFlow input
            image_bytes = cv2.imencode('.jpg', image_resized)[1].tobytes()
           
            # Pass the input image and obtain the inferred outputs from the tensor
            try:
                detection, scores, boxes, classes = session.run(
                    ['num_detections:0', 'detection_scores:0', 'detection_boxes:0', 'detection_classes:0'],
                    feed_dict={'encoded_image_string_tensor:0': [image_bytes]}
                )
            except Exception as e:
                print(f"Error during inference: {e}")
                continue
           
            # Loop through all detections
            detections_found = False
            # Create a copy of the image for display
            display_image = image.copy()
            
            for i in range(int(detection[0])):
                # Filter frames with accuracy above 75%
                if scores[0][i] * 100 > 75:
                    detections_found = True
                   
                    # Get bounding box coordinates from the output tensor
                    per_box = boxes[0][i]
                    det_class = int(classes[0][i]) - 1  # Adjust for 0-based indexing
                   
                    if det_class < 0 or det_class >= len(object_map):
                        object_name = f"Object-{det_class+1}"
                    else:
                        object_name = object_map[det_class]
                   
                    # Calculate bounding box coordinates
                    y1 = int(per_box[0] * height)
                    x1 = int(per_box[1] * width)
                    y2 = int(per_box[2] * height)
                    x2 = int(per_box[3] * width)
                   
                    # Draw bounding box on the image
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                   
                    # Add label with confidence score
                    score_percent = round(scores[0][i] * 100, 2)
                    label = f"{object_name}: {score_percent}%"
                    cv2.putText(display_image, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                   
                    print(f"Detected: {object_name}, Score: {score_percent}%")
                   
                    # Show popup if enough time has passed since last detection
                    if time.time() - detect_timer > notification_interval:
                        detect_timer = time.time()
                        
                        # Create a popup window with the detection image
                        popup_thread = CVPopupAlert(object_name, score_percent, display_image)
                        popup_thread.start()
                        print(f"OpenCV popup alert triggered for {object_name}")
           
            # Calculate and display FPS
            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0
            cv2.putText(display_image, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
           
            # Display the output image
            cv2.imshow("Weapon Detection", display_image)
           
            # Break loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting application...")
                break
           
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Application terminated")

if __name__ == "__main__":
    main()
