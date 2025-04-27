#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive inference script for weapon detection with TensorRT support
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

# Use TensorRT optimized model if available
base_model_path = os.path.expanduser("~/models/research/weapon_detection/models")
tensorrt_model_path = os.path.join(base_model_path, "trt_fp16")  # TensorRT FP16 model
standard_model_path = os.path.join(base_model_path, "saved_model")  # Original model

# Try to use TensorRT model first, fall back to standard model
if os.path.exists(tensorrt_model_path):
    model_path = tensorrt_model_path
    print(f"Using TensorRT optimized model: {model_path}")
    use_tensorrt = True
else:
    model_path = standard_model_path
    print(f"Using standard model: {model_path}")
    use_tensorrt = False

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    print("Please update model_path variable")
    exit(1)

# Alert configuration
notification_interval = 10  # Seconds between notifications
skip_frames = 4  # Process every 4th frame to reduce CPU load

# Disabling TF version 2 behavior
tf.disable_v2_behavior()

# OpenCV Popup Alert class with threading
class CVPopupAlert(threading.Thread):
    def __init__(self, detect_obj, score, image):
        threading.Thread.__init__(self)
        self.detect_obj = detect_obj
        self.score = score
        # Use a smaller image for the alert to save memory
        h, w = image.shape[:2]
        self.image = cv2.resize(image, (w//2, h//2))
        
    # Function to show a popup alert with OpenCV
    def run(self):
        # Create a window name with detection details
        window_name = f"ALERT: {self.detect_obj.upper()} DETECTED!"
        
        # Add alert text to the image
        h, w = self.image.shape[:2]
        alert_text = f"ALERT: {self.detect_obj.upper()} DETECTED!"
        confidence_text = f"Confidence: {self.score:.2f}%"
        
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
        
        # Save the detection image (smaller size)
        cv2.imwrite("detected_weapon.jpg", self.image)
        
        print(f"OpenCV popup alert displayed for: {self.detect_obj}")
        
        # Wait for key press or timeout
        cv2.waitKey(5000)  # Reduced from 10000 to 5000 ms
        cv2.destroyWindow(window_name)

def get_available_operations(session):
    """Get all available operations in the graph"""
    operations = []
    for op in session.graph.get_operations():
        operations.append(op.name)
    return operations

def find_tensor_names(session):
    """Find the appropriate tensor names for detection in the model"""
    # Default tensor names in the standard model
    default_names = {
        'input': 'encoded_image_string_tensor:0',
        'num_detections': 'num_detections:0',
        'detection_scores': 'detection_scores:0',
        'detection_boxes': 'detection_boxes:0',
        'detection_classes': 'detection_classes:0'
    }
    
    # Get all operations in the graph
    operations = get_available_operations(session)
    print("Available operations:", operations[:20])  # Print first 20 for debugging
    
    # TensorRT often adds prefixes to tensor names or changes them
    # Look for tensor names containing key detection terms
    found_names = {}
    
    # Search for input tensor
    input_candidates = [op for op in operations if 'input' in op.lower() or 'image' in op.lower() or 'encoded' in op.lower()]
    if input_candidates:
        print("Input tensor candidates:", input_candidates)
        found_names['input'] = input_candidates[0] + ':0'
    else:
        found_names['input'] = default_names['input']
    
    # Search for detection tensors
    detected_outputs = []
    for op in operations:
        if 'detection' in op.lower() or 'predict' in op.lower():
            detected_outputs.append(op)
    
    # Associate detected tensors with the required outputs
    detection_mapping = {
        'num_detections': ['num_detect', 'detection_count', 'count'],
        'detection_scores': ['score', 'confidence'],
        'detection_boxes': ['box', 'bbox', 'bound'],
        'detection_classes': ['class', 'label']
    }
    
    # For each required output, find a matching tensor
    for output_name, search_terms in detection_mapping.items():
        candidates = []
        for op in detected_outputs:
            if any(term in op.lower() for term in search_terms):
                candidates.append(op)
        
        if candidates:
            found_names[output_name] = candidates[0] + ':0'
        else:
            # Try common TensorRT output patterns
            trt_candidates = [op for op in operations if output_name.replace('_', '') in op.lower()]
            if trt_candidates:
                found_names[output_name] = trt_candidates[0] + ':0'
            else:
                # Fallback to default
                found_names[output_name] = default_names[output_name]
    
    print("Using the following tensor names:")
    for key, value in found_names.items():
        print(f"  {key}: {value}")
    
    return found_names

def main():
    global detect_timer
   
    print("Starting adaptive weapon detection system...")
    
    # Configure TensorFlow for better performance
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
   
    # Build TF graph and load model into session for inference
    try:
        print(f"Loading model from {model_path}")
        session = tf.Session(graph=tf.Graph(), config=config)
        tf.saved_model.loader.load(session, ['serve'], model_path)
        print("Model loaded successfully")
        
        # Find the correct tensor names for the loaded model
        tensor_names = find_tensor_names(session)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
   
    # Attach primary camera source with preferred settings
    try:
        print("Connecting to camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        print("Camera connected successfully")
        
        # Set lower resolution to improve frame rate (720p instead of 1080p)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Set more camera properties to optimize performance
        cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30fps
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
    except Exception as e:
        print(f"Error accessing camera: {e}")
        return
   
    # Initialize a display window
    cv2.namedWindow("Weapon Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Weapon Detection", 640, 360)  # Smaller display window
   
    print("System initialized and running. Press 'q' to quit.")
    
    # Variables for frame skipping and FPS calculation
    frame_count = 0
    fps_list = []  # Store recent FPS values
    fps_update_time = time.time()
    avg_fps = 0
   
    # Infinite loop to receive frames from camera source
    try:
        while True:
            # Reading frames from camera (ret is True if read is successful)
            ret, image = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                time.sleep(0.1)
                continue
            
            # Skip frames to improve performance
            frame_count += 1
            if frame_count % skip_frames != 0:
                # Still display the frame but skip detection
                # Use a smaller display image
                display_img = cv2.resize(image, (640, 360))
                cv2.putText(display_img, f"Monitoring... FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Weapon Detection", display_img)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting application...")
                    break
                continue
           
            # Resize image for faster processing - use smaller size for inference
            start_time = time.time()
            
            # Use an even smaller resolution for inference (320x180)
            image_resized = cv2.resize(image, (320, 180), cv2.INTER_AREA)
           
            # Convert image to bytes for TensorFlow input
            # Use lower JPEG quality (80 instead of default 95) for faster encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            image_bytes = cv2.imencode('.jpg', image_resized, encode_param)[1].tobytes()
           
            # Pass the input image and obtain the inferred outputs from the tensor
            try:
                feed_dict = {tensor_names['input']: [image_bytes]}
                outputs = session.run(
                    [tensor_names['num_detections'], 
                     tensor_names['detection_scores'], 
                     tensor_names['detection_boxes'], 
                     tensor_names['detection_classes']],
                    feed_dict=feed_dict
                )
                detection, scores, boxes, classes = outputs
                
            except Exception as e:
                print(f"Error during inference: {e}")
                print("Detailed error information:")
                print(f"Tensor names being used: {tensor_names}")
                print("Try running the model in non-TensorRT mode by using standard_model_path")
                
                # Fall back to standard model if TensorRT fails
                if use_tensorrt:
                    print("Falling back to standard model...")
                    model_path = standard_model_path
                    print(f"Loading standard model from {model_path}")
                    
                    # Release current session and create a new one
                    session.close()
                    session = tf.Session(graph=tf.Graph(), config=config)
                    tf.saved_model.loader.load(session, ['serve'], model_path)
                    
                    # Get tensor names for standard model
                    tensor_names = find_tensor_names(session)
                    use_tensorrt = False
                    continue
                else:
                    print("Error with standard model. Check tensor names or model integrity.")
                    continue
           
            # Create a smaller display image
            height, width, ch = image.shape
            display_image = cv2.resize(image, (640, 360))
            display_height, display_width = display_image.shape[:2]
            
            # Scale factor between original and display image
            h_scale = display_height / height
            w_scale = display_width / width
            
            # Loop through detections with early exit after first high-confidence detection
            detections_found = False
            
            # Only process top 5 detections instead of all to save time
            max_detections = min(5, int(detection[0]))
            
            for i in range(max_detections):
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
                   
                    # Calculate bounding box coordinates on the display image
                    y1 = int(per_box[0] * height * h_scale)
                    x1 = int(per_box[1] * width * w_scale)
                    y2 = int(per_box[2] * height * h_scale)
                    x2 = int(per_box[3] * width * w_scale)
                   
                    # Draw bounding box on the display image
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                   
                    # Add label with confidence score
                    score_percent = round(scores[0][i] * 100, 1)  # Reduced decimal precision
                    label = f"{object_name}: {score_percent}%"
                    cv2.putText(display_image, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # Smaller font
                   
                    print(f"Detected: {object_name}, Score: {score_percent}%")
                   
                    # Show popup if enough time has passed since last detection
                    if time.time() - detect_timer > notification_interval:
                        detect_timer = time.time()
                        
                        # Create a popup window with the detection image (in a thread)
                        popup_thread = CVPopupAlert(object_name, score_percent, image)
                        popup_thread.daemon = True  # Make thread daemonic
                        popup_thread.start()
                        print(f"OpenCV popup alert triggered for {object_name}")
                        
                        # Only process the first high-confidence detection to save time
                        break
           
            # Calculate FPS
            process_time = time.time() - start_time
            current_fps = 1.0 / process_time if process_time > 0 else 0
            
            # Store recent FPS values (last 10)
            fps_list.append(current_fps)
            if len(fps_list) > 10:
                fps_list.pop(0)
            
            # Update average FPS every second
            if time.time() - fps_update_time > 1.0:
                avg_fps = sum(fps_list) / len(fps_list)
                fps_update_time = time.time()
            
            # Display FPS on the image
            cv2.putText(display_image, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           
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
