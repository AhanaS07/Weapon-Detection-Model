#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weapon Detection Model Server for Jetson Nano

This script creates a lightweight Flask server that runs the weapon detection model
on a Jetson Nano and exposes HTTP endpoints for detection requests.

Updated to include direct camera capture on the Jetson Nano.

Usage:
    python model_server.py
"""

import os
import time
import uuid
import json
import logging
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
from werkzeug.utils import secure_filename
import threading
import base64
from queue import Queue
import gc  # For garbage collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Variable declarations
object_map = ["knife", "gun"]

# Model path for Jetson Nano
MODEL_PATH = os.path.expanduser("~/models/research/weapon_detection/models/saved_model")

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Camera and frame handling
camera_stream_active = False
frame_queue = Queue(maxsize=5)  # Smaller buffer for memory efficiency
camera_thread = None
skip_frames = 2  # Process every nth frame to reduce load
detect_timer = 0  # Used to limit detection frequency
notification_interval = 5  # Seconds between notifications

# Initialize TensorFlow session and load model
def load_model():
    """Load TensorFlow model optimized for Jetson Nano."""
    logger.info(f"Loading model from {MODEL_PATH}")
    
    # Configure TensorFlow for Jetson Nano
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.25  # Reduced from 0.3
    
    # Optimization for Jetson Nano
    config.intra_op_parallelism_threads = 2
    config.inter_op_parallelism_threads = 2
    
    # Environment variables for TensorFlow on Jetson
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # CRITICAL FIX: Disable CUDA Malloc Async for CUDA 10.2 compatibility
    os.environ['TF_CUDA_MALLOC_ASYNC'] = '0'
    
    # Disable TensorFlow GPU memory preallocation
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_host'
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    # CRITICAL FIX: Modify TF process state to avoid using CudaMallocAsync
    os.environ['TF_USE_CUDNN'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
    
    try:
        with tf.device('/device:GPU:0'):
            session = tf.Session(graph=tf.Graph(), config=config)
            tf.saved_model.loader.load(session, ['serve'], MODEL_PATH)
            logger.info("Model loaded successfully")
            return session
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        
        # Try CPU fallback
        logger.info("Attempting CPU fallback...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        config = tf.ConfigProto(device_count={'GPU': 0})
        session = tf.Session(graph=tf.Graph(), config=config)
        
        try:
            tf.saved_model.loader.load(session, ['serve'], MODEL_PATH)
            logger.info("Model loaded in CPU mode")
            return session
        except Exception as e:
            logger.error(f"Failed to load model in CPU mode: {e}")
            return None

# Function to start the camera
def start_camera_capture():
    """Start capturing from the Jetson camera."""
    global camera_stream_active, camera_thread
    
    if camera_stream_active:
        logger.info("Camera capture already running")
        return True
    
    camera_stream_active = True
    camera_thread = threading.Thread(target=camera_capture_thread)
    camera_thread.daemon = True
    camera_thread.start()
    logger.info("Camera capture thread started")
    return True

# Function to stop the camera
def stop_camera_capture():
    """Stop the camera capture thread."""
    global camera_stream_active
    
    if not camera_stream_active:
        logger.info("Camera not running")
        return True
    
    camera_stream_active = False
    logger.info("Camera capture stopping")
    return True

# Camera capture thread
def camera_capture_thread():
    """Thread function to capture frames from the Jetson camera."""
    global camera_stream_active, frame_queue
    
    logger.info("Initializing camera on Jetson Nano")
    
    # Initialize camera with Jetson-specific optimizations
    try:
        # First try with gstreamer pipeline for CSI camera
        try:
            gstreamer_pipeline = (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=15/1 ! "
                "nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! "
                "appsink"
            )
            cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                logger.info("CSI camera not detected, trying USB camera")
                raise Exception("CSI camera not available")
                
        except Exception as e:
            logger.info(f"CSI camera error: {e}")
            # Fall back to USB camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error("Failed to open USB camera")
                camera_stream_active = False
                return
                
            # Optimize USB camera settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus to save processing
            
        logger.info("Camera opened successfully")
        
        # Variables for frame skipping
        frame_count = 0
        last_gc_time = time.time()  # For periodic garbage collection
        
        # Main capture loop
        while camera_stream_active:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                time.sleep(0.1)  # Short delay before retry
                continue
            
            # Skip frames to reduce processing load
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
                
            # Perform garbage collection periodically
            if time.time() - last_gc_time > 30:  # Every 30 seconds
                gc.collect()
                last_gc_time = time.time()
                logger.info("Memory cleanup performed")
                
            # Clear the queue if it's full
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
                    
            # Add new frame to queue
            try:
                frame_queue.put_nowait(frame)
            except:
                pass
                
            # Brief delay to prevent CPU overuse
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Error in camera thread: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Release camera when thread stops
        if 'cap' in locals() and cap is not None:
            cap.release()
        logger.info("Camera released")

# Authentication decorator for the API (use same key as frontend)
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your_secure_api_key_here':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Process image through detection model
def detect_weapons(image_path, detection_session):
    """
    Run weapon detection on an image.
    
    Args:
        image_path: Path to image file or cv2 image
        detection_session: TensorFlow session
        
    Returns:
        List of detection results
    """
    try:
        # Check if image_path is a string (file path) or numpy array (image)
        if isinstance(image_path, str):
            # Read image from file
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image at {image_path}")
                return None
        else:
            # Use the provided image directly
            image = image_path
            
        # Get original dimensions for scaling boxes later
        height, width = image.shape[:2]
        
        # Resize image for model input (224x224)
        image_resized = cv2.resize(image, (224, 224), cv2.INTER_AREA)
        
        # Convert image to bytes for TensorFlow input
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        image_bytes = cv2.imencode('.jpg', image_resized, encode_param)[1].tobytes()
        
        # Run inference
        start_time = time.time()
        detection, scores, boxes, classes = detection_session.run(
            ['num_detections:0', 'detection_scores:0', 'detection_boxes:0', 'detection_classes:0'],
            feed_dict={'encoded_image_string_tensor:0': [image_bytes]}
        )
        inference_time = time.time() - start_time
        
        # Process results
        results = []
        num_detections = int(detection[0])
        
        for i in range(num_detections):
            # Only include detections with confidence > 40%
            if scores[0][i] * 100 > 40:
                # Get detection class
                det_class = int(classes[0][i]) - 1  # Adjust for 0-based indexing
                
                if det_class < 0 or det_class >= len(object_map):
                    object_name = f"Object-{det_class+1}"
                else:
                    object_name = object_map[det_class]
                
                # Get bounding box coordinates
                box = boxes[0][i]
                y1, x1, y2, x2 = box
                
                # Scale coordinates to original image size
                y1 = int(y1 * height)
                x1 = int(x1 * width)
                y2 = int(y2 * height)
                x2 = int(x2 * width)
                
                # Add detection to results
                results.append({
                    'class': object_name,
                    'confidence': float(scores[0][i] * 100),
                    'bbox': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }
                })
        
        logger.info(f"Detection completed in {inference_time:.3f} seconds, found {len(results)} objects")
        return results
    
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Flask routes
@app.route('/status', methods=['GET'])
def status():
    """Health check endpoint."""
    model_status = 'operational' if detection_session is not None else 'not_loaded'
    camera_status = 'active' if camera_stream_active else 'inactive'
    return jsonify({
        'status': model_status,
        'model_loaded': detection_session is not None,
        'camera_active': camera_stream_active,
        'version': '1.0.0',
        'device': 'Jetson Nano'
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint for basic health check and API info."""
    return jsonify({
        'name': 'Weapon Detection Model API',
        'version': '1.0.0',
        'endpoints': [
            '/status - Health check endpoint',
            '/api/detect - Detection endpoint (POST)',
            '/camera/start - Start camera (POST)',
            '/camera/stop - Stop camera (POST)',
            '/camera/frame - Get latest frame with detections (GET)'
        ]
    })

@app.route('/api/detect', methods=['POST'])
@api_key_required
def detect():
    """
    API endpoint to detect weapons in an uploaded image.
    
    Expects:
        - A file upload named 'image'
        
    Returns:
        JSON with detection results
    """
    if detection_session is None:
        return jsonify({'error': 'Model not loaded'}), 503
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    
    # Check if file exists
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        unique_filename = f"{unique_id}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # Process the image
        start_time = time.time()
        results = detect_weapons(file_path, detection_session)
        processing_time = time.time() - start_time
        
        # Clean up the file if needed
        try:
            os.remove(file_path)
        except:
            pass
            
        if results is None:
            return jsonify({'error': 'Detection failed'}), 500
            
        # Format response to match what the frontend expects
        response = {
            'success': True,
            'detections': results,
            'processing_time_ms': round(processing_time * 1000, 2)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing detection request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error'}), 500

@app.route('/camera/start', methods=['POST'])
@api_key_required
def start_camera():
    """API endpoint to start the camera capture thread."""
    if start_camera_capture():
        return jsonify({"success": True, "message": "Camera started"})
    return jsonify({"success": False, "message": "Failed to start camera"}), 500

@app.route('/camera/stop', methods=['POST'])
@api_key_required
def stop_camera():
    """API endpoint to stop the camera capture thread."""
    if stop_camera_capture():
        return jsonify({"success": True, "message": "Camera stopped"})
    return jsonify({"success": False, "message": "Failed to stop camera"}), 500

@app.route('/camera/frame', methods=['GET'])
@api_key_required
def get_frame():
    """API endpoint to get the latest frame from the camera with detections."""
    global detect_timer
    
    if not camera_stream_active:
        return jsonify({"error": "Camera not active"}), 400
        
    if detection_session is None:
        return jsonify({'error': 'Model not loaded'}), 503
        
    try:
        # Get the latest frame from the queue
        if frame_queue.empty():
            return jsonify({"error": "No frames available"}), 404
            
        frame = frame_queue.get()
        
        # Process the frame through the model
        start_time = time.time()
        results = detect_weapons(frame, detection_session)
        processing_time = time.time() - start_time
        
        if results is None:
            return jsonify({'error': 'Detection failed'}), 500
            
        # Draw boxes on frame for any detections
        if results and len(results) > 0:
            for detection in results:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{detection['class']}: {detection['confidence']:.1f}%"
                cv2.putText(frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check for high confidence detections
                if detection['confidence'] > 75 and time.time() - detect_timer > notification_interval:
                    detect_timer = time.time()
                    logger.info(f"High confidence detection: {detection['class']} ({detection['confidence']:.1f}%)")
                    
                    # Save the detection image
                    detection_time = time.strftime('%Y%m%d_%H%M%S')
                    detection_filename = f"detection_{detection_time}.jpg"
                    detection_path = os.path.join(UPLOAD_FOLDER, detection_filename)
                    cv2.imwrite(detection_path, frame)
                    logger.info(f"Detection image saved to {detection_path}")
        
        # Encode frame as base64 for transmission
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Format response
        response = {
            'success': True,
            'detections': results,
            'frame': frame_base64,
            'processing_time_ms': round(processing_time * 1000, 2),
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting frame: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error'}), 500

@app.route('/camera/status', methods=['GET'])
def camera_status():
    """Check if the camera is active."""
    return jsonify({
        'active': camera_stream_active,
        'queue_size': frame_queue.qsize() if not frame_queue.empty() else 0
    })

# Load the model at startup
if __name__ == '__main__':
    # Load model with more robust error handling
    try:
        detection_session = load_model()
    except Exception as e:
        logger.error(f"Critical error during model loading: {e}")
        logger.info("Attempting to continue in limited mode...")
        detection_session = None
    
    # Set up the server
    if detection_session is None:
        logger.error("Failed to load model, server starting in limited mode")
    
    # Start the Flask server - use port 8000 to avoid conflicts
    port = 8000
    logger.info(f"Starting server on port {port}")
    
    # Use a single process for stability on resource-constrained Jetson
    app.run(host='0.0.0.0', port=port, threaded=False, processes=1)
