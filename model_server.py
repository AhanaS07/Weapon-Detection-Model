#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weapon Detection Model Server for Jetson Nano

This script creates a lightweight Flask server that runs the weapon detection model
on a Jetson Nano and exposes HTTP endpoints for detection requests.

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

# Initialize TensorFlow session and load model
def load_model():
    """Load TensorFlow model optimized for Jetson Nano."""
    logger.info(f"Loading model from {MODEL_PATH}")
    
    # Configure TensorFlow for Jetson Nano
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3  # Adjust based on Jetson memory
    
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
        image_path: Path to image file
        detection_session: TensorFlow session
        
    Returns:
        List of detection results
    """
    try:
        # Read and resize image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image at {image_path}")
            return None
            
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
    return jsonify({
        'status': model_status,
        'model_loaded': detection_session is not None,
        'version': '1.0.0',
        'device': 'Jetson Nano'
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
