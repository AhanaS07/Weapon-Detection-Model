#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weapon Detection API

This module provides a Flask API interface to the weapon detection model.
It loads the model once at startup and serves predictions via REST API.

Author: akashsingh
"""

import os
import time
import uuid
import json
import logging
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
from flask import Blueprint, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('detection_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Create a Blueprint for this module
detection_api = Blueprint('detection_api', __name__)

# Variable declarations
object_map = ["knife", "gun"]

# Load the TensorFlow model (once, at startup)
def load_detection_model():
    """Load the weapon detection model into memory."""
    logger.info("Loading weapon detection model...")
    
    # Model path (modify this to point to the correct model path)
    model_path = os.path.expanduser("~/models/research/weapon_detection/models/saved_model")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    logger.info(f"Using TensorFlow 1.15 model: {model_path}")
    
    # Configure TensorFlow for better performance
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.intra_op_parallelism_threads = 2
    config.inter_op_parallelism_threads = 2
    
    # Disable CUDA malloc async allocator if causing errors
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CUDA_MALLOC_ASYNC'] = '0'
    
    try:
        # First try to load with GPU
        session = tf.Session(graph=tf.Graph(), config=config)
        tf.saved_model.loader.load(session, ['serve'], model_path)
        logger.info("Model loaded successfully")
        return session
    except Exception as e:
        logger.error(f"Error loading model with GPU: {e}")
        
        # Fallback to pure CPU mode if GPU loading fails
        logger.info("Falling back to pure CPU mode...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Try again with CPU only
        config = tf.ConfigProto(device_count={'GPU': 0})
        session = tf.Session(graph=tf.Graph(), config=config)
        
        try:
            tf.saved_model.loader.load(session, ['serve'], model_path)
            logger.info("Model loaded successfully in CPU-only mode")
            return session
        except Exception as e:
            logger.error(f"Error loading model in CPU mode: {e}")
            return None

# Global session variable to store the loaded model
detection_session = load_detection_model()

# Directory to save uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Authentication decorator (simplified for example)
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        # In production, use a secure method to validate API keys
        if api_key != 'your_secure_api_key_here':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Process image through detection model
def detect_weapons(image_path):
    """
    Process an image through the weapon detection model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of detection results
    """
    if detection_session is None:
        logger.error("Model not loaded, cannot perform detection")
        return None
    
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
        detection, scores, boxes, classes = detection_session.run(
            ['num_detections:0', 'detection_scores:0', 'detection_boxes:0', 'detection_classes:0'],
            feed_dict={'encoded_image_string_tensor:0': [image_bytes]}
        )
        
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
        
        return results
    
    except Exception as e:
        logger.error(f"Error during weapon detection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# API endpoint for image detection
@detection_api.route('/api/detect', methods=['POST'])
@api_key_required
def detect_image():
    """
    API endpoint to detect weapons in an uploaded image.
    
    Expects:
        - A file upload named 'image'
        
    Returns:
        JSON with detection results
    """
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    
    # Check if the file has a name
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    # Check if the file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use JPG, JPEG or PNG.'}), 400
    
    try:
        # Secure the filename and create a unique name to prevent overwrites
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save the file
        file.save(file_path)
        logger.info(f"Image saved to {file_path}")
        
        # Process the image
        start_time = time.time()
        results = detect_weapons(file_path)
        processing_time = time.time() - start_time
        
        if results is None:
            return jsonify({'error': 'Failed to process image'}), 500
            
        # Create response
        response = {
            'success': True,
            'detections': results,
            'processing_time_ms': round(processing_time * 1000, 2),
            'image_url': f"/static/uploads/{unique_filename}"
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error handling image upload: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error processing image'}), 500

# API endpoint for status/health check
@detection_api.route('/api/status', methods=['GET'])
def api_status():
    """Check if the API and model are operational."""
    return jsonify({
        'status': 'operational' if detection_session is not None else 'model_not_loaded',
        'model_loaded': detection_session is not None,
        'api_version': '1.0.0'
    })

# Webhook endpoint for notifications
@detection_api.route('/api/webhook/register', methods=['POST'])
@api_key_required
def register_webhook():
    """Register a webhook URL to receive notifications for positive detections."""
    if not request.is_json:
        return jsonify({'error': 'Content type must be application/json'}), 400
        
    data = request.get_json()
    if 'webhook_url' not in data:
        return jsonify({'error': 'webhook_url is required'}), 400
        
    # In a real implementation, you would store this webhook URL in a database
    # For this example, we'll just acknowledge receipt
    logger.info(f"Webhook URL registered: {data['webhook_url']}")
    
    return jsonify({
        'success': True,
        'message': 'Webhook registered successfully'
    }), 201

# Error handler
@detection_api.errorhandler(Exception)
def handle_exception(e):
    """Handle exceptions in the API."""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(e)
    }), 500 