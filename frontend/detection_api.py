#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weapon Detection API

This module provides a Flask API interface to the weapon detection model.
It loads the model once at startup and serves predictions via REST API.

Updated to use the Jetson Nano camera.
"""

import os
import time
import uuid
import json
import logging
import numpy as np
import cv2
from flask import Blueprint, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from werkzeug.utils import secure_filename
import requests
import base64

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

# Default model URL (will be updated via API)
DEFAULT_MODEL_URL = "http://localhost:8000"
# Store the model URL and status - will be updated via API
model_config = {
    "model_url": DEFAULT_MODEL_URL,
    "status": "not_connected"
}

# Try to verify the connection to the model at startup
try:
    logger.info("Verifying connection to model at startup")
    verify_model_connection()
except Exception as e:
    logger.error(f"Failed to connect to model at startup: {e}")

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

# Verify connection to the remote model API
def verify_model_connection(model_url=None):
    """Verify connection to the remote model API.
    
    Args:
        model_url: Optional URL to the model API, uses the configured URL if None
    
    Returns:
        True if connection is successful, False otherwise
    """
    if model_url is None:
        model_url = model_config["model_url"]
        
    logger.info(f"Verifying connection to model API at: {model_url}")
    
    try:
        # Try to connect to the model API (health check)
        status_url = f"{model_url.rstrip('/')}/status"
        logger.info(f"Checking status at: {status_url}")
        response = requests.get(status_url, timeout=5)
        
        if response.status_code == 200:
            logger.info("Successfully connected to model API")
            model_config["status"] = "connected"
            return True
        else:
            logger.error(f"Model API returned status code: {response.status_code}")
            model_config["status"] = "error_connection"
            return False
    except Exception as e:
        logger.error(f"Failed to connect to model API: {e}")
        model_config["status"] = "error_connection"
        return False

# Function to start the camera on the Jetson Nano
def start_jetson_camera():
    """Start the camera on the Jetson Nano.
    
    Returns:
        True if successful, False otherwise
    """
    if model_config["status"] != "connected":
        logger.error("Not connected to model API, cannot start camera")
        return False
    
    try:
        model_url = model_config["model_url"]
        camera_url = f"{model_url.rstrip('/')}/camera/start"
        
        logger.info(f"Starting camera on Jetson at: {camera_url}")
        
        response = requests.post(
            camera_url,
            headers={'X-API-Key': 'your_secure_api_key_here'},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Successfully started Jetson camera")
            return True
        else:
            logger.error(f"Failed to start Jetson camera: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error starting Jetson camera: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Function to stop the camera on the Jetson Nano
def stop_jetson_camera():
    """Stop the camera on the Jetson Nano.
    
    Returns:
        True if successful, False otherwise
    """
    if model_config["status"] != "connected":
        logger.error("Not connected to model API, cannot stop camera")
        return False
    
    try:
        model_url = model_config["model_url"]
        camera_url = f"{model_url.rstrip('/')}/camera/stop"
        
        logger.info(f"Stopping camera on Jetson at: {camera_url}")
        
        response = requests.post(
            camera_url,
            headers={'X-API-Key': 'your_secure_api_key_here'},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Successfully stopped Jetson camera")
            return True
        else:
            logger.error(f"Failed to stop Jetson camera: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error stopping Jetson camera: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Function to get a frame from the Jetson Nano camera
def get_jetson_frame():
    """Get the latest frame from the Jetson camera with detections.
    
    Returns:
        JSON response with frame and detections if successful, None otherwise
    """
    if model_config["status"] != "connected":
        logger.error("Not connected to model API, cannot get frame")
        return None
    
    try:
        model_url = model_config["model_url"]
        frame_url = f"{model_url.rstrip('/')}/camera/frame"
        
        logger.info(f"Getting frame from Jetson at: {frame_url}")
        
        response = requests.get(
            frame_url,
            headers={'X-API-Key': 'your_secure_api_key_here'},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get frame from Jetson: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting frame from Jetson: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Process image through remote model API
def detect_weapons(image_path):
    """
    Process an image through the remote model API.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of detection results
    """
    if model_config["status"] != "connected":
        logger.error("Not connected to model API, cannot perform detection")
        return None
    
    try:
        # Read the image
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Send image to model API for detection
        model_url = model_config["model_url"]
        detect_url = f"{model_url.rstrip('/')}/api/detect"
        logger.info(f"Sending detection request to: {detect_url}")
        
        response = requests.post(
            detect_url,
            files={'image': ('image.jpg', image_data, 'image/jpeg')},
            headers={'X-API-Key': 'your_secure_api_key_here'},
            timeout=30
        )
        
        logger.info(f"Detection response status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Model API returned error status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None
            
        # Parse the response
        result = response.json()
        
        # Check if the response has the expected format
        if 'detections' not in result:
            logger.error("Model API response does not have 'detections' field")
            logger.error(f"Response content: {result}")
            return None
            
        return result['detections']
    
    except Exception as e:
        logger.error(f"Error sending image to model API: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# API endpoint for status/health check
@detection_api.route('/api/status', methods=['GET'])
def api_status():
    """Check if the API and model are operational."""
    return jsonify({
        'status': 'operational' if verify_model_connection() else 'model_not_connected',
        'model_connected': verify_model_connection(),
        'api_version': '1.0.0'
    })

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

# API endpoint to set the model URL
@detection_api.route('/api/set_model_path', methods=['POST'])
def set_model_path():
    """
    API endpoint to set the model URL.
    
    Expects:
        - JSON with 'model_path' field (which will be treated as the model URL)
        
    Returns:
        JSON with success/error status
    """
    data = request.json
    if not data or 'model_path' not in data:
        return jsonify({'error': 'No model URL provided'}), 400
        
    new_url = data['model_path']
    
    # Ensure URL starts with http:// or https://
    if not (new_url.startswith('http://') or new_url.startswith('https://')):
        new_url = 'http://' + new_url

    # Ensure that the URL doesn't have double slashes at the end
    new_url = new_url.rstrip('/')
    
    # Update the model URL
    model_config["model_url"] = new_url
    model_config["status"] = "pending"
    
    # Attempt to connect to the model API
    try:
        if verify_model_connection(new_url):
            return jsonify({
                'success': True,
                'message': f'Successfully connected to model API at {new_url}',
                'status': model_config["status"]
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to connect to model API at {new_url}',
                'status': model_config["status"]
            }), 500
    except Exception as e:
        model_config["status"] = "error"
        return jsonify({
            'success': False,
            'message': f'Error connecting to model API: {str(e)}',
            'status': model_config["status"]
        }), 500

# API endpoint to get model configuration
@detection_api.route('/api/model_config', methods=['GET'])
def get_model_config():
    """
    API endpoint to get current model configuration.
    
    Returns:
        JSON with model URL and status
    """
    return jsonify({
        'model_path': model_config["model_url"],  # Keep key as model_path for frontend compatibility
        'status': model_config["status"]
    })

# Jetson camera control endpoints
@detection_api.route('/api/camera/start', methods=['POST'])
@api_key_required
def api_start_camera():
    """API endpoint to start the Jetson camera."""
    if start_jetson_camera():
        return jsonify({'success': True, 'message': 'Camera started successfully'})
    return jsonify({'error': 'Failed to start camera'}), 500

@detection_api.route('/api/camera/stop', methods=['POST'])
@api_key_required
def api_stop_camera():
    """API endpoint to stop the Jetson camera."""
    if stop_jetson_camera():
        return jsonify({'success': True, 'message': 'Camera stopped successfully'})
    return jsonify({'error': 'Failed to stop camera'}), 500

@detection_api.route('/api/camera/frame', methods=['GET'])
@api_key_required
def api_get_frame():
    """API endpoint to get the latest frame from the Jetson camera."""
    result = get_jetson_frame()
    
    if result is None:
        return jsonify({'error': 'Failed to get frame from Jetson camera'}), 500
        
    return jsonify(result)

# Error handler
@detection_api.errorhandler(Exception)
def handle_exception(e):
    """Handle exceptions in the API."""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(e)
    }), 500
