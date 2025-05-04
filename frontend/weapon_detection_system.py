import os
import glob
import tensorflow as tf
import io
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import pandas as pd
import time
import json
import cv2
import requests
import threading
import base64
import numpy as np
from io import BytesIO

# A global list to store processed image data (to simulate the live feed)
processed_images_data = []
# Flag to use model API instead of static images
use_model_api = False
# Flag to control the camera thread
camera_running = False
# Last detection timestamp to avoid too frequent alerts
last_detection_time = 0
# Minimum time between detections (seconds)
min_detection_interval = 2

def set_use_model_api(value):
    """Set whether to use the model API or static images"""
    global use_model_api
    use_model_api = value

def draw_bounding_boxes(image, bboxes, labels, confidences=None):
    """Draw bounding boxes on the image and return it
    
    Args:
        image: PIL Image or numpy array
        bboxes: List of bounding boxes [(xmin, ymin), (xmax, ymax)]
        labels: List of labels for each box
        confidences: Optional list of confidence scores
        
    Returns:
        PIL Image with boxes drawn
    """
    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_pil = image
        
    draw = ImageDraw.Draw(image_pil)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if isinstance(bbox[0], tuple):
            (xmin, ymin), (xmax, ymax) = bbox
        else:
            xmin, ymin, xmax, ymax = bbox
            
        # Draw box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        
        # Draw label with confidence if available
        if confidences and i < len(confidences):
            label_text = f"{label} ({confidences[i]:.1f}%)"
        else:
            label_text = label
            
        draw.text((xmin, ymin - 10), label_text, fill="red")
        
    return image_pil

def jetson_camera_thread_function():
    """
    Function to get frames from Jetson camera API and process them.
    This function now connects to the Jetson Nano camera endpoints.
    """
    global camera_running, last_detection_time, processed_images_data
    
    print("Starting feed from Jetson Nano camera")
    
    # Start the Jetson camera
    try:
        response = requests.post(
            'http://localhost:5001/api/camera/start',
            headers={'X-API-Key': 'your_secure_api_key_here'}
        )
        if response.status_code != 200:
            print(f"Failed to start Jetson camera: {response.status_code}")
            camera_running = False
            return
    except Exception as e:
        print(f"Error starting Jetson camera: {e}")
        camera_running = False
        return
        
    print("Jetson camera started successfully")
    
    while camera_running:
        # Only process frames at a reasonable interval
        time.sleep(0.2)
        
        # Get current time
        current_time = time.time()
        
        # Skip if we've detected something too recently
        if current_time - last_detection_time < min_detection_interval:
            continue
            
        # Get frame from Jetson camera
        try:
            response = requests.get(
                'http://localhost:5001/api/camera/frame',
                headers={'X-API-Key': 'your_secure_api_key_here'}
            )
            
            if response.status_code != 200:
                print(f"Failed to get frame from Jetson: {response.status_code}")
                time.sleep(1)  # Wait longer before retrying
                continue
                
            # Process the response
            result = response.json()
            
            # If there are detections in the response
            if 'detections' in result and len(result['detections']) > 0:
                # Update detection timestamp
                last_detection_time = current_time
                
                # Convert base64 frame to image
                frame_data = base64.b64decode(result['frame'])
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # Save the detection image
                detection_time = time.strftime('%Y%m%d_%H%M%S')
                filename = f"detection_{detection_time}.jpg"
                output_path = os.path.join('static', 'processed_images', filename)
                cv2.imwrite(output_path, frame)
                
                # Create image info for frontend
                image_info = {
                    "file_name": filename,
                    "view_url": f"/static/processed_images/{filename}",
                    "date": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "detections": result['detections']
                }
                
                # Add to processed images
                processed_images_data.append(image_info)
                
        except Exception as e:
            print(f"Error in Jetson camera thread: {e}")
            time.sleep(1)  # Wait longer before retrying
            
    # Stop the Jetson camera
    try:
        requests.post(
            'http://localhost:5001/api/camera/stop',
            headers={'X-API-Key': 'your_secure_api_key_here'}
        )
        print("Jetson camera stopped")
    except Exception as e:
        print(f"Error stopping Jetson camera: {e}")

def xml_to_csv(path):
    """Converts XML annotations to CSV format for easier processing."""
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, '*.xml')):  # Ensure directory path
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            # Ensure that the necessary elements exist before accessing them
            bndbox = member.find('bndbox')
            if bndbox is not None:
                try:
                    xmin = int(bndbox.find('xmin').text) if bndbox.find('xmin') is not None else 0
                    ymin = int(bndbox.find('ymin').text) if bndbox.find('ymin') is not None else 0
                    xmax = int(bndbox.find('xmax').text) if bndbox.find('xmax') is not None else 0
                    ymax = int(bndbox.find('ymax').text) if bndbox.find('ymax') is not None else 0
                except ValueError:
                    continue  # Skip if the values are invalid (e.g., cannot be converted to int)

                # Check if the required fields are present
                if xmin > 0 and ymin > 0 and xmax > 0 and ymax > 0:
                    value = (
                        root.find('filename').text,
                        int(root.find('size/width').text),
                        int(root.find('size/height').text),
                        member.find('name').text,
                        xmin, ymin, xmax, ymax
                    )
                    xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def process_image_for_feed(filename, xml_df, images_folder, output_img_folder):
    """Processes each image and adds bounding boxes to it."""
    image_df = xml_df[xml_df['filename'] == filename]
    bboxes = []
    labels = []

    for _, row in image_df.iterrows():
        bbox = [(row['xmin'], row['ymin']), (row['xmax'], row['ymax'])]
        bboxes.append(bbox)
        labels.append(row['class'])

    img_path = os.path.join(images_folder, filename)
    output_path = os.path.join(output_img_folder, filename)

    if os.path.exists(img_path):
        # Read image with PIL
        image = Image.open(img_path)
        # Draw bounding boxes
        img_with_boxes = draw_bounding_boxes(image, bboxes, labels)
        # Save the image
        img_with_boxes.save(output_path)

        # Track the processed image and its data
        processed_image_info = {
            "file_name": filename,
            "view_url": f"/static/processed_images/{filename}",
            "date": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        processed_images_data.append(processed_image_info)

        # Return processed image info (so it can be sent to frontend)
        return json.dumps(processed_image_info)
    return None


def weapon_detection_system():
    """Main function to process images and send the data to the frontend."""
    global camera_running, processed_images_data
    
    # Clear previous data
    processed_images_data = []
    
    if use_model_api:
        # Start the Jetson camera thread for real-time detection
        camera_running = True
        camera_thread = threading.Thread(target=jetson_camera_thread_function)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Keep yielding any new detections as they come in
        try:
            while camera_running:
                if processed_images_data:
                    # Get and remove the first item
                    image_info = processed_images_data.pop(0)
                    yield json.dumps(image_info)
                time.sleep(0.5)
        except GeneratorExit:
            # Stop camera when the generator is closed
            camera_running = False
            print("Weapon detection stopped")
    else:
        # Use the static images approach with XML files
        xml_folder = os.path.join(os.getcwd(), 'inputs')  # Path to input folder containing XML files
        xml_df = xml_to_csv(xml_folder)  # Convert XML annotations to DataFrame
        xml_df.to_csv('annotations.csv', index=False)
        print('✅ Successfully converted XML to CSV.')

        annotations_file = os.path.join(os.getcwd(), 'annotations.csv')
        images_folder = os.path.join(os.getcwd(), 'inputs')  # Folder with the images
        output_img_folder = os.path.join('static', 'processed_images')
        os.makedirs(output_img_folder, exist_ok=True)

        # Process each image one by one, add bounding boxes, and return image data
        for filename in xml_df['filename'].unique():
            processed_image_info = process_image_for_feed(filename, xml_df, images_folder, output_img_folder)
            if processed_image_info:
                yield processed_image_info  # Yield the processed image info
            time.sleep(1)  # 1-second delay before processing the next image

        print('✅ Image annotation complete.')
