import os
import glob
import tensorflow as tf
import io
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import pandas as pd
import time
import json

# A global list to store processed image data (to simulate the live feed)
processed_images_data = []


def draw_bounding_boxes(image_path, output_path, bboxes, labels):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for bbox, label in zip(bboxes, labels):
        (xmin, ymin), (xmax, ymax) = bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 10), label, fill="red")
    image.save(output_path)


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
        draw_bounding_boxes(img_path, output_path, bboxes, labels)

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
