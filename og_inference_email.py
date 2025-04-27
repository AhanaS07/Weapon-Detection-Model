#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Modified inference script for weapon detection with email notifications
"""

# import packages for Tensorflow and openCV
import cv2
import time
import threading
import tensorflow.compat.v1 as tf
import os

# import packages for smtplib
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# variable declarations
detect_timer = 0
object_map = ["knife", "gun"]

# Path to your saved model - update this to your actual path
model_path = os.path.expanduser("~/models/research/models/saved_model")
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    print("Please update the model_path variable to point to your saved_model directory")
    exit(1)

# Email configuration
sender_email = "nanojetson40@gmail.com"  # Your Gmail address
receiver_email = "vihanmurthy@gmail.com"  # Recipient email
mass_email = "1ds22cb007@dsce.edu.in" 
app_password = "oqju vyom chdw xvaq"  # 16-character App Password
notification_interval = 10  # Seconds between email notifications

# disabling TF version 2 behaviour
tf.disable_v2_behavior()

# Email sending class with threading
class Email(threading.Thread):
    def __init__(self, detect_obj, score):
        threading.Thread.__init__(self)
        self.detect_obj = detect_obj
        self.score = score
        
    # funtion to send an email to the preferred email ID
    def run(self):
        # initializing variables
        subject = f"ALERT: {self.detect_obj} detected!"  
        body = f"A {self.detect_obj} was detected by the system with a confidence score of {self.score:.2f}%"
        
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message["Bcc"] = mass_email  # Recommended for mass emails
        
        # Add body to email
        message.attach(MIMEText(body, "plain"))
        
        filename = "detected_weapon.jpg"  # Image file name
        
        # Open image file in binary mode
        try:
            with open(filename, "rb") as attachment:
                # Add file as application/octet-stream
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            
            # Encode file in ASCII characters to send by email    
            encoders.encode_base64(part)
            
            # Add header as key/value pair to attachment part
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )
            
            # Add attachment to message and convert message to string
            message.attach(part)
        except Exception as e:
            print(f"Error attaching image: {e}")
            # Still send email even if attachment fails
            message.attach(MIMEText("\n\nError: Could not attach image.", "plain"))
        
        text = message.as_string()
        
        # Log in to server using secure context and send email
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, app_password)
                server.sendmail(sender_email, receiver_email, text)
            print(f"Email sent successfully: {subject}")
        except Exception as e:
            print(f"Error sending email: {e}")

def main():
    global detect_timer
   
    print("Starting weapon detection system...")
   
    # Build TF graph and load model into session for inference
    try:
        print(f"Loading model from {model_path}")
        session = tf.Session(graph=tf.Graph())
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
    except Exception as e:
        print(f"Error accessing camera: {e}")
        return
   
    # Initialize a display window
    cv2.namedWindow("Weapon Detection", cv2.WINDOW_NORMAL)
   
    print("System initialized and running. Press 'q' to quit.")
   
    # Infinite loop to receive frames from camera source
    try:
        while True:
            # Reading frames from camera (ret is True if read is successful)
            ret, image = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                time.sleep(1)
                continue
           
            # Resize image to half the original size for faster processing
            height, width, ch = image.shape
            image = cv2.resize(image, (int(width/2), int(height/2)), cv2.INTER_LINEAR)
           
            height, width, ch = image.shape
            start_time = time.time()
           
            # Convert image to bytes for TensorFlow input
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
           
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
            for i in range(int(detection[0])):
                # Filter frames with accuracy above 80%
                if scores[0][i] * 100 > 80:
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
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                   
                    # Add label with confidence score
                    score_percent = round(scores[0][i] * 100, 2)
                    label = f"{object_name}: {score_percent}%"
                    cv2.putText(image, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                   
                    print(f"Detected: {object_name}, Score: {score_percent}%")
                   
                    # Send notification email if enough time has passed since last detection
                    if time.time() - detect_timer > notification_interval:
                        detect_timer = time.time()
                        cv2.imwrite("detected_weapon.jpg", image)
                       
                        # Uncomment to enable email notifications
                        email_thread = Email(object_name, score_percent)
                        email_thread.start()
                        print(f"Email notification triggered for {object_name}")
           
            # Calculate and display FPS
            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
           
            # Display the output image
            cv2.imshow("Weapon Detection", image)
           
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
