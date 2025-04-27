#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script for TensorFlow 1.15 model loading on Jetson Nano
This script tests model loading without the overhead of camera and other processing

@author: ahana
"""
import tensorflow.compat.v1 as tf
import os
import time
import sys

# Disable TF2 behavior
tf.disable_v2_behavior()

def check_tensorflow_gpu():
    """Check if TensorFlow can access the GPU"""
    print("TensorFlow version:", tf.__version__)
    print("Checking GPU availability...")
    
    # List physical devices
    physical_devices = tf.config.experimental.list_physical_devices()
    print("Physical devices:", physical_devices)
    
    # Alternative approach for TF 1.x
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    
    print("\nAvailable devices:")
    for device in devices:
        print(f" - {device.name} ({device.device_type})")
    
    # Check if GPU is available
    if tf.test.is_gpu_available():
        print("\nGPU is available!")
        print("GPU Device Name:", tf.test.gpu_device_name())
    else:
        print("\nGPU is NOT available. Using CPU only.")
    
    # Memory info (for Jetson)
    try:
        import subprocess
        memory_info = subprocess.check_output(['free', '-h']).decode('utf-8')
        print("\nMemory Information:")
        print(memory_info)
    except:
        print("Could not retrieve memory information")

def debug_model_loading(model_path):
    """Test loading the saved model with different configurations"""
    print(f"\nAttempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model path {model_path} does not exist!")
        return False
        
    print(f"Model directory contents: {os.listdir(model_path)}")
    
    # Try with minimal memory allocation first
    print("\nAttempt 1: Minimal memory allocation")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    
    try:
        print("Creating session...")
        with tf.Session(graph=tf.Graph(), config=config) as sess:
            print("Loading model (this may take a minute)...")
            start_time = time.time()
            tf.saved_model.loader.load(sess, ['serve'], model_path)
            load_time = time.time() - start_time
            print(f"Model loaded successfully in {load_time:.2f} seconds!")
            return True
    except tf.errors.ResourceExhaustedError as e:
        print(f"Failed with resource exhausted: {e}")
    except Exception as e:
        print(f"Failed with error: {e}")
    
    # Try with CPU only
    print("\nAttempt 2: CPU only")
    config = tf.ConfigProto(device_count={'GPU': 0})
    
    try:
        print("Creating CPU-only session...")
        with tf.Session(graph=tf.Graph(), config=config) as sess:
            print("Loading model on CPU (this may take several minutes)...")
            start_time = time.time()
            tf.saved_model.loader.load(sess, ['serve'], model_path)
            load_time = time.time() - start_time
            print(f"Model loaded successfully on CPU in {load_time:.2f} seconds!")
            return True
    except Exception as e:
        print(f"CPU loading failed with error: {e}")
    
    print("\nAll loading attempts failed.")
    return False

def main():
    # Default model path
    model_path = os.path.expanduser("~/models/research/weapon_detection/models/saved_model")
    
    # Allow command line override
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print("TensorFlow Model Debug Utility")
    print("============================")
    
    # Check TensorFlow and GPU status
    check_tensorflow_gpu()
    
    # Try to load the model
    success = debug_model_loading(model_path)
    
    if success:
        print("\nModel loading successful!")
        print("You can now try running the main detection script.")
    else:
        print("\nModel loading failed.")
        print("Suggestions:")
        print("1. Check model path and contents")
        print("2. Verify TensorFlow installation (try 'pip3 install --upgrade tensorflow==1.15.5')")
        print("3. Check GPU drivers and CUDA installation")
        print("4. Try with a simpler model first to verify environment")

if __name__ == "__main__":
    main()
