#!/usr/bin/env python3
# Fixed TensorRT conversion for TensorFlow 1.15 on Jetson Nano

import tensorflow.compat.v1 as tf
import os
import time
import numpy as np

# Check if TensorRT is available
try:
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    print("TensorRT module imported successfully")
except ImportError:
    try:
        # Alternative import path for older TensorFlow versions
        from tensorflow.contrib import tensorrt as trt
        print("TensorRT imported from tensorflow.contrib")
    except ImportError:
        print("Error: TensorRT not found in TensorFlow")
        print("Please make sure you have the correct version of TensorFlow with TensorRT support")
        exit(1)

# Disable TF2 behavior
tf.disable_v2_behavior()

def convert_to_tensorrt(model_path, precision="FP16"):
    """
    Convert TensorFlow model to TensorRT optimized model for TF 1.15
    
    Args:
        model_path: Path to saved model
        precision: Precision mode ("FP32", "FP16", or "INT8")
    
    Returns:
        Path to the optimized model
    """
    print(f"Converting model to TensorRT with {precision} precision...")
    
    # Output directory for TensorRT model
    output_dir = os.path.join(os.path.dirname(model_path), f"trt_{precision.lower()}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create configuration for TensorFlow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Load the saved model
    with tf.Session(config=config) as sess:
        # Import the meta graph
        print("Loading model...")
        tf.saved_model.loader.load(sess, ['serve'], model_path)
        
        # Get input and output tensors
        graph = tf.get_default_graph()
        
        # Try to find the input tensor
        try:
            input_tensor_name = "encoded_image_string_tensor:0"
            input_tensor = graph.get_tensor_by_name(input_tensor_name)
            print(f"Found input tensor: {input_tensor_name}")
        except KeyError:
            print("Warning: Default input tensor not found, listing available tensors:")
            # List operations to help identify input/output tensors
            ops = graph.get_operations()
            input_tensors = [op.values() for op in ops if op.type == 'Placeholder']
            if input_tensors:
                flat_list = [item for sublist in input_tensors for item in sublist]
                print("Available input tensors:")
                for t in flat_list:
                    print(f"  - {t.name}")
                input_tensor = flat_list[0]
                print(f"Using {input_tensor.name} as input")
            else:
                print("No placeholders found. Model may not be compatible with TensorRT.")
                return model_path
                
        # Try different TensorRT conversion approaches
        try:
            print("Attempting TensorRT conversion...")
            
            # Get the frozen graph def
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                [node.name.split(':')[0] for node in sess.graph.get_operations()[-5:]]  # Guess output nodes
            )
            
            # Different TensorRT API versions have different interfaces
            try:
                # Try the newer TF 1.15 API
                if hasattr(trt, 'create_inference_graph'):
                    print("Using create_inference_graph API")
                    
                    # Convert to TensorRT model
                    precision_mode = trt.TrtPrecisionMode.FP16 if precision == "FP16" else trt.TrtPrecisionMode.FP32
                    trt_graph = trt.create_inference_graph(
                        input_graph_def=frozen_graph,
                        outputs=['num_detections', 'detection_scores', 'detection_boxes', 'detection_classes'],
                        max_batch_size=1,
                        max_workspace_size_bytes=1 << 25,  # 32MB
                        precision_mode=precision_mode,
                    )
                elif hasattr(trt, 'TRTGraphConverter'):
                    print("Using TRTGraphConverter API")
                    converter = trt.TRTGraphConverter(
                        input_graph_def=frozen_graph,
                        precision_mode=precision,
                        maximum_cached_engines=1,
                    )
                    trt_graph = converter.convert()
                else:
                    # Fall back to direct session approach
                    print("TensorRT conversion APIs not found, model will be used without TensorRT optimization")
                    return model_path
                    
                # Save the TensorRT model
                output_graph_path = os.path.join(output_dir, "saved_model.pb")
                with tf.gfile.GFile(output_graph_path, "wb") as f:
                    f.write(trt_graph.SerializeToString())
                print(f"TensorRT graph saved to: {output_graph_path}")
                
                # Create a new session with the TensorRT graph
                with tf.Graph().as_default() as trt_graph_tf:
                    with tf.Session(config=config) as trt_sess:
                        # Import the TensorRT graph
                        tf.import_graph_def(trt_graph, name='')
                        
                        # Save as SavedModel format
                        builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
                        builder.add_meta_graph_and_variables(
                            trt_sess,
                            [tf.saved_model.tag_constants.SERVING],
                            strip_default_attrs=True
                        )
                        builder.save()
                
            except Exception as conversion_error:
                print(f"Error during TensorRT conversion: {conversion_error}")
                print("Using alternative approach...")
                
                # Alternative: Directly save the frozen graph with SavedModel API
                with tf.Graph().as_default() as g:
                    with tf.Session(config=config) as new_sess:
                        tf.import_graph_def(frozen_graph, name='')
                        builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
                        builder.add_meta_graph_and_variables(
                            new_sess,
                            [tf.saved_model.tag_constants.SERVING],
                            strip_default_attrs=True
                        )
                        builder.save()
                
                print(f"Saved frozen model to {output_dir}")
                
        except Exception as e:
            print(f"Error during optimization: {e}")
            print("Using original model without TensorRT optimization")
            return model_path
    
    print(f"TensorRT model conversion complete. Saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Path to saved model
    model_path = os.path.expanduser("~/models/research/weapon_detection/models/saved_model")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please update model_path variable")
        exit(1)
    
    print(f"Starting conversion of model at: {model_path}")
    
    # Convert to TensorRT FP16 (half precision - faster but slightly less accurate)
    optimized_model_path = convert_to_tensorrt(model_path, precision="FP16")
    
    print(f"Process completed. Optimized model path: {optimized_model_path}")
    print("Now update the model_path in your detection script to use this optimized model")
