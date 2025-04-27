#!/usr/bin/env python3
# TensorRT conversion for weapon detection model

import tensorflow.compat.v1 as tf
import tensorflow.contrib.tensorrt as trt
import os

tf.disable_v2_behavior()

def convert_to_tensorrt(model_path, precision="FP16"):
    """
    Convert TensorFlow model to TensorRT optimized model
    
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
    
    # Create TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the saved model
        tf.saved_model.loader.load(sess, ['serve'], model_path)
        
        # Get input and output tensors
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name("encoded_image_string_tensor:0")
        output_tensors = [
            graph.get_tensor_by_name("num_detections:0"),
            graph.get_tensor_by_name("detection_scores:0"),
            graph.get_tensor_by_name("detection_boxes:0"),
            graph.get_tensor_by_name("detection_classes:0")
        ]
        
        # Convert to TensorRT graph
        if precision == "FP16":
            converter = trt.TrtGraphConverter(
                input_graph_def=graph.as_graph_def(),
                nodes_blacklist=[t.split(':')[0] for t in [tensor.name for tensor in output_tensors]],
                precision_mode="FP16",
                maximum_cached_engines=16
            )
        elif precision == "FP32":
            converter = trt.TrtGraphConverter(
                input_graph_def=graph.as_graph_def(),
                nodes_blacklist=[t.split(':')[0] for t in [tensor.name for tensor in output_tensors]],
                precision_mode="FP32",
                maximum_cached_engines=16
            )
        
        trt_graph = converter.convert()
        
        # Save the TensorRT model
        output_graph_path = os.path.join(output_dir, "saved_model.pb")
        with tf.gfile.GFile(output_graph_path, "wb") as f:
            f.write(trt_graph.SerializeToString())
        
        # Save the model in SavedModel format
        builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            strip_default_attrs=True
        )
        builder.save()
        
    print(f"TensorRT model saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Path to saved model
    model_path = os.path.expanduser("~/models/research/weapon_detection/models/saved_model")
    
    # Convert to TensorRT FP16 (half precision - faster but slightly less accurate)
    optimized_model_path = convert_to_tensorrt(model_path, precision="FP16")
    
    print(f"Optimized model path: {optimized_model_path}")
    print("Now update the model_path in your detection script to use this optimized model")
