import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="deepfake_efficientnet_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== MODEL X-RAY ===")
print(f"Expected Input Shape:  {input_details[0]['shape']}")
print(f"Expected Output Shape: {output_details[0]['shape']}\n")