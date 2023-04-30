import tensorflow as tf
import numpy as np
import PIL.Image as Image
import cv2

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="Salinan DLmodel.tflite")
interpreter.allocate_tensors()
class_name = ["GLS", "CR", "NLB", ""]

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess input image.
# image = Image.open("NLB.jpg").resize((256, 256))
image = cv2.imread("NLB.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
input_data = np.array(image, dtype=np.float32)
# input_data = (input_data - 127.5) / 127.5  # Normalize to [-1, 1]
input_data = np.expand_dims(input_data, axis=0)

# Run inference.
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print predicted class and confidence score.
predicted_class = np.argmax(output_data[0])
confidence_score = output_data[0][predicted_class]
print(
    f"Predicted class: {class_name[predicted_class]}, Confidence score: {confidence_score}")
