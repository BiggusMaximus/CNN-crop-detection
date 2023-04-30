import time
import cv2
import numpy as np
import tensorflow as tf
import PIL.Image as Image

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="Salinan DLmodel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the webcam.
TIME_DELAY = 5
COUNT = 0
class_name = ["GLS", "CR", "NLB", "H"]
cap = cv2.VideoCapture(0)

while True:
    # Wait for 5 seconds.
    time.sleep(TIME_DELAY)

    # Capture a frame from the webcam.
    ret, frame = cap.read()

    # Preprocess the input frame.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((256, 256))
    input_data = np.array(image, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    # Run inference.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted class and confidence score.
    predicted_class = np.argmax(output_data[0])
    confidence_score = output_data[0][predicted_class]

    # Print the predicted class and confidence score.
    if confidence_score >= 0.8:
        print(
            f"Class: {class_name[predicted_class]}, Confidence: {confidence_score:.2f}")
        cv2.putText(frame, f"Class: {class_name[predicted_class]}, Confidence: {confidence_score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print("TIDAK ADA DAUN!")
        cv2.putText(frame, "TIDAK ADA DAUN", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imwrite(f"IMAGE_{COUNT}S.jpg", frame)
    COUNT += 5

# Release the capture and close the window.
cap.release()
cv2.destroyAllWindows()
