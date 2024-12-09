import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('freshness_model.h5')  # Path to the saved model
image_size = (128, 128)  # Input size for the model

# Define labels
labels = {0: 'Fresh', 1: 'Stale'}

# Function to make predictions on a frame
def predict_freshness(frame):
    # Resize the frame to match the model's input size
    resized_frame = cv2.resize(frame, image_size)
    img_array = img_to_array(resized_frame) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(img_array)
    class_label = np.argmax(prediction)
    confidence = prediction[0][class_label]

    return labels[class_label], confidence

# Open webcam for real-time prediction
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Predict the freshness of the object in the frame
    label, confidence = predict_freshness(frame)

    # Display the prediction on the frame
    text = f"{label} ({confidence*100:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Freshness Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
