import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('../model/product_model.h5')

# Load the class names (in the same order as during training)
class_names = ['Product1', 'Product2', 'Product3']

# Initialize video capture for webcam
cap = cv2.VideoCapture(0)
product_count = 0
last_detected_product = None  # Keep track of the last detected product to avoid constant miscounts

# Confidence threshold to consider a valid product detection
CONFIDENCE_THRESHOLD = 0.75

# Loop through camera frames for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (150, 150))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.expand_dims(normalized_frame, axis=0)

    # Make prediction using the model
    predictions = model.predict(reshaped_frame)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions)

    # Only update product count if confidence is above the threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        product_name = class_names[predicted_class]
        
        # Check if the detected product is different from the last detected one
        if product_name != last_detected_product:
            product_count += 1  # Increment the count only for new products
            last_detected_product = product_name
    else:
        product_name = 'No Product Detected'
        last_detected_product = None  # Reset last detected product when nothing is detected

    # Display the product name and count on the frame
    display_text = f"Product: {product_name}, Count: {product_count}"
    cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with the text
    cv2.imshow('Product Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
