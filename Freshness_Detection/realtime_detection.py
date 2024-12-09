import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model_path = '../model/freshness_model.h5'
model = load_model(model_path)

# Parameters
image_size = (128, 128)

def predict_frame(frame):
    # Preprocess the frame
    frame_resized = cv2.resize(frame, image_size)
    frame_array = img_to_array(frame_resized) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    
    # Predict freshness
    prediction = model.predict(frame_array)
    result = 'Fresh' if np.argmax(prediction) == 0 else 'Stale'
    return result

def start_realtime_detection():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time freshness detection... Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Predict freshness
        result = predict_frame(frame)
        
        # Display the result on the frame
        cv2.putText(frame, f"Freshness: {result}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result == 'Fresh' else (0, 0, 255), 2)
        cv2.imshow("Freshness Detection", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_realtime_detection()
