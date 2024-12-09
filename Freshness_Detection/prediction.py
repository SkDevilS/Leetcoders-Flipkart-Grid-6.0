from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Load the trained model
model = load_model('freshness_model.h5')

# Test with a sample image
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Fresh' if np.argmax(prediction) == 0 else 'Stale'

# Provide a test image path
test_image_path = 'dataset/fresh/sample_image.jpg'
result = predict_image(test_image_path)
print(f"Prediction: {result}")
