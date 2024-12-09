import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define dataset path
dataset_path = 'dataset'
image_size = (128, 128)

# Prepare data
def prepare_data():
    data = []
    labels = []
    for category in ['fresh', 'stale']:
        folder_path = os.path.join(dataset_path, category)
        label = 0 if category == 'fresh' else 1
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = load_img(img_path, target_size=image_size)
                img_array = img_to_array(img)
                data.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    
    data = np.array(data, dtype='float32') / 255.0
    labels = np.array(labels)
    return data, labels

# Prepare data and labels
data, labels = prepare_data()

# Show data shape and label distribution
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
