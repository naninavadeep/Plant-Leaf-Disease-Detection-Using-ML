import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Paths
DATASET_DIR = 'PlantVillage'
RELIABLE_DIR = 'PlantVillage_reliable'
MODEL_PATH = 'plant_disease_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'

# Load model and class indices
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
class_names = [None] * len(class_indices)
for k, v in class_indices.items():
    class_names[v] = k

# Preprocess function (must match training)
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Ensure output directory exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Scan and move only reliable images
for class_folder in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_folder)
    if not os.path.isdir(class_path):
        continue
    reliable_class_path = os.path.join(RELIABLE_DIR, class_folder)
    ensure_dir(reliable_class_path)
    for fname in os.listdir(class_path):
        fpath = os.path.join(class_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img_array = preprocess_image(fpath)
            pred = model.predict(img_array, verbose=0)
            pred_class = class_names[np.argmax(pred[0])]
            if pred_class == class_folder:
                print(f"Moving {fpath} to reliable (predicted: {pred_class})")
                shutil.move(fpath, os.path.join(reliable_class_path, fname))
        except Exception as e:
            print(f"Error processing {fpath}: {e}") 