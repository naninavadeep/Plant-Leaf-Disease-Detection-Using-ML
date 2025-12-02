import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load model and class indices
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_indices.json') as f:
    class_indices = json.load(f)

# Ensure class order matches training indices
class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

# Use the same dataset and preprocessing as training; evaluate on validation split
DATASET_DIR = 'PlantVillage_reliable'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Support augmented dataset split if present
AUG_ROOT = os.path.join(
    DATASET_DIR,
    'archive (1)',
    'New Plant Diseases Dataset(Augmented)',
    'New Plant Diseases Dataset(Augmented)'
)
VALID_DIR = os.path.join(AUG_ROOT, 'valid')
USING_BUILTIN_SPLIT = os.path.isdir(VALID_DIR)

# Windows long path support
def to_windows_long_path(path: str) -> str:
    if os.name == 'nt':
        abs_path = os.path.abspath(path)
        if not abs_path.startswith('\\\\?\\'):
            return '\\\\?\\' + abs_path
        return abs_path
    return path

if USING_BUILTIN_SPLIT:
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = val_datagen.flow_from_directory(
        to_windows_long_path(VALID_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=class_names
    )
else:
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation',
        classes=class_names
    )

# Predict
Y_pred = model.predict(test_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Confusion matrix plot without seaborn to avoid extra dependency
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, fraction=0.046, pad=0.04)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight')
print('Confusion matrix saved as confusion_matrix.png')

# Classification report
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=class_names)) 