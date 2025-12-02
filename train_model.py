import os
import numpy as np
import json
import sys

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from keras import layers, Model
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    print("TensorFlow version:", tf.__version__)
    import keras
    print("Keras version:", keras.__version__)
except ImportError as e:
    print("Error importing TensorFlow/Keras:", e)
    print("Please install the requirements using: pip install -r requirements.txt")
    sys.exit(1)

try:
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    print("Error importing scikit-learn:", e)
    print("Please install scikit-learn using: pip install scikit-learn")
    sys.exit(1)

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print("Could not set memory growth for GPU")

# Configuration
IMG_SIZE = 224  # MobileNetV2's expected input size
BATCH_SIZE = 32
EPOCHS = 20
# NUM_CLASSES will be set dynamically after generators are built

# Paths
DATASET_DIR = 'PlantVillage_reliable'
MODEL_PATH = 'plant_disease_model.h5'

# Detect 38-class augmented dataset structure
AUG_ROOT = os.path.join(
    DATASET_DIR,
    'archive (1)',
    'New Plant Diseases Dataset(Augmented)',
    'New Plant Diseases Dataset(Augmented)'
)
TRAIN_DIR = os.path.join(AUG_ROOT, 'train')
VALID_DIR = os.path.join(AUG_ROOT, 'valid')
USING_BUILTIN_SPLIT = os.path.isdir(TRAIN_DIR) and os.path.isdir(VALID_DIR)

# Windows long path support
def to_windows_long_path(path):
    if os.name == 'nt':
        abs_path = os.path.abspath(path)
        if not abs_path.startswith('\\\\?\\'):
            return '\\\\?\\' + abs_path
        return abs_path
    return path

# Verify dataset directory exists
if not os.path.exists(DATASET_DIR):
    print(f"Error: Dataset directory '{DATASET_DIR}' not found!")
    print("Please make sure your dataset is in the correct location.")
    sys.exit(1)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.85, 1.15),
    channel_shift_range=15.0,
    fill_mode='nearest',
    validation_split=(0.0 if USING_BUILTIN_SPLIT else 0.2)
)

# Only preprocessing for validation
valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=(0.0 if USING_BUILTIN_SPLIT else 0.2)
)

# Prepare class list
try:
    if USING_BUILTIN_SPLIT:
        class_dirs = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
        if len(class_dirs) == 0:
            raise RuntimeError("No class folders found under train directory.")
        print(f"Detected {len(class_dirs)} class folders in augmented dataset:")
        for cls in class_dirs:
            print(f" - {cls}")

        train_generator = train_datagen.flow_from_directory(
            to_windows_long_path(TRAIN_DIR),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            classes=class_dirs
        )

        valid_generator = valid_datagen.flow_from_directory(
            to_windows_long_path(VALID_DIR),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            classes=class_dirs
        )
    else:
        # Fallback to legacy single-folder dataset with validation split
        EXCLUDE_DIRS = {"archive", "archive (1)", "misc", "_misc", "others", "other", "unlabeled"}
        all_subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        class_dirs = sorted([d for d in all_subdirs if d not in EXCLUDE_DIRS and not d.lower().endswith('.zip')])
        if len(class_dirs) == 0:
            raise RuntimeError("No valid class folders found in dataset directory.")
        print(f"Detected {len(class_dirs)} class folders (excluding {len(all_subdirs) - len(class_dirs)} non-class folders)")
        print("Classes:")
        for cls in class_dirs:
            print(f" - {cls}")

        train_generator = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            classes=class_dirs
        )

        valid_generator = valid_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            classes=class_dirs
        )
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please check your dataset directory structure.")
    sys.exit(1)

print(f"Found {train_generator.samples} training images")
print(f"Found {valid_generator.samples} validation images")
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Calculate class weights to handle imbalance
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))

# Save class indices for prediction
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Class indices saved to class_indices.json")

# Ensure disease_info.json has entries for all classes (add defaults if missing)
default_info = {
    'description': 'Detected plant class from dataset. Specific guidance pending.',
    'treatment': 'Maintain good hygiene, remove infected leaves, and apply appropriate fungicide/insecticide if necessary.',
    'supplement': 'General Plant Care Product',
    'link': 'https://www.amazon.in/s?k=plant+disease+control'
}
try:
    info_path = 'disease_info.json'
    existing = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = {}
    updated = False
    for cls_name in class_dirs:
        if cls_name not in existing:
            existing[cls_name] = default_info
            updated = True
    if updated:
        with open(info_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print("disease_info.json updated with default entries for missing classes")
except Exception as e:
    print(f"Warning: could not update disease_info.json automatically: {e}")

# Create the base model from MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model layers
base_model.trainable = False

# Add custom layers
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                       bias_regularizer=tf.keras.regularizers.l2(1e-5))(x)

# Create the model
model = Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)

print("Model created and compiled")
model.summary()

# Train the model
print("Starting initial training phase...")
try:
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=6,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=4,
                min_lr=1e-6
            )
        ]
    )
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)

# Fine-tuning phase
print("Starting fine-tuning phase...")
base_model.trainable = True
# Unfreeze more layers for better adaptation
for layer in base_model.layers[:-60]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)

# Fine-tune
try:
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=valid_generator,
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
except Exception as e:
    print(f"Error during fine-tuning: {e}")
    sys.exit(1)

# Save the model
try:
    model.save(MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
except Exception as e:
    print(f"Error saving model: {e}")
    sys.exit(1)
