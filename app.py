from flask import Flask, render_template, request, jsonify
from flask import redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
import json
from keras.applications.mobilenet_v2 import preprocess_input
 
app = Flask(__name__)

# Redirect root to welcome page without modifying existing '/' route definition
@app.before_request
def _redirect_root_to_welcome():
    if request.path == '/' and request.method == 'GET':
        # Allow static files and avoid interfering with other endpoints
        return redirect(url_for('welcome'))

# New welcome route
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

# Dedicated path to reach the original index page
@app.route('/start')
def start():
    return render_template('index.html')

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Load class indices for PlantVillage
with open('class_indices.json') as f:
    class_indices = json.load(f)
disease_classes = [None] * len(class_indices)
for k, v in class_indices.items():
    disease_classes[v] = k

# Load comprehensive disease info mapping
with open('disease_info.json') as f:
    disease_info = json.load(f)

def preprocess_image(image):
    # image is expected as a NumPy array in RGB
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)  # MobileNetV2 preprocessing
    img = tf.expand_dims(img, axis=0)
    return img.numpy()

def get_treatment(disease):
    # Try direct match
    if disease in disease_info:
        result = disease_info[disease]
        # Replace newlines with <br> for HTML rendering
        result['treatment'] = result['treatment'].replace('\n', '<br>')
        return result
    
    # Try replacing double with triple underscores
    alt_disease = disease.replace('__', '___')
    if alt_disease in disease_info:
        result = disease_info[alt_disease]
        # Replace newlines with <br> for HTML rendering
        result['treatment'] = result['treatment'].replace('\n', '<br>')
        return result

    # Try replacing triple with double underscores
    alt_disease2 = disease.replace('___', '__')
    if alt_disease2 in disease_info:
        result = disease_info[alt_disease2]
        # Replace newlines with <br> for HTML rendering
        result['treatment'] = result['treatment'].replace('\n', '<br>')
        return result
    
    # Fallback
    return {
        'description': 'Consult a plant pathologist for treatment or refer to local agricultural guidelines.',
        'treatment': 'No specific treatment available.',
        'supplement': 'General Plant Care Product',
        'link': 'https://www.amazon.in/s?k=plant+disease+control'
    }

# Removed dataset plant-type restriction to support updated 38-class model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return render_template('index.html', error='No files uploaded')
        
        files = request.files.getlist('image')
        
        if len(files) > 15:
            return render_template('index.html', error='You can only upload a maximum of 15 images at a time.')

        results = []

        for file in files:
            if file.filename == '':
                continue

            # Read and save the image for display
            img = Image.open(file.stream)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert image to base64 for display
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Convert to numpy array for prediction
            img_array = np.array(img)
            processed_img = preprocess_image(img_array)
            
            # Make prediction
            prediction = model.predict(processed_img, verbose=0)
            probs = prediction[0]
            top_indices = np.argsort(probs)[::-1]
            top1_idx, top2_idx = top_indices[0], top_indices[1]
            top1_conf, top2_conf = probs[top1_idx], probs[top2_idx]
            disease_index = int(top1_idx)
            disease_name = disease_classes[disease_index]

            # Strict acceptance criteria: high confidence and clear margin over second-best
            MIN_CONFIDENCE = 0.75
            MIN_MARGIN = 0.15

            if (top1_conf < MIN_CONFIDENCE) or ((top1_conf - top2_conf) < MIN_MARGIN):
                disease_name = "Unknown / Unable to detect the image."
                treatment_info = {
                    'description': 'The model is not confident about the prediction. This might not be a plant leaf from the dataset, or the image quality is poor.',
                    'treatment': 'Please try with a clearer image of a plant leaf from the supported categories in your dataset.',
                    'supplement': 'N/A',
                    'link': '#'
                }
                results.append({
                    'image': img_str,
                    'disease_name': disease_name,
                    'confidence': f"{top1_conf*100:.2f}%",
                    'treatment': treatment_info,
                    'validation_error': True,
                    'error_type': 'low_confidence'
                })
            else:
                treatment_info = get_treatment(disease_name)
                results.append({
                    'image': img_str,
                    'disease_name': disease_name,
                    'confidence': f"{top1_conf*100:.2f}%",
                    'treatment': treatment_info
                })

        if not results:
            return render_template('index.html', error='No valid files uploaded')

        return render_template('result.html', results=results)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', error='An error occurred during prediction.')

if __name__ == '__main__':
    app.run(debug=True)
