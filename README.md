# Plant Disease Detection Web App

A modern web application for detecting plant diseases from leaf images using deep learning and transfer learning (MobileNetV2). Trained on the PlantVillage dataset with 15 classes (Tomato, Potato, Pepper, healthy and diseased).

## Features
- Upload a leaf image and get instant disease prediction
- Shows disease name, confidence, treatment recommendation, and the analyzed image
- Uses transfer learning for high accuracy
- Supports 15 classes from PlantVillage

## Project Structure
```
PlantVillage/                # Dataset folders (one per class)
static/
  css/style.css              # Custom styles
templates/
  welcome page.              
  index.html                 # Upload page
  result.html                # Result page
app.py                       # Flask web app
train_model.py               # Model training script
class_indices.json           # Class label mapping
plant_disease_model.h5       # Trained model
requirements.txt             # Python dependencies
```

## Setup
1. **Clone the repository and navigate to the folder:**
   ```bash
   git clone <repo-url>
   cd plant-disease-detection
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Mac/Linux
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download or prepare the PlantVillage dataset:**
   - Place all class folders (e.g., `Potato___Early_blight`, `Tomato_healthy`, etc.) inside the `PlantVillage` directory.

## Training the Model
1. **Edit `train_model.py` if needed (e.g., change epochs, batch size).**
2. **Run the training script:**
   ```bash
   python train_model.py
   ```
   - This will train a MobileNetV2-based model and save `plant_disease_model.h5` and `class_indices.json`.

## Running the Web App
1. **Start the Flask app:**
   ```bash
   python app.py
   ```
2. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000/
   ```
3. **Upload a leaf image and view the prediction and treatment.**

## Troubleshooting
- **Import errors:**
  - Make sure you have activated your virtual environment and installed all requirements.
  - In VS Code, select the correct Python interpreter (from your venv).
- **Model not accurate?**
  - Try increasing epochs in `train_model.py`.
  - Use more images per class if possible.
  - Try unfreezing some layers of MobileNetV2 for fine-tuning.
- **Blank result page?**
  - Check your terminal for errors and ensure all template variables are passed.

## Credits
- PlantVillage dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
- Built with TensorFlow, Keras, Flask, and Bootstrap 
