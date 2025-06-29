# This Python script contains the code for a Flask web application
# that classifies waste images using a pre-trained VGG16 model.
#
# To run this application locally:
# 1. Save this code as 'app.py' in your project's root directory (or where you run your Flask app from).
# 2. Ensure your trained model file, 'vgg16.h5', is also in the same directory as 'app.py'.
# 3. Make sure you have a 'templates' folder (containing ONLY index.html)
#    and a 'static' folder (with an 'uploads' subfolder for images) inside your Flask app's directory.
# 4. Install required Python libraries (preferably in a virtual environment):
#    pip install Flask tensorflow numpy Pillow werkzeug
# 5. Open your terminal/Anaconda Prompt, navigate to the Flask app's directory.
# 6. Activate your Python virtual environment (e.g., 'conda activate your_env_name' or '.\venv\\Scripts\\activate').
# 7. Run the application: python app.py
# 8. Access the web interface in your browser at: http://127.0.0.1:2222

# --- Required Libraries ---
import os
from flask import Flask, render_template, request, url_for, jsonify # Import jsonify for API responses
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image # Pillow library is used by load_img, good to keep it imported
from werkzeug.utils import secure_filename # For sanitizing filenames

# --- Global Configurations ---
# Define the path where uploaded images will be temporarily stored.
# This path is relative to the Flask application's root directory (e.g., /static/uploads).
UPLOAD_FOLDER = os.path.join('static', 'uploads')

# Define the target size for images expected by the VGG16 model.
# VGG16 typically expects 224x224 pixel input.
IMG_TARGET_SIZE = (224, 224)

# Define the class labels mapping for your model's output.
# IMPORTANT: This order must exactly match the order in which the ImageDataGenerator
# assigned numerical indices to your classes during model training.
# E.g., if 'Biodegradable' was index 0, 'Recyclable' was 1, 'Trash' was 2.
CLASS_LABELS = ['Biodegradable', 'Recyclable', 'Trash']

# --- Flask Application Initialization ---
app = Flask(__name__)
# Configure Flask to use the defined upload folder.
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists. If not, create it.
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    print(f"Created upload directory: {app.config['UPLOAD_FOLDER']}")


# --- Load the Pre-trained Deep Learning Model ---
# The model file (e.g., 'vgg16.h5') must be in the same directory as this 'app.py' script.
# If your model file has a different name, update 'vgg16.h5' below.
model = None # Initialize model to None
try:
    model = tf.keras.models.load_model('vgg16.h5')
    print("Deep learning model (vgg16.h5) loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load the model 'vgg16.h5'.")
    print(f"Please ensure 'vgg16.h5' is in the same directory as 'app.py'.")
    print(f"Detailed error: {e}")
    # The application will still run, but predictions will not be possible.


# --- Flask Routes ---

@app.route('/')
def index_page():
    """
    Serves the single-page application (index.html).
    All content (Home, About, Predict, Contact) is contained within this file,
    and navigation is handled by client-side JavaScript.
    """
    return render_template("index.html")

@app.route('/predict_api', methods=['POST'])
def predict_waste_api():
    """
    Handles image uploads, preprocesses them, makes a prediction using the loaded model,
    and returns a JSON response with the result. This endpoint is designed to be called
    via AJAX/Fetch from the frontend.
    """
    if 'pc_image' not in request.files:
        return jsonify({"error": "No file part in the request. Please select an image."}), 400

    f = request.files['pc_image']
    if f.filename == '':
        return jsonify({"error": "No file selected. Please choose an image to upload."}), 400

    if f:
        img_filename = f.filename
        # Sanitize filename to prevent directory traversal issues
        secure_img_filename = secure_filename(img_filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_img_filename)

        try:
            f.save(img_path)
            print(f"Image saved successfully to: {img_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            return jsonify({"error": f"Could not save the uploaded image. Details: {e}"}), 500

        if model is None:
            return jsonify({"error": "Classification model not loaded. Please check server logs."}), 500

        try:
            img = load_img(img_path, target_size=IMG_TARGET_SIZE)
            image_array = img_to_array(img)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0 # Normalize pixel values to [0, 1]

            predictions = model.predict(image_array)
            predicted_class_index = np.argmax(predictions[0])
            prediction_label = CLASS_LABELS[int(predicted_class_index)]

            print(f"Prediction for '{secure_img_filename}': {prediction_label}")

            # Return JSON response with prediction and image URL
            # url_for automatically handles static file paths for Flask
            return jsonify({
                "prediction": prediction_label,
                "uploaded_image_url": url_for('static', filename=os.path.join('uploads', secure_img_filename))
            }), 200

        except Exception as e:
            print(f"Error during image processing or prediction: {e}")
            return jsonify({"error": f"An error occurred during prediction. Please try again. Details: {e}"}), 500
    
    # Fallback for unexpected cases
    return jsonify({"error": "An unexpected error occurred."}), 500


# --- Main Entry Point for the Flask Application ---
if __name__ == '__main__':
    # Run the Flask development server.
    # debug=True: Enables debug mode, which provides detailed error messages
    # and reloads the server automatically on code changes.
    # IMPORTANT: Set debug=False in production environments for security and performance.
    # port=2222: Specifies the port for the development server.
    app.run(debug=True, port=2222)