from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model('model/clean_tech_model.h5')
class_names = ['biodegradable', 'recyclable', 'trash']
CONFIDENCE_THRESHOLD = 0.5  # Minimum required confidence to make a reliable prediction

# Home page route
@app.route('/')
def home():
    accuracy = 94.25  # Fixed or dynamically load if needed
    return render_template('home.html', accuracy=accuracy)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/classify')
def classify():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        max_confidence = prediction[predicted_index]

        # Use threshold to decide prediction
        if max_confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "Uncertain"
        else:
            predicted_class = class_names[predicted_index]

        # Format confidence scores
        confidence_scores = {
            class_names[i]: round(float(score) * 100, 2)
            for i, score in enumerate(prediction)
        }

        return render_template(
            'index.html',
            prediction=predicted_class,
            confidence_scores=confidence_scores,
            filename=file.filename
        )

    return redirect(url_for('classify'))

if __name__ == '__main__':
    app.run(debug=True)