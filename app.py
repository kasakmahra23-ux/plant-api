import os

# Download model from Google Drive (only once)
if not os.path.exists("final_plant_model.keras"):
    url = "https://drive.google.com/uc?id=1_O0vosEt-AHD5_fEGFnBABgafb7WXV39"
    gdown.download(url, "final_plant_model.keras", quiet=False)
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

app = Flask(__name__)

# Load model
model = load_model('final_plant_model.keras')

# Load class names
with open('class_names.json') as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

@app.route('/')
def home():
    return "🌿 Plant Disease API Live!"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    filepath = 'temp.jpg'
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    return jsonify({
        "disease": class_names[index],
        "confidence": float(prediction[0][index]) * 100
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
