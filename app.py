from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os
import json

url = "https://drive.google.com/uc?id=1kuAciB3tyYC6QLZ8dnsvUk8CExKwB38"
output = "model.h5"


if os.path.exists(output):
    os.remove(output)

gdown.download(url, output, quiet=False)

model = load_model(output, compile=False)

with open("class_names.json") as f:
    class_names = list(json.load(f).keys())

app = Flask(__name__)

@app.route('/')
def home():
    return "🌿 Plant Disease API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    filepath = "temp.jpg"
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    
    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    result = class_names[index]
    confidence = float(prediction[0][index]) * 100

    return jsonify({
        "disease": result,
        "confidence": round(confidence, 2)
    })
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
