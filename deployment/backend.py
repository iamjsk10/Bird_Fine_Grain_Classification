import os
import json
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
MODEL_PATH = "recovery_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Load enhanced class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed_extensions = {"png", "jpg", "jpeg", "gif"}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        file_path = "temp.jpg"
        file.save(file_path)

        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)[0]
        predicted_index = str(np.argmax(predictions))

        # Get species info with defaults
        species_info = class_labels.get(predicted_index, {
            "name": "Unknown Species",
            "habitat": "Various habitats",
            "wingspan": "Not available",
            "diet": "Not specified"
        })

        os.remove(file_path)

        return jsonify({
            "species": species_info,
            "species_id": predicted_index,
            "confidence": float(predictions[np.argmax(predictions)])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)