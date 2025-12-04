from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io

app = Flask(__name__)

# ---------------------------
# LOAD MODEL + CLASS MAPPING
# ---------------------------
model = tf.keras.models.load_model("sign_language_model_final.h5")

with open("class_mapping.json", "r") as f:
    class_names = json.load(f)

# ---------------------------
# ROUTES
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    # Load image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Adjust size to your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    predictions = model.predict(img_array)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    result = {
        "class": class_names[str(predicted_idx)],
        "confidence": round(confidence, 3)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
