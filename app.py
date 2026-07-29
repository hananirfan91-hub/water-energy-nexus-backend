"""
Water-Energy Nexus — Flask Backend

Local:
    pip install -r requirements.txt
    python app.py

Render:
    gunicorn app:app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)

# Allow requests from your Vercel frontend
CORS(app, origins=[
    "https://water-energy-nexus-r44x.vercel.app"
])

BASE = os.path.dirname(os.path.abspath(__file__))

# Load ML model
try:
    ml_model = joblib.load(
        os.path.join(BASE, "best_ml_model.pkl")
    )
    scaler = joblib.load(
        os.path.join(BASE, "scaler.pkl")
    )
    le_target = joblib.load(
        os.path.join(BASE, "le_target.pkl")
    )

    print("[SUCCESS] ML model loaded")

except Exception as e:
    print(f"[WARNING] ML load error: {e}")
    ml_model = None
    scaler = None
    le_target = None


# Load Deep Learning model
try:
    from tensorflow.keras.models import load_model

    dl_model = load_model(
        os.path.join(BASE, "best_dl_model.h5")
    )

    print("[SUCCESS] DL model loaded")

except Exception as e:
    print(f"[WARNING] DL load error: {e}")
    dl_model = None


CLASSES = [
    "Patent",
    "Policy",
    "Project",
    "Publication"
]


# Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "classes": CLASSES,
        "ml": ml_model is not None,
        "dl": dl_model is not None
    })


# ML prediction
@app.route("/predict/ml", methods=["POST"])
def predict_ml():

    if ml_model is None:
        return jsonify({
            "error": "ML model is not loaded"
        }), 500

    try:
        data = request.get_json()

        feats = np.array(
            data["features"]
        ).reshape(1, -1)

        feats_s = scaler.transform(feats)

        pred = int(
            ml_model.predict(feats_s)[0]
        )

        prob = ml_model.predict_proba(
            feats_s
        )[0].tolist()

        return jsonify({
            "prediction": pred,
            "label": CLASSES[pred],
            "probability": prob,
            "model": "ML"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Deep Learning prediction
@app.route("/predict/dl", methods=["POST"])
def predict_dl():

    if dl_model is None:
        return jsonify({
            "error": "DL model is not loaded"
        }), 500

    try:
        data = request.get_json()

        feats = np.array(
            data["features"]
        ).reshape(1, -1)

        feats_s = scaler.transform(
            feats
        ).reshape(1, 1, -1)

        probs = dl_model.predict(
            feats_s,
            verbose=0
        )[0].tolist()

        pred = int(
            np.argmax(probs)
        )

        return jsonify({
            "prediction": pred,
            "label": CLASSES[pred],
            "probability": probs,
            "model": "DL"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Local development
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
