"""
Water-Energy Nexus — Flask Backend
Run: pip install flask flask-cors joblib scikit-learn tensorflow
Then: python app.py
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os

app = Flask(__name__)
CORS(app, origins=["https://water-energy-nexus-r44x.vercel.app"])

BASE = os.path.dirname(__file__)
try:
    ml_model  = joblib.load(os.path.join(BASE, 'best_ml_model.pkl'))
    scaler    = joblib.load(os.path.join(BASE, 'scaler.pkl'))
    le_target = joblib.load(os.path.join(BASE, 'le_target.pkl'))
    print("[SUCCESS] ML model loaded")
except Exception as e:
    print(f"[WARNING] ML load error: {e}"); ml_model = None

try:
    from tensorflow.keras.models import load_model
    dl_model = load_model(os.path.join(BASE, 'best_dl_model.h5'))
    print("[SUCCESS] DL model loaded")
except Exception as e:
    print(f"[WARNING] DL load error: {e}"); dl_model = None

CLASSES = ['Patent', 'Policy', 'Project', 'Publication']  # alphabetical after LabelEncoder

@app.route('/')
def home():
    return jsonify({"status": "running", "classes": CLASSES,
                    "ml": bool(ml_model), "dl": bool(dl_model)})

@app.route('/predict/ml', methods=['POST'])
def predict_ml():
    try:
        feats = np.array(request.json['features']).reshape(1, -1)
        feats_s = scaler.transform(feats)
        pred = int(ml_model.predict(feats_s)[0])
        prob = ml_model.predict_proba(feats_s)[0].tolist()
        return jsonify({"prediction": pred, "label": CLASSES[pred],
                        "probability": prob, "model": "ML"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/dl', methods=['POST'])
def predict_dl():
    try:
        feats = np.array(request.json['features']).reshape(1, -1)
        feats_s = scaler.transform(feats).reshape(1, 1, -1)
        probs = dl_model.predict(feats_s)[0].tolist()
        pred = int(np.argmax(probs))
        return jsonify({"prediction": pred, "label": CLASSES[pred],
                        "probability": probs, "model": "DL"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
