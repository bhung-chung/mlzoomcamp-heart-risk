from flask import Flask, jsonify, request
import pickle
import numpy as np
import pathlib as Path
import pandas as pd

#---------- Model Loading -----------------------
BASE_DIR = Path.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'model.bin'

with MODEL_PATH.open('rb') as f_in:
    scaler, model = pickle.load(f_in)
# order of features (without target)
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

app = Flask(__name__)


@app.route("/predict", methods=["POST"])

def predict():
    data = request.get_json()
    X = pd.DataFrame([data], columns=FEATURES)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = proba >= 0.5
    result = {
    "heart_disease_probability": float(proba),
    "heart_disease": bool(pred),
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)

