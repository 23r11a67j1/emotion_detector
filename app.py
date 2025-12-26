from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import shap

app = Flask(__name__)

# ===============================
# Load Model
# ===============================
model = joblib.load("model.joblib")
explainer = shap.TreeExplainer(model)

# ===============================
# Config
# ===============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

LABEL_MAP = {
    "female_neutral": "Neutral 😐",
    "female_angry": "Angry 😡",
    "female_happy": "Happy 😄",
    "female_sad": "Sad 😢",
    "female_fear": "Fear 😨",
    "female_disgust": "Disgust 🤢",
    "female_surprise": "Surprise 😲"
}

# ===============================
# Feature Extraction
# ===============================
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)

    traits = {
        "Energy": float(np.mean(librosa.feature.rms(y=y))),
        "Pitch": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "ZCR": float(np.mean(librosa.feature.zero_crossing_rate(y)))
    }

    features = {
        "rms": traits["Energy"],
        "zcr": traits["ZCR"],
        "spectral_centroid": traits["Pitch"],
        "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    }

    return features, traits

# ===============================
# Utility: SAFE SCALAR CONVERSION
# ===============================
def to_scalar(value):
    """
    Converts ANY numpy / list / array value into a safe float.
    """
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.item())
    return float(arr.mean())

# ===============================
# Main Route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    explanation = None

    if request.method == "POST":
        audio = request.files.get("audio")

        if audio and audio.filename:
            audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
            audio.save(audio_path)

            # Extract features
            features, traits = extract_features(audio_path)
            df = pd.DataFrame([features])
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)

            # Predict
            raw_pred = model.predict(df)[0]
            probs = model.predict_proba(df)[0]
            confidence = round(float(np.max(probs)) * 100, 2)

            prediction = LABEL_MAP.get(raw_pred, raw_pred)

            # Natural adjustment
            if confidence < 60:
                if traits["Energy"] < 0.02:
                    prediction = "Sad 😢"
                elif traits["Energy"] > 0.08 and traits["Pitch"] > 2000:
                    prediction = "Angry 😡"
                elif traits["Energy"] > 0.05:
                    prediction = "Happy 😄"
                else:
                    prediction = "Neutral 😐"

            # ===============================
            # EXPLAINABLE AI (SHAP) — FINAL SAFE VERSION
            # ===============================
            shap_values = explainer.shap_values(df)

            if isinstance(shap_values, list):
                class_index = list(model.classes_).index(raw_pred)
                shap_vals = shap_values[class_index]
            else:
                shap_vals = shap_values

            shap_vals = np.asarray(shap_vals)

            if shap_vals.ndim > 1:
                shap_vals = shap_vals[0]

            # Convert EVERY value safely to float
            shap_scalar_values = [
                to_scalar(v) for v in shap_vals
            ]

            top_features = sorted(
                zip(df.columns.tolist(), shap_scalar_values),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            explanation = [
                f"{feature} {'↑' if value > 0 else '↓'}"
                for feature, value in top_features
            ]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation
    )

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
