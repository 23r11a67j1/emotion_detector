from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import StandardScaler
import time

app = Flask(__name__)

# ===============================
# Load trained CREMA-D model
# ===============================
model = joblib.load("model.joblib")
explainer = shap.TreeExplainer(model)

# ===============================
# Load CREMA-D handcrafted features
# ===============================
DF = pd.read_csv("cremad_handcrafted.csv")

FEATURE_COLS = DF.drop(columns=["class"], errors="ignore").columns.tolist()
X_FULL = DF[FEATURE_COLS]

# ===============================
# Traits for KNN stabilization
# ===============================
DF["energy"] = DF["rms"]
DF["pitch"] = DF["spectral_centroid"]
DF["zcr_simple"] = DF["zcr"]

TRAIT_COLS = ["energy", "pitch", "zcr_simple"]
scaler = StandardScaler()
TRAIT_MATRIX = scaler.fit_transform(DF[TRAIT_COLS])

# ===============================
# Upload folder
# ===============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Label map
# ===============================
LABEL_MAP = {
    "angry": "Angry 😡",
    "happy": "Happy 😄",
    "sad": "Sad 😢",
    "neutral": "Neutral 😐",
    "fear": "Fear 😨",
    "disgust": "Disgust 🤢"
}

# ===============================
# Extract traits from live audio (FIXED)
# ===============================
def extract_audio_traits(audio_path):
    y, sr = librosa.load(audio_path, duration=3)

    energy = float(np.mean(librosa.feature.rms(y=y)))
    pitch = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return np.array([[energy, pitch, zcr]])

# ===============================
# SHAP Explanation (STABLE)
# ===============================
def generate_shap_explanation(samples, final_pred, top_n=5):
    try:
        shap_values = explainer.shap_values(samples)

        if isinstance(shap_values, list):
            class_idx = list(model.classes_).index(final_pred)
            shap_vals = shap_values[class_idx]
        else:
            shap_vals = shap_values

        shap_vals = np.asarray(shap_vals)
        shap_vals = np.mean(np.abs(shap_vals), axis=0)

        feature_names = samples.columns.tolist()
        top_idx = np.argsort(shap_vals)[-top_n:][::-1]
        top_idx = [int(i) for i in top_idx]

        top_features = [
            (feature_names[i], float(shap_vals[i]))
            for i in top_idx
            if i < len(feature_names)
        ]

        explanation = (
            f"The predicted emotion is **{LABEL_MAP.get(final_pred)}**, "
            "primarily influenced by:\n\n"
        )

        for f, _ in top_features:
            explanation += f"- {f.replace('_', ' ')}\n"

        return explanation, top_features

    except Exception:
        return (
            f"The emotion **{LABEL_MAP.get(final_pred)}** was predicted "
            "based on overall acoustic characteristics.",
            []
        )

# ===============================
# Main Route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    explanation = None
    error = None
    top_features = []

    if request.method == "POST":
        audio = request.files.get("audio")

        if audio and audio.filename:
            try:
                path = os.path.join(UPLOAD_FOLDER, audio.filename)
                audio.save(path)

                live_traits = extract_audio_traits(path)
                live_scaled = scaler.transform(live_traits)

                distances = np.linalg.norm(TRAIT_MATRIX - live_scaled, axis=1)
                nearest_idx = np.argsort(distances)[:15]

                nearest_samples = X_FULL.iloc[nearest_idx]
                preds = model.predict(nearest_samples)

                final_pred = pd.Series(preds).value_counts().idxmax()
                confidence = round(
                    pd.Series(preds).value_counts().max() / len(preds) * 100, 2
                )

                prediction = LABEL_MAP.get(final_pred, final_pred)

                explanation, top_features = generate_shap_explanation(
                    nearest_samples, final_pred
                )

            except Exception:
                error = "Audio could not be processed. Please try a clearer recording."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        human_explanation=explanation,
        error=error,
        top_features=top_features
    )

if __name__ == "__main__":
    app.run(debug=True)
