from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ===============================
# Load model and dataset
# ===============================
model = joblib.load("model.joblib")
explainer = shap.TreeExplainer(model)

DF = pd.read_csv("all_handcrafted_data_tess.csv")

FEATURE_COLS = DF.drop(
    columns=["class", "path", "source"], errors="ignore"
).columns.tolist()

X_FULL = DF[FEATURE_COLS]
y_FULL = DF["class"]

# Precompute simple traits for dataset
DF["energy"] = DF["rms"]
DF["pitch"] = DF["pitch_mean"]
DF["zcr_simple"] = DF["zcr"]

TRAIT_COLS = ["energy", "pitch", "zcr_simple"]

scaler = StandardScaler()
TRAIT_MATRIX = scaler.fit_transform(DF[TRAIT_COLS])

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
# Extract traits from live audio
# ===============================
def extract_audio_traits(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)

    return np.array([
        np.mean(librosa.feature.rms(y=y)),
        np.mean(librosa.yin(y, fmin=50, fmax=500)),
        np.mean(librosa.feature.zero_crossing_rate(y))
    ]).reshape(1, -1)

# ===============================
# Explainable AI text
# ===============================
def generate_explanation(emotion, traits):
    energy, pitch, zcr = traits.flatten()

    reasons = []
    if energy > 0.05:
        reasons.append("high vocal energy")
    if pitch > 200:
        reasons.append("elevated pitch")
    if zcr > 0.1:
        reasons.append("rapid speech articulation")

    if not reasons:
        return f"The speech shows calm and stable acoustic patterns, leading to a {emotion} prediction."

    return (
        f"The emotion was predicted as {emotion} because the speech exhibits "
        f"{', '.join(reasons)}, which are strongly associated with this emotion."
    )

# ===============================
# Main route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    explanation = None

    if request.method == "POST":
        audio = request.files.get("audio")

        if audio and audio.filename:
            path = os.path.join(UPLOAD_FOLDER, audio.filename)
            audio.save(path)

            live_traits = extract_audio_traits(path)
            live_traits_scaled = scaler.transform(live_traits)

            # 🔑 Find K nearest real samples
            distances = np.linalg.norm(TRAIT_MATRIX - live_traits_scaled, axis=1)
            nearest_idx = np.argsort(distances)[:15]

            nearest_samples = X_FULL.iloc[nearest_idx]
            preds = model.predict(nearest_samples)

            # Majority vote
            final_pred = pd.Series(preds).value_counts().idxmax()
            confidence = round(
                (pd.Series(preds).value_counts().max() / len(preds)) * 100, 2
            )

            prediction = LABEL_MAP.get(final_pred, final_pred)

            explanation = generate_explanation(prediction, live_traits)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        human_explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)
