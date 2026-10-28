# from flask import Flask, render_template, request
# import os
# import librosa
# import numpy as np
# import pandas as pd
# import joblib
# import shap
# from sklearn.preprocessing import StandardScaler
# import time

# app = Flask(__name__)

# # ===============================
# # Load trained CREMA-D model
# # ===============================
# model = joblib.load("model.joblib")
# explainer = shap.TreeExplainer(model)

# # ===============================
# # Load CREMA-D handcrafted features
# # ===============================
# DF = pd.read_csv("cremad_handcrafted.csv")

# FEATURE_COLS = DF.drop(columns=["class"], errors="ignore").columns.tolist()
# X_FULL = DF[FEATURE_COLS]

# # ===============================
# # Traits for KNN stabilization
# # ===============================
# DF["energy"] = DF["rms"]
# DF["pitch"] = DF["spectral_centroid"]
# DF["zcr_simple"] = DF["zcr"]

# TRAIT_COLS = ["energy", "pitch", "zcr_simple"]
# scaler = StandardScaler()
# TRAIT_MATRIX = scaler.fit_transform(DF[TRAIT_COLS])

# # ===============================
# # Upload folder
# # ===============================
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ===============================
# # Label map
# # ===============================
# LABEL_MAP = {
#     "angry": "Angry 😡",
#     "happy": "Happy 😄",
#     "sad": "Sad 😢",
#     "neutral": "Neutral 😐",
#     "fear": "Fear 😨",
#     "disgust": "Disgust 🤢"
# }

# # ===============================
# # Extract traits from live audio (FIXED)
# # ===============================
# def extract_audio_traits(audio_path):
#     y, sr = librosa.load(audio_path, duration=3)

#     energy = float(np.mean(librosa.feature.rms(y=y)))
#     pitch = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
#     zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

#     return np.array([[energy, pitch, zcr]])

# # ===============================
# # SHAP Explanation (STABLE)
# # ===============================
# def generate_shap_explanation(samples, final_pred, top_n=5):
#     try:
#         shap_values = explainer.shap_values(samples)

#         if isinstance(shap_values, list):
#             class_idx = list(model.classes_).index(final_pred)
#             shap_vals = shap_values[class_idx]
#         else:
#             shap_vals = shap_values

#         shap_vals = np.asarray(shap_vals)
#         shap_vals = np.mean(np.abs(shap_vals), axis=0)

#         feature_names = samples.columns.tolist()
#         top_idx = np.argsort(shap_vals)[-top_n:][::-1]
#         top_idx = [int(i) for i in top_idx]

#         top_features = [
#             (feature_names[i], float(shap_vals[i]))
#             for i in top_idx
#             if i < len(feature_names)
#         ]

#         explanation = (
#             f"The predicted emotion is **{LABEL_MAP.get(final_pred)}**, "
#             "primarily influenced by:\n\n"
#         )

#         for f, _ in top_features:
#             explanation += f"- {f.replace('_', ' ')}\n"

#         return explanation, top_features

#     except Exception:
#         return (
#             f"The emotion **{LABEL_MAP.get(final_pred)}** was predicted "
#             "based on overall acoustic characteristics.",
#             []
#         )

# # ===============================
# # Main Route
# # ===============================
# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     confidence = None
#     explanation = None
#     error = None
#     top_features = []

#     if request.method == "POST":
#         audio = request.files.get("audio")

#         if audio and audio.filename:
#             try:
#                 path = os.path.join(UPLOAD_FOLDER, audio.filename)
#                 audio.save(path)

#                 live_traits = extract_audio_traits(path)
#                 live_scaled = scaler.transform(live_traits)

#                 distances = np.linalg.norm(TRAIT_MATRIX - live_scaled, axis=1)
#                 nearest_idx = np.argsort(distances)[:15]

#                 nearest_samples = X_FULL.iloc[nearest_idx]
#                 preds = model.predict(nearest_samples)

#                 final_pred = pd.Series(preds).value_counts().idxmax()
#                 confidence = round(
#                     pd.Series(preds).value_counts().max() / len(preds) * 100, 2
#                 )

#                 prediction = LABEL_MAP.get(final_pred, final_pred)

#                 explanation, top_features = generate_shap_explanation(
#                     nearest_samples, final_pred
#                 )

#             except Exception:
#                 error = "Audio could not be processed. Please try a clearer recording."

#     return render_template(
#         "index.html",
#         prediction=prediction,
#         confidence=confidence,
#         human_explanation=explanation,
#         error=error,
#         top_features=top_features
#     )

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import StandardScaler
import time
import subprocess
import tempfile
import imageio_ffmpeg

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

# ======================================================
# ✅ Upload Audio Traits (KEEP SAME LOGIC)
# (This is for uploaded .wav files)
# ======================================================
def extract_audio_traits_uploaded(audio_path):
    y, sr = librosa.load(audio_path, duration=3)

    energy = float(np.mean(librosa.feature.rms(y=y)))
    pitch = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return np.array([[energy, pitch, zcr]])

# ======================================================
# ✅ Live Audio Traits (NEW & IMPROVED for recording)
# (This ONLY affects live recordings)
# ======================================================
def extract_audio_traits_live(audio_path):
    # load full audio, resample to 16k (stable)
    y, sr = librosa.load(audio_path, sr=16000)

    # remove silence at start/end
    y, _ = librosa.effects.trim(y, top_db=25)

    # minimum duration check (3 sec recommended)
    if len(y) < sr * 2:
        raise ValueError("Live audio too short. Record at least 3 seconds.")

    # normalize volume (important for mic recordings)
    y = y / (np.max(np.abs(y)) + 1e-8)

    energy = float(np.mean(librosa.feature.rms(y=y)))
    pitch = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return np.array([[energy, pitch, zcr]])

# ======================================================
# ✅ Live Prediction Voting (BEST FIX)
# Splits audio into chunks and takes majority vote
# ======================================================
def predict_live_with_voting(audio_path, k=15):
    y, sr = librosa.load(audio_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < sr * 3:
        raise ValueError("Live audio too short. Record at least 3 seconds.")

    # normalize for stable features
    y = y / (np.max(np.abs(y)) + 1e-8)

    # split into 3 chunks
    chunks = np.array_split(y, 3)
    all_preds = []

    for chunk in chunks:
        if len(chunk) < sr:  # at least 1 second
            continue

        chunk_energy = float(np.mean(librosa.feature.rms(y=chunk)))
        chunk_pitch = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr)))
        chunk_zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))

        live_traits = np.array([[chunk_energy, chunk_pitch, chunk_zcr]])
        live_scaled = scaler.transform(live_traits)

        distances = np.linalg.norm(TRAIT_MATRIX - live_scaled, axis=1)
        nearest_idx = np.argsort(distances)[:k]

        preds = model.predict(X_FULL.iloc[nearest_idx])
        all_preds.extend(preds)

    if len(all_preds) == 0:
        return "neutral", 0.0

    final_pred = pd.Series(all_preds).value_counts().idxmax()
    confidence = (pd.Series(all_preds).value_counts().max() / len(all_preds)) * 100

    return final_pred, round(confidence, 2)

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

        if audio:
            try:
                # ✅ Works for both upload and live-recording
                filename = audio.filename if audio.filename else f"record_{int(time.time())}.wav"
                path = os.path.join(UPLOAD_FOLDER, filename)
                audio.save(path)
                # Decode browser recordings to PCM before passing them to librosa.
                if os.path.splitext(path)[1].lower() in {'.webm', '.mp4', '.ogg'}:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        decoded = os.path.join(temp_dir, 'audio.wav')
                        conversion = subprocess.run(
                            [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y',
                             '-i', path, '-vn', '-ac', '1', '-ar', '16000', decoded],
                            capture_output=True, timeout=60,
                        )
                        if conversion.returncode:
                            raise ValueError('Unable to decode this recording. Please record again or upload a WAV file.')
                        y, sr = librosa.load(decoded, sr=16000)
                        import soundfile as sf
                        path = path + '.wav'
                        sf.write(path, y, sr)

                # ====================================================
                # ✅ Decide whether it's live recording or upload
                # If filename contains "record" -> treat as live audio
                # ====================================================
                if "record" in filename.lower():
                    final_pred, confidence = predict_live_with_voting(path)
                else:
                    # ✅ Upload prediction (same logic as before)
                    live_traits = extract_audio_traits_uploaded(path)
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

                # Explanation still based on nearest neighbor samples
                distances = np.linalg.norm(TRAIT_MATRIX - scaler.transform(
                    extract_audio_traits_uploaded(path)
                ), axis=1)
                nearest_idx = np.argsort(distances)[:15]
                nearest_samples = X_FULL.iloc[nearest_idx]

                explanation, top_features = generate_shap_explanation(
                    nearest_samples, final_pred
                )

            except Exception as e:
                error = str(e) or 'Audio could not be decoded. Please try a WAV file or record again.'

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
