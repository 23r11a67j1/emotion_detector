from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import StandardScaler
import time
from pydub import AudioSegment

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
# Convert audio to WAV format if needed
# ===============================
def convert_to_wav_if_needed(audio_path):
    """Convert audio file to WAV format if it's not already WAV"""
    try:
        # Check file extension
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        # If already WAV, return as is
        if file_ext == '.wav':
            print(f"File is already WAV format: {audio_path}")
            return audio_path
        
        # Convert to WAV using pydub
        print(f"Converting {file_ext} file to WAV...")
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # Create WAV output path
            wav_path = os.path.splitext(audio_path)[0] + '.wav'
            
            # Export as WAV
            audio.export(wav_path, format="wav")
            print(f"Successfully converted to WAV: {wav_path}")
            
            return wav_path
        except Exception as conv_error:
            # pydub requires ffmpeg for many formats including WebM
            error_msg = str(conv_error)
            if "ffmpeg" in error_msg.lower() or "ffprobe" in error_msg.lower():
                raise ValueError(
                    "WebM audio format requires ffmpeg. Please install ffmpeg:\n"
                    "Windows: Download from https://ffmpeg.org/download.html\n"
                    "Or: choco install ffmpeg (if using Chocolatey)"
                )
            else:
                raise conv_error
    except Exception as e:
        error_msg = str(e)
        print(f"Conversion error: {error_msg}")
        raise ValueError(f"Could not convert audio file: {error_msg}")

# ===============================
# Extract traits from live audio
# ===============================
def extract_audio_traits(audio_path):
    try:
        # librosa can handle various formats including webm, but may need ffmpeg
        # Try loading with librosa (it will attempt to use available backends)
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=None)
        except Exception as e1:
            # If librosa fails, try with soundfile backend
            try:
                import soundfile as sf
                y, sr = sf.read(audio_path)
                # Convert to mono if stereo
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
            except Exception as e2:
                raise ValueError(f"Failed to load audio file. Please ensure the file is a valid audio format. Librosa error: {str(e1)}, Soundfile error: {str(e2)}")
        
        # Ensure we have enough audio data (at least 1 second)
        min_duration = 1.0  # 1 second minimum
        if len(y) / sr < min_duration:
            raise ValueError(f"Audio is too short. Please record at least {min_duration} second(s).")
        
        # Extract features from the audio
        return np.array([
            np.mean(librosa.feature.rms(y=y)),
            np.mean(librosa.yin(y, fmin=50, fmax=500)),
            np.mean(librosa.feature.zero_crossing_rate(y))
        ]).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Failed to extract audio traits: {str(e)}")

# ===============================
# Explainable AI using SHAP
# ===============================
def generate_shap_explanation(nearest_samples, model, explainer, final_pred, top_n=10):
    """
    Generate explainable AI insights using SHAP values to reveal hidden factors.
    Shows which features contributed most to the emotion prediction.
    """
    try:
        # Calculate SHAP values for the nearest samples
        shap_values = explainer.shap_values(nearest_samples)
        
        # Handle multi-class SHAP output (list of arrays)
        if isinstance(shap_values, list):
            # Find the index of the predicted class
            try:
                class_keys = list(LABEL_MAP.keys())
                if final_pred in class_keys:
                    class_idx = class_keys.index(final_pred)
                else:
                    # If class not found, use the class with most predictions
                    class_idx = 0
                # Make sure index is valid
                if class_idx < len(shap_values):
                    shap_vals = shap_values[class_idx]
                else:
                    shap_vals = shap_values[0]  # Fallback to first class
            except (IndexError, ValueError):
                shap_vals = shap_values[0]  # Fallback to first class
        else:
            shap_vals = shap_values
        
        # Calculate mean SHAP values across all nearest samples
        mean_shap = np.mean(np.abs(shap_vals), axis=0)
        
        # Get feature names
        feature_names = nearest_samples.columns.tolist()
        
        # Get top N contributing features (hidden factors)
        top_indices = np.argsort(mean_shap)[-top_n:][::-1]
        # Convert numpy array to list to avoid indexing issues
        top_indices = top_indices.tolist() if hasattr(top_indices, 'tolist') else list(top_indices)
        top_features = [(feature_names[i], float(mean_shap[i])) for i in top_indices]
        
        # Generate human-readable explanation
        explanation_parts = []
        explanation_parts.append(f"The emotion '{LABEL_MAP.get(final_pred, final_pred)}' was predicted based on these key acoustic factors:\n")
        
        # Describe top contributing features
        feature_descriptions = {
            'pitch_mean': 'average pitch',
            'intensity_mean': 'average vocal intensity',
            'rms': 'root mean square energy',
            'zcr': 'zero crossing rate',
            'spectral_centroid': 'spectral centroid',
            'spectral_bandwidth': 'spectral bandwidth',
            'f1_mean': 'first formant frequency',
            'f2_mean': 'second formant frequency',
            'jitter': 'pitch jitter',
            'shimmer': 'amplitude shimmer',
            'hnr': 'harmonics-to-noise ratio',
            'mfcc': 'mel-frequency cepstral coefficient',
            'spectral_rolloff': 'spectral rolloff',
            'minF0': 'minimum fundamental frequency',
            'maxF0': 'maximum fundamental frequency',
            'avgF0': 'average fundamental frequency'
        }
        
        top_factors = []
        if len(top_features) > 0 and mean_shap.max() > 0:
            for i, (feature_name, importance) in enumerate(top_features[:5], 1):
                # Get human-readable description
                desc = feature_descriptions.get(feature_name, feature_name.replace('_', ' '))
                # Normalize importance for readability
                importance_pct = (importance / mean_shap.max()) * 100
                top_factors.append(f"{i}. {desc} ({importance_pct:.1f}% contribution)")
            
            if top_factors:
                explanation_parts.append("\nTop contributing features:\n" + "\n".join(top_factors))
        
        # Add interpretation based on feature types
        interpretation = interpret_features(top_features, final_pred, feature_descriptions)
        if interpretation:
            explanation_parts.append(f"\n{interpretation}")
        
        return "\n".join(explanation_parts), top_features
    
    except Exception as e:
        print(f"SHAP explanation error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Fallback to simple explanation
        return f"The emotion '{LABEL_MAP.get(final_pred, final_pred)}' was predicted based on acoustic analysis of the speech signal.", []


def interpret_features(top_features, predicted_emotion, feature_descriptions):
    """Provide interpretation of feature contributions"""
    interpretations = []
    
    # Check for pitch-related features
    pitch_features = [f for f, _ in top_features if 'pitch' in f.lower() or 'f0' in f.lower()]
    if pitch_features:
        if 'happy' in predicted_emotion.lower() or 'angry' in predicted_emotion.lower():
            interpretations.append("Elevated pitch characteristics indicate high emotional intensity.")
        elif 'sad' in predicted_emotion.lower() or 'neutral' in predicted_emotion.lower():
            interpretations.append("Lower pitch characteristics suggest subdued emotional expression.")
    
    # Check for energy/intensity features
    energy_features = [f for f, _ in top_features if 'rms' in f.lower() or 'intensity' in f.lower()]
    if energy_features:
        if 'happy' in predicted_emotion.lower() or 'angry' in predicted_emotion.lower():
            interpretations.append("High energy levels in the speech signal contribute to this prediction.")
        elif 'sad' in predicted_emotion.lower():
            interpretations.append("Reduced energy levels are characteristic of this emotion.")
    
    # Check for spectral features
    spectral_features = [f for f, _ in top_features if 'spectral' in f.lower() or 'mfcc' in f.lower()]
    if spectral_features:
        interpretations.append("Spectral characteristics reveal important timbral qualities of the emotional speech.")
    
    return " ".join(interpretations) if interpretations else ""

# ===============================
# Main route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    explanation = None
    error = None
    top_features = []  # Initialize empty list for feature importance

    if request.method == "POST":
        audio = request.files.get("audio")
        print(f"POST request received. Audio file: {audio}")

        if audio:
            # Handle both file uploads and recorded audio (which may not have filename)
            try:
                filename = audio.filename if audio.filename else f"recording_{int(time.time())}.webm"
                path = os.path.join(UPLOAD_FOLDER, filename)
                print(f"Saving audio to: {path}")
                audio.save(path)
                print(f"Audio saved successfully. File exists: {os.path.exists(path)}")

                try:
                    print(f"Processing audio file: {path}")
                    print(f"File size: {os.path.getsize(path)} bytes")
                    
                    # Convert WebM/Opus to WAV if needed
                    processed_path = convert_to_wav_if_needed(path)
                    print(f"Using processed path: {processed_path}")
                    
                    live_traits = extract_audio_traits(processed_path)
                    print(f"Extracted traits: {live_traits}")
                    
                    live_traits_scaled = scaler.transform(live_traits)

                    # Find K nearest real samples
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
                    
                    # Generate SHAP-based explanation (reveals hidden factors)
                    explanation, top_features = generate_shap_explanation(
                        nearest_samples, model, explainer, final_pred, top_n=10
                    )
                    
                    print(f"Prediction: {prediction}, Confidence: {confidence}%")
                    print(f"Top contributing features: {[f[0] for f in top_features[:5]]}")
                except Exception as e:
                    error = f"Audio processing error: {str(e)}"
                    import traceback
                    print(f"Error details: {traceback.format_exc()}")  # Debug logging
                    print(f"Error message: {str(e)}")
                    top_features = []  # Initialize on error
            except Exception as e:
                error = f"File upload error: {str(e)}"
                import traceback
                print(f"Upload error details: {traceback.format_exc()}")  # Debug logging
                top_features = []  # Initialize on error
        else:
            top_features = []  # Initialize if no audio
    else:
        top_features = []  # Initialize for GET requests

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        human_explanation=explanation,
        error=error,
        top_features=top_features
    )

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)