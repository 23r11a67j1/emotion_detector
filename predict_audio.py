import librosa
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.joblib")

# Feature extraction function
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=3, offset=0.5)

    features = {}

    # Basic features
    features["rms"] = np.mean(librosa.feature.rms(y=y))
    features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))

    # Spectral features
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # MFCCs (mean only to match handcrafted style)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}"] = np.mean(mfcc[i])

    return features


# 🔊 CHANGE THIS to your audio file name
audio_file = "sample_audio.wav"

# Extract features
audio_features = extract_features(audio_file)

# Convert to DataFrame
input_df = pd.DataFrame([audio_features])

# 🔧 Align input features with training features
model_features = model.feature_names_in_
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Predict emotion
prediction = model.predict(input_df)

print("Predicted Emotion:", prediction[0])
