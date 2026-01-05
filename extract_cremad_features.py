import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "CREMA-D"
OUTPUT_FILE = "cremad_handcrafted.csv"

records = []

print("🔍 Scanning CREMA-D dataset...")

for file in os.listdir(DATASET_PATH):
    if not file.lower().endswith(".wav"):
        continue

    filepath = os.path.join(DATASET_PATH, file)

    try:
        y, sr = librosa.load(filepath, duration=3)

        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        emotion_code = file.split("_")[2]

        label_map = {
            "ANG": "angry",
            "HAP": "happy",
            "SAD": "sad",
            "NEU": "neutral",
            "FEA": "fear",
            "DIS": "disgust"
        }

        emotion = label_map.get(emotion_code)
        if emotion is None:
            continue

        records.append({
            "rms": rms,
            "zcr": zcr,
            "spectral_centroid": spec_centroid,
            "class": emotion
        })

    except Exception as e:
        print("❌ Error processing:", file, e)

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("❌ No features extracted. Dataset path or filenames are wrong.")

df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Feature extraction complete")
print(f"📊 Shape: {df.shape}")
print(f"📁 Saved as {OUTPUT_FILE}")
