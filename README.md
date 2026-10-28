🎤 Speech Emotion Recognition System (SER)
📌 Project Overview

The Speech Emotion Recognition (SER) System is a machine learning–based application that analyzes human speech audio and predicts the emotion expressed in the voice.
The system takes a .wav audio file as input, extracts acoustic features, and classifies the emotion using a trained machine learning model.

This project demonstrates the practical application of audio signal processing, machine learning, and web-based user interaction.

🎯 Objectives

To analyze speech signals and extract meaningful acoustic features

To train a machine learning model for emotion classification

To predict emotions from unseen speech audio files

To provide a simple and intuitive web-based interface for users

🧠 Emotions Recognized

The system can classify speech into emotion categories such as:

Angry

Happy

Sad

Neutral

Fear

Disgust

Surprise

(Emotion labels depend on the dataset used)

🗂️ Dataset Used

TESS – Toronto Emotional Speech Set

Contains emotional speech samples spoken by female speakers

Preprocessed into a CSV file with handcrafted acoustic features

File used:

all_handcrafted_data_tess.csv

⚙️ Technology Stack
🧪 Machine Learning & Audio Processing

Python 3.10

Librosa (audio feature extraction)

NumPy, Pandas

Scikit-learn (Random Forest Classifier)

Joblib (model saving/loading)

🌐 Web Application

Flask (backend framework)

HTML & CSS (frontend UI)

🏗️ System Architecture
Audio File (.wav)
        ↓
Feature Extraction (Librosa)
        ↓
Trained ML Model (Random Forest)
        ↓
Emotion Prediction
        ↓
Web Interface Output

📁 Project Structure
ser_major-project/
│
├── app.py                      # Flask web application
├── train_model.py              # Model training script
├── predict_audio.py             # Standalone audio prediction script
├── model.joblib                 # Trained ML model
├── all_handcrafted_data_tess.csv
│
├── templates/
│   └── index.html               # Frontend HTML
│
├── static/
│   └── style.css                # Frontend styling
│
├── uploads/                     # Uploaded audio files
└── venv/                        # Virtual environment

🚀 How to Run the Project
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install Dependencies
pip install flask librosa numpy pandas scikit-learn joblib soundfile

3️⃣ Train the Model
python train_model.py


This will generate:

model.joblib

4️⃣ Run the Web Application
python app.py


Open browser:

http://127.0.0.1:5000

🎤 How to Use the Application

Open the web interface

Upload any .wav speech audio file

Click Analyze Emotion

View the predicted emotion and confidence score

📊 Model Performance

Algorithm: Random Forest Classifier

Training Accuracy: 0.9875

Evaluation Method: Train–Test Split

🎓 Academic Justification

“The system extracts acoustic features from speech audio and uses a supervised machine learning classifier to predict emotions. Metadata columns were removed during preprocessing, and only numeric acoustic features were used for training.”

🔮 Future Enhancements

Real-time microphone recording

Support for more datasets

Deep learning models (CNN / LSTM)

Emotion visualization graphs

Cloud deployment

✅ Conclusion

The Speech Emotion Recognition System successfully demonstrates how machine learning and audio processing techniques can be applied to identify emotions from human speech.
The project achieves accurate emotion classification and provides an easy-to-use web interface, making it suitable for academic demonstrations and real-world extensions.

👤 Author

S. Yaseen Ahmed
Major Project – Speech Emotion Recognition
Python | Machine Learning | Audio Processing
