from flask import Flask, render_template, request
import time

from SER.inference.audio_input_handler import handle_audio_input
from SER.inference.emotion_predictor import predict_from_audio

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        audio_path = handle_audio_input(request)

        # buffer / processing delay for UX
        time.sleep(2)

        result = predict_from_audio(audio_path)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
