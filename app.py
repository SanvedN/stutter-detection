from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from analyzer import SpeechAnalyzer

app = Flask(__name__)
CORS(app)

# Create necessary directories
UPLOAD_FOLDER = "uploads/"
RESULT_FOLDER = "results/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize speech analyzer
analyzer = SpeechAnalyzer()

@app.route("/")
def home():
    return jsonify({"message": "Stutter Detection API is Running!"})


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    """Handles audio file upload and processing"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Process the audio
    analysis_result = analyzer.analyze_audio_file(filepath)

    return jsonify(
        {"message": "Audio processed successfully!", "result": analysis_result}
    )


@app.route("/get_results", methods=["GET"])
def get_results():
    """Fetch all processed analysis results"""
    results = []
    for folder in os.listdir(RESULT_FOLDER):
        result_path = os.path.join(RESULT_FOLDER, folder, "stutter_analysis.txt")
        if os.path.exists(result_path):
            results.append({"report_path": result_path})
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
