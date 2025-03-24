from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import threading
from analyzer import SpeechAnalyzer

app = Flask(__name__)
CORS(app)

# Create necessary directories
UPLOAD_FOLDER = "uploads/"
RESULT_FOLDER = "results/"
RESULTS = {}  # Store task statuses and results
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize speech analyzer
analyzer = SpeechAnalyzer()

def analyze_audio_thread(filepath, task_id):
    """Runs the audio analysis in a background thread."""
    try:
        RESULTS[task_id] = {"status": "processing"}
        result = analyzer.analyze_audio_file(filepath)
        RESULTS[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        RESULTS[task_id] = {"status": "failed", "error": str(e)}


@app.route("/")
def home():
    return jsonify({"message": "Stutter Detection API is Running!"})


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    """Handles file upload and starts processing in a separate thread."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Generate a task ID (use filename or UUID)
    task_id = filename.split(".")[0]  # You can replace with str(uuid.uuid4())

    # Start processing in a separate thread
    thread = threading.Thread(target=analyze_audio_thread, args=(filepath, task_id))
    thread.start()

    return jsonify({"message": "Processing started!", "task_id": task_id})


@app.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id):
    """Check the status of an audio analysis task."""
    if task_id in RESULTS:
        return jsonify(RESULTS[task_id])
    return jsonify({"status": "not found"}), 404


if __name__ == "__main__":
    app.run(debug=False)
