from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import threading
import ffmpeg
from datetime import datetime
from analyzer import SpeechAnalyzer

app = Flask(__name__)
CORS(app)

# Create necessary directories
UPLOAD_FOLDER = "uploads/"
RESULT_FOLDER = "results/"
RESULTS = {}  # Store task statuses and results
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize speech analyzer
analyzer = SpeechAnalyzer()


def extract_audio(mp4_filepath, wav_filepath):
    """Extracts audio from an MP4 file and saves it as a WAV file."""
    try:
        ffmpeg.input(mp4_filepath).output(
            wav_filepath, format="wav", acodec="pcm_s16le", ar="16000"
        ).run(overwrite_output=True, quiet=True)
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def analyze_audio_thread(filepath, task_id, result_dir):
    """Runs the audio analysis in a background thread."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_folder = os.path.join(result_dir, timestamp)
        os.makedirs(task_folder, exist_ok=True)

        RESULTS[task_id] = {"status": "processing", "folder": timestamp}

        result = analyzer.analyze_audio_file(filepath)
        result["visualization_path"] = f"/get_visualization/{task_id}"
        result["transcript_path"] = f"/get_transcript/{task_id}"

        RESULTS[task_id] = {
            "status": "completed",
            "result": result,
            "folder": timestamp,
        }
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

    # Generate a task ID
    task_id = filename.rsplit(".", 1)[0]

    # Extract audio if MP4
    if filename.endswith(".mp4"):
        wav_filepath = os.path.join(UPLOAD_FOLDER, task_id + ".wav")
        if not extract_audio(filepath, wav_filepath):
            return jsonify({"error": "Failed to extract audio"}), 500
        filepath = wav_filepath

    # Start processing in a separate thread
    thread = threading.Thread(
        target=analyze_audio_thread, args=(filepath, task_id, RESULT_FOLDER)
    )
    thread.start()

    return jsonify({"message": "Processing started!", "task_id": task_id})


@app.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id):
    """Check the status of an audio analysis task."""
    if task_id in RESULTS:
        return jsonify(RESULTS[task_id])
    return jsonify({"status": "not found"}), 404


@app.route("/get_visualization/<task_id>", methods=["GET"])
def get_visualization(task_id):
    """Serve visualization image file."""
    if task_id in RESULTS:
        task_folder = os.path.join(
            RESULT_FOLDER, RESULTS[task_id]["folder"], "visualizations"
        )
        if os.path.exists(task_folder):
            files = os.listdir(task_folder)
            if files:
                return send_from_directory(task_folder, files[0])
    return jsonify({"error": "Visualization not found"}), 404


@app.route("/get_transcript/<task_id>", methods=["GET"])
def get_transcript(task_id):
    """Serve transcript text file."""
    if task_id in RESULTS:
        task_folder = os.path.join(
            RESULT_FOLDER, RESULTS[task_id]["folder"], "transcripts"
        )
        if os.path.exists(task_folder):
            files = os.listdir(task_folder)
            if files:
                return send_from_directory(task_folder, files[0])
    return jsonify({"error": "Transcript not found"}), 404


@app.route("/get_passage_comparison/<task_id>", methods=["GET"])
def get_passage_comparison(task_id):
    """Get detailed passage comparison results."""
    if task_id in RESULTS and RESULTS[task_id]["status"] == "completed":
        result = RESULTS[task_id]["result"]
        if "passage_comparison" in result:
            return jsonify(result["passage_comparison"])
    return jsonify({"error": "Comparison data not found"}), 404


if __name__ == "__main__":
    app.run(debug=False)
