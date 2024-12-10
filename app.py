from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import os
from src.audio.transcription_analyzer import TranscriptionAnalyzer
from src.audio.feature_extractor import FeatureExtractor
from src.audio.stutter_detector import StutterDetector
from src.visualization.speech_visualizer import SpeechVisualizer

app = Flask(__name__)
UPLOAD_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    try:
        # Step 1: Handle Uploaded File
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400
        file = request.files["audio"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(UPLOAD_FOLDER, f"analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, file.filename)
        file.save(audio_path)

        # Step 2: Initialize Components
        transcriber = TranscriptionAnalyzer()
        extractor = FeatureExtractor()
        stutter_detector = StutterDetector()
        visualizer = SpeechVisualizer()

        # Step 3: Process Audio
        # Transcription
        transcription = transcriber.transcribe(audio_path)
        transcription_path = os.path.join(output_dir, "transcription.txt")
        with open(transcription_path, "w") as f:
            f.write(transcription)

        # Feature Extraction
        features = extractor.extract_features(audio_path)
        features_path = os.path.join(output_dir, "features.json")
        with open(features_path, "w") as f:
            f.write(features)

        # Stutter Analysis
        stutter_summary = stutter_detector.detect_stutters(audio_path)
        stutter_path = os.path.join(output_dir, "stutter_summary.json")
        with open(stutter_path, "w") as f:
            f.write(stutter_summary)

        # Visualizations
        waveform_path = os.path.join(output_dir, "waveform.png")
        visualizer.generate_waveform(audio_path, waveform_path)

        stutter_analysis_path = os.path.join(output_dir, "stutter_analysis.png")
        visualizer.generate_stutter_analysis_plot(audio_path, stutter_analysis_path)

        # Step 4: Return Result
        return jsonify({"message": "Analysis complete", "output_directory": output_dir}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    """Endpoint to download generated files."""
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
