"""
app.py

Flask application for audio analysis with complete reporting functionality.
"""

from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import os
from pathlib import Path
import soundfile as sf
import librosa
import threading
from src.audio.transcription_analyzer import TranscriptionAnalyzer
from src.utils.audio_utils import load_audio, normalize_audio, apply_noise_reduction

app = Flask(__name__)
UPLOAD_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add thread lock for safe concurrent processing
processing_lock = threading.Lock()

@app.route("/process-audio", methods=["POST"])
def process_audio_pipeline():
    """Complete pipeline endpoint for audio processing."""
    with processing_lock:  # Use lock to prevent concurrent processing issues
        try:
            # Validate Input
            if "audio" not in request.files:
                return jsonify({"error": "No audio file uploaded"}), 400
                
            file = request.files["audio"]
            if not file.filename:
                return jsonify({"error": "No file selected"}), 400

            # Setup Directories
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(UPLOAD_FOLDER) / f"analysis_{timestamp}"
            reports_dir = output_dir / "reports"
            transcripts_dir = output_dir / "transcripts"
            
            for directory in [output_dir, reports_dir, transcripts_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            # Process Audio
            audio_path = output_dir / file.filename
            file.save(str(audio_path))

            try:
                # Load and preprocess
                audio_data, sample_rate = load_audio(str(audio_path), target_sr=16000)
                audio_data = normalize_audio(audio_data)
                audio_data = apply_noise_reduction(audio_data, sample_rate)

                # Initialize analyzer in try block
                transcriber = TranscriptionAnalyzer(model_size="medium")
                result = transcriber.analyze_audio(audio_data, sample_rate, transcripts_dir)

                # Generate and Save Reports
                reports = {
                    "transcription.txt": {
                        "content": result.text,
                        "description": "Raw transcription"
                    },
                    "analysis_summary.txt": {
                        "content": generate_summary_report(result),
                        "description": "Overview of analysis results"
                    },
                    "detailed_analysis.txt": {
                        "content": generate_detailed_report(result),
                        "description": "Detailed speech analysis"
                    },
                    "stutter_analysis.txt": {
                        "content": generate_stutter_report(result),
                        "description": "Detailed stutter analysis"
                    }
                }

                saved_reports = {}
                for filename, report_data in reports.items():
                    file_path = reports_dir / filename
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(report_data["content"])
                    
                    saved_reports[filename] = {
                        "path": str(file_path),
                        "description": report_data["description"],
                        "download_url": f"/download/{output_dir.name}/reports/{filename}"
                    }

                # Prepare Analysis Summary
                analysis_summary = {
                    "duration": result.duration,
                    "word_count": len(result.word_timings),
                    "speech_rate": result.speech_rate,
                    "filler_count": len(result.fillers),
                    "repetition_count": len(result.repetitions),
                    "grammar_error_count": len(result.grammar_errors),
                    "language_score": result.language_score,
                    "confidence": result.confidence
                }

                return jsonify({
                    "status": "success",
                    "timestamp": timestamp,
                    "summary": analysis_summary,
                    "reports": saved_reports,
                    "output_directory": str(output_dir)
                }), 200

            finally:
                # Cleanup
                if 'transcriber' in locals():
                    del transcriber
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass

        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }), 500

# [Previous helper functions remain the same]
def generate_summary_report(result) -> str:
    """Generate a concise summary report."""
    summary = [
        "SPEECH ANALYSIS SUMMARY",
        "=" * 30,
        "",
        "Key Metrics:",
        f"- Duration: {result.duration:.2f} seconds",
        f"- Speech Rate: {result.speech_rate:.1f} words/minute",
        f"- Language Score: {result.language_score:.2f}/1.0",
        f"- Confidence: {result.confidence:.2f}/1.0",
        "",
        "Event Statistics:",
        f"- Total Words: {len(result.word_timings)}",
        f"- Filler Words: {len(result.fillers)}",
        f"- Repetitions: {len(result.repetitions)}",
        f"- Grammar Errors: {len(result.grammar_errors)}"
    ]
    return "\n".join(summary)

def generate_detailed_report(result) -> str:
    """Generate a detailed analysis report."""
    sections = [
        "DETAILED SPEECH ANALYSIS",
        "=" * 50,
        "",
        "1. TRANSCRIPTION",
        "-" * 20,
        result.text,
        "",
        "2. TIMING ANALYSIS",
        "-" * 20,
        ""
    ]
    
    for i, segment in enumerate(result.segments, 1):
        sections.extend([
            f"Segment {i}:",
            f"  Time: {segment['start']:.2f}s - {segment['end']:.2f}s",
            f"  Text: {segment['text']}",
            ""
        ])

    return "\n".join(sections)

def generate_stutter_report(result) -> str:
    """Generate a detailed stutter analysis report."""
    sections = [
        "STUTTER ANALYSIS REPORT",
        "=" * 50,
        "",
        "1. REPETITIONS",
        "-" * 20,
        ""
    ]

    if result.repetitions:
        for rep in result.repetitions:
            sections.extend([
                f"Pattern: {rep.get('word', '')}",
                f"Times Repeated: {rep.get('count', 0)}",
                f"Time: {rep.get('start', 0):.2f}s - {rep.get('end', 0):.2f}s",
                f"Type: {rep.get('event_type', 'unknown')}",
                f"Confidence: {rep.get('confidence', 0):.2f}",
                ""
            ])
    else:
        sections.append("No repetitions detected\n")

    sections.extend([
        "2. FILLER WORDS",
        "-" * 20,
        ""
    ])

    if result.fillers:
        for filler in result.fillers:
            sections.extend([
                f"Word: {filler.get('word', '')}",
                f"Time: {filler.get('start', 0):.2f}s",
                f"Type: {filler.get('event_type', 'unknown')}",
                f"Context: {filler.get('context', '')}",
                ""
            ])
    else:
        sections.append("No filler words detected\n")

    return "\n".join(sections)

if __name__ == "__main__":
    app.run(debug=False)  # Set debug to False to avoid reloader issues