import os
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
from src.utils.audio_utils import load_audio, normalize_audio, apply_noise_reduction
from src.audio.transcription_analyzer import TranscriptionAnalyzer, TranscriptionResult
from src.visualization.speech_visualizer import SpeechVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SpeechAnalyzer:
    def __init__(self):
        """Initialize all analysis components."""
        try:
            self.transcriber = TranscriptionAnalyzer(model_size="medium")
            self.visualizer = SpeechVisualizer()
            logger.info("Speech Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Speech Analyzer: {e}")
            raise

    def analyze_audio_file(self, file_path: str) -> dict:
        """Perform full speech analysis and return the results."""
        try:
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"results/{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)

            viz_dir = output_dir / "visualizations"
            transcripts_dir = output_dir / "transcripts"

            for directory in [viz_dir, transcripts_dir]:
                directory.mkdir(exist_ok=True)

            # Load and preprocess audio
            logger.info("Loading and preprocessing audio...")
            audio_data, sample_rate = self._load_and_preprocess_audio(file_path)

            # Perform transcription and analysis
            logger.info("Performing transcription and analysis...")
            result = self.transcriber.analyze_audio(
                audio_data, sample_rate, transcripts_dir
            )

            # Compute fluency score
            fluency_score, severity = self._calculate_fluency_score(result)

            # Generate visualizations
            logger.info("Generating visualizations...")
            self._generate_visualizations(audio_data, result, viz_dir)

            # Return results as a dictionary
            return {
                "transcription": result.text,
                "stutter_events": result.repetitions + result.fillers,
                "fluency_score": fluency_score,
                "num_repetitions": len(result.repetitions),
                "num_fillers": len(result.fillers),
                "severity": severity,
                "visualization_path": str(viz_dir / "waveform_analysis.png"),
            }

        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"error": str(e)}

    def _load_and_preprocess_audio(self, file_path: str) -> tuple:
        """Load and preprocess audio file."""
        audio_data, sample_rate = load_audio(file_path, target_sr=16000)
        audio_data = normalize_audio(audio_data)
        audio_data = apply_noise_reduction(audio_data, sample_rate)
        return audio_data, sample_rate

    def _generate_visualizations(self, audio_data: np.ndarray, result, viz_dir: Path):
        """Generate visualizations for analysis."""
        fig_wave = self.visualizer.create_analysis_dashboard(
            audio_data=audio_data,
            features=result.word_timings,
            events=result.repetitions + result.fillers,
            sample_rate=16000,
        )
        self.visualizer.save_visualization(fig_wave, viz_dir / "waveform_analysis.png")

    def _calculate_fluency_score(self, result) -> tuple:
        """Calculate stutter fluency score and severity."""
        total_syllables = max(1, len(result.text.split()))  # Avoid division by zero
        stutter_events = len(result.repetitions) + len(result.fillers)

        # Compute %SS (Percentage of Syllables Stuttered)
        percent_ss = (stutter_events / total_syllables) * 100

        # Extract durations safely
        try:
            longest_stutter = max(
                [
                    event.get("duration", 0)
                    for event in result.repetitions + result.fillers
                ],
                default=0,
            )
        except AttributeError:
            longest_stutter = 0  # Default if duration is missing

        duration_score = self._get_duration_score(longest_stutter)

        # Compute final fluency score
        fluency_score = int(percent_ss) + duration_score

        # Determine severity level
        severity = self._get_severity_level(fluency_score)

        return fluency_score, severity

    def _get_duration_score(self, duration) -> int:
        """Assigns duration score based on the longest stuttering event."""
        if duration < 1.0:
            return 2
        elif duration < 2.0:
            return 4
        elif duration < 3.0:
            return 6
        elif duration < 5.0:
            return 8
        else:
            return 10

    def _get_severity_level(self, score) -> str:
        """Determines severity level based on SSI-4 scoring tables."""
        if score <= 10:
            return "Very Mild"
        elif score <= 20:
            return "Mild"
        elif score <= 30:
            return "Moderate"
        elif score <= 40:
            return "Severe"
        else:
            return "Very Severe"
