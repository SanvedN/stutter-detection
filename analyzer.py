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

            # Generate visualizations
            logger.info("Generating visualizations...")
            self._generate_visualizations(audio_data, result, viz_dir)

            # Return results as a dictionary
            return {
                "transcription": result.text,
                "stutter_events": result.repetitions + result.fillers,
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
