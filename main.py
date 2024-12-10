"""
main.py

Main script for complete speech analysis system.
Integrates audio processing, transcription, analysis, and visualization.
"""

import os
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from typing import Dict
import nltk
from src.utils.audio_utils import load_audio, normalize_audio, apply_noise_reduction
from src.audio.transcription_analyzer import TranscriptionAnalyzer
from src.visualization.speech_visualizer import SpeechVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

    def analyze_audio_file(self, file_path: str) -> None:
        """
        Perform comprehensive analysis of audio file.
        
        Args:
            file_path: Path to input audio file
        """
        try:
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"analysis_results_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            viz_dir = output_dir / "visualizations"
            reports_dir = output_dir / "reports"
            transcripts_dir = output_dir / "transcripts"
            
            for directory in [viz_dir, reports_dir, transcripts_dir]:
                directory.mkdir(exist_ok=True)

            # Load and preprocess audio
            logger.info("Loading and preprocessing audio...")
            audio_data, sample_rate = self._load_and_preprocess_audio(file_path)

            # Perform transcription and analysis
            logger.info("Performing transcription and analysis...")
            result = self.transcriber.analyze_audio(audio_data, sample_rate, transcripts_dir)

            # Generate visualizations
            logger.info("Generating visualizations...")
            self._generate_visualizations(audio_data, result, viz_dir)

            # Generate detailed reports
            logger.info("Generating analysis reports...")
            self._generate_reports(result, reports_dir)

            # Create main summary report
            self._create_summary_report(result, audio_data, output_dir)

            logger.info(f"Analysis complete! Results saved to {output_dir}")
            self._print_analysis_summary(result)

        except Exception as e:
            logger.error(f"Error analyzing audio file: {e}")
            raise

    def _load_and_preprocess_audio(self, file_path: str) -> tuple:
        """Load and preprocess audio file."""
        audio_data, sample_rate = load_audio(file_path, target_sr=16000)
        audio_data = normalize_audio(audio_data)
        audio_data = apply_noise_reduction(audio_data, sample_rate)
        return audio_data, sample_rate

    def _generate_visualizations(self, audio_data: np.ndarray, result, viz_dir: Path):
        """Generate all visualizations."""
        # Waveform with stutter markers
        fig_wave = self.visualizer.create_analysis_dashboard(
            audio_data=audio_data,
            features=result.word_timings,
            events=result.repetitions + result.fillers,
            sample_rate=16000
        )
        self.visualizer.save_visualization(fig_wave, viz_dir / "waveform_analysis.png")

        # Stutter patterns visualization
        fig_stutter = self.visualizer.create_stutter_report(
            result.repetitions + result.fillers
        )
        self.visualizer.save_visualization(fig_stutter, viz_dir / "stutter_patterns.png")

        # Speech rate and fluency visualization
        fig_fluency = self.visualizer.create_fluency_analysis(result)
        self.visualizer.save_visualization(fig_fluency, viz_dir / "fluency_analysis.png")

    def _generate_reports(self, result, reports_dir: Path):
        """Generate detailed analysis reports."""
        # Stutter analysis report
        self._generate_stutter_report(result, reports_dir / "stutter_analysis.txt")
        
        # Fluency report
        self._generate_fluency_report(result, reports_dir / "fluency_analysis.txt")
        
        # Grammar report
        self._generate_grammar_report(result, reports_dir / "grammar_analysis.txt")
        
        # Detailed timing analysis
        self._generate_timing_report(result, reports_dir / "timing_analysis.txt")

    def _generate_stutter_report(self, result, output_path: Path):
        """Generate detailed stutter analysis report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("STUTTER ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Repetitions analysis
            f.write("1. REPETITION PATTERNS\n")
            f.write("-" * 20 + "\n")
            for rep in result.repetitions:
                f.write(f"- Pattern: {rep['word']} (repeated {rep['count']} times)\n")
                f.write(f"  Time: {rep['start']:.2f}s - {rep['end']:.2f}s\n")
                f.write(f"  Type: {rep['type']}\n")
                f.write(f"  Confidence: {rep['confidence']:.2f}\n\n")

            # Filler analysis
            f.write("\n2. FILLER WORDS\n")
            f.write("-" * 20 + "\n")
            filler_types = {}
            for filler in result.fillers:
                filler_type = filler['type']
                if filler_type not in filler_types:
                    filler_types[filler_type] = []
                filler_types[filler_type].append(filler)

            for ftype, fillers in filler_types.items():
                f.write(f"\n{ftype.upper()}:\n")
                for filler in fillers:
                    f.write(f"- Word: {filler['word']}\n")
                    f.write(f"  Time: {filler['start']:.2f}s\n")
                    f.write(f"  Context: {filler['context']}\n")

    def _generate_fluency_report(self, result, output_path: Path):
        """Generate detailed fluency analysis report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("FLUENCY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall metrics
            f.write("1. OVERALL METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Speech Rate: {result.speech_rate:.1f} words per minute\n")
            f.write(f"Language Score: {result.language_score:.2f}/1.0\n")
            
            # Detailed breakdown
            total_words = len(result.word_timings)
            filler_count = len(result.fillers)
            repetition_count = len(result.repetitions)
            
            f.write(f"\nTotal Words: {total_words}\n")
            f.write(f"Filler Word Rate: {(filler_count/total_words*100):.1f}%\n")
            f.write(f"Repetition Rate: {(repetition_count/total_words*100):.1f}%\n")

    def _generate_grammar_report(self, result, output_path: Path):
        """Generate detailed grammar analysis report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("GRAMMAR ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            if result.grammar_errors:
                for error in result.grammar_errors:
                    f.write(f"Error Type: {error['type']}\n")
                    f.write(f"Text: {error['text']}\n")
                    f.write(f"Position: {error['start_pos']} - {error['end_pos']}\n")
                    f.write("-" * 30 + "\n")
            else:
                f.write("No significant grammar errors detected.\n")

    def _generate_timing_report(self, result, output_path: Path):
        """Generate detailed timing analysis report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("TIMING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Analyze pauses and timing patterns
            f.write("1. SPEECH SEGMENTS\n")
            f.write("-" * 20 + "\n")
            for i, segment in enumerate(result.segments):
                duration = segment['end'] - segment['start']
                f.write(f"\nSegment {i+1}:\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Text: {segment['text']}\n")

    def _create_summary_report(self, result, audio_data: np.ndarray, output_dir: Path):
        """Create main summary report."""
        with open(output_dir / "analysis_summary.txt", 'w', encoding='utf-8') as f:
            f.write("SPEECH ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Audio information
            f.write("1. AUDIO INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Duration: {result.duration:.2f} seconds\n")
            f.write(f"Sample Rate: 16000 Hz\n\n")

            # Key metrics
            f.write("2. KEY METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Speech Rate: {result.speech_rate:.1f} words/minute\n")
            f.write(f"Language Score: {result.language_score:.2f}/1.0\n")
            f.write(f"Overall Confidence: {result.confidence:.2f}\n\n")

            # Statistics
            f.write("3. STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Words: {len(result.word_timings)}\n")
            f.write(f"Filler Words: {len(result.fillers)}\n")
            f.write(f"Repetitions: {len(result.repetitions)}\n")
            f.write(f"Grammar Errors: {len(result.grammar_errors)}\n")

    def _print_analysis_summary(self, result):
        """Print analysis summary to console."""
        print("\nAnalysis Summary:")
        print("-" * 20)
        print(f"Duration: {result.duration:.2f} seconds")
        print(f"Speech Rate: {result.speech_rate:.1f} words/minute")
        print(f"Filler Words: {len(result.fillers)}")
        print(f"Repetitions: {len(result.repetitions)}")
        print(f"Grammar Errors: {len(result.grammar_errors)}")

def main():
    """Main entry point."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Analyze speech audio file")
    parser.add_argument("file_path", help="Path to audio file")
    parser.add_argument("--model", default="medium", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size to use")
    
    args = parser.parse_args()
    
    try:
        # Check for required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger_eng')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')
            nltk.download('punkt_tab')
        
        analyzer = SpeechAnalyzer()
        analyzer.analyze_audio_file(args.file_path)
        
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        print("Please make sure all required packages are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()