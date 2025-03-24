"""
speech_visualizer.py

Optimized visualization module for the stutter detection system.
Generates various visualizations for speech analysis and stutter detection.

Features:
   - Waveform visualization with detected events
   - Spectrogram display
   - Pitch and energy contour plotting
   - Zero-crossing rate analysis
   - Event distribution statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
import logging
from typing import List, Dict, Optional
from matplotlib.figure import Figure

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a non-interactive Matplotlib backend for compatibility
import matplotlib

matplotlib.use("Agg")


class SpeechVisualizer:
    """
    Handles visualizations for speech analysis, including stutter events.
    """

    def __init__(self):
        """Initialize visualizer with style settings and color themes."""
        plt.style.use("ggplot")  # Improved readability
        sns.set_palette("pastel")  # Soft color scheme

        # Define colors for different event types
        self.event_colors = {
            "filler": "#4ECDC4",
            "repetition": "#FF6B6B",
            "prolongation": "#45B7D1",
            "block": "#96CEB4",
            "unknown": "#95A5A6",
        }
        logger.info("SpeechVisualizer initialized.")

    def create_analysis_dashboard(
        self,
        audio_data: np.ndarray,
        features: List[Dict],
        events: List[Dict],
        sample_rate: int,
    ) -> Figure:
        """
        Generate a complete speech analysis dashboard.

        Args:
            audio_data: Raw audio waveform data.
            features: List of word features (timing, confidence).
            events: List of detected stuttering events.
            sample_rate: Audio sample rate.

        Returns:
            Matplotlib figure with multiple subplots.
        """
        try:
            fig, axes = plt.subplots(4, 1, figsize=(15, 14), constrained_layout=True)

            # 1. Waveform with stutter markers
            self._plot_waveform(axes[0], audio_data, sample_rate, events)

            # 2. Spectrogram
            self._plot_spectrogram(axes[1], audio_data, sample_rate)

            # 3. Word timing and confidence plot
            self._plot_word_timing(axes[2], features)

            # 4. Event distribution graph
            self._plot_event_distribution(axes[3], events)

            return fig

        except Exception as e:
            logger.error(f"Error creating analysis dashboard: {e}")
            raise

    def _plot_waveform(
        self, ax: plt.Axes, audio_data: np.ndarray, sample_rate: int, events: List[Dict]
    ) -> None:
        """Plot waveform with highlighted stutter events."""
        times = np.arange(len(audio_data)) / sample_rate
        ax.plot(times, audio_data, color="#2E4057", alpha=0.7)
        ax.set_title("Speech Waveform with Detected Events")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")

        # Mark detected stuttering events
        for event in events:
            start = event.get("start", 0)
            end = event.get("end", start + 0.5)
            event_type = event.get("event_type", "unknown")
            color = self.event_colors.get(event_type, "#95A5A6")
            ax.axvspan(start, end, color=color, alpha=0.3)

            # Add text label for event type
            ax.text(
                start,
                ax.get_ylim()[1] * 0.8,
                event_type,
                fontsize=8,
                rotation=45,
                verticalalignment="bottom",
            )

    def _plot_spectrogram(
        self, ax: plt.Axes, audio_data: np.ndarray, sample_rate: int
    ) -> None:
        """Generate a spectrogram visualization."""
        S = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(
            S, sr=sample_rate, x_axis="time", y_axis="log", ax=ax, cmap="magma"
        )
        ax.set_title("Spectrogram")
        fig = ax.get_figure()
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    def _plot_word_timing(self, ax: plt.Axes, features: List[Dict]) -> None:
        """Plot word timing and confidence levels."""
        if not features:
            ax.text(0.5, 0.5, "No word features available", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return

        for feature in features:
            start = feature.get("start", 0)
            end = feature.get("end", start + 0.5)
            confidence = feature.get("confidence", 0.5)
            word = feature.get("word", "")

            # Map confidence to a color scale
            color = plt.cm.RdYlGn(confidence)

            ax.barh(
                y=0, width=end - start, left=start, height=0.8, alpha=0.6, color=color
            )
            ax.text(start, 0, word, rotation=45, fontsize=8, verticalalignment="bottom")

        ax.set_title("Word Timing and Confidence")
        ax.set_xlabel("Time (seconds)")
        ax.set_yticks([])

    def _plot_event_distribution(self, ax: plt.Axes, events: List[Dict]) -> None:
        """Plot distribution of different stutter events."""
        if not events:
            ax.text(0.5, 0.5, "No events detected", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Count occurrences of each event type
        event_counts = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        labels = list(event_counts.keys())
        counts = list(event_counts.values())
        colors = [self.event_colors.get(t, "#95A5A6") for t in labels]

        ax.bar(labels, counts, color=colors)
        ax.set_title("Stutter Event Distribution")
        ax.set_xlabel("Event Type")
        ax.set_ylabel("Count")

        # Add value labels on bars
        for i, count in enumerate(counts):
            ax.text(i, count, str(count), ha="center", va="bottom")

    def save_visualization(self, fig: Figure, filepath: str) -> None:
        """
        Save a visualization figure.

        Args:
            fig: Matplotlib figure.
            filepath: Path to save the figure.
        """
        try:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
