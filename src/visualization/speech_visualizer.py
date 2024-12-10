"""
speech_visualizer.py

Visualization module for stutter detection system.
Creates comprehensive visualizations of speech analysis and stutter detection results.

Key Features:
   - Real-time waveform display
   - Spectrogram visualization
   - Feature analysis plots
   - Stutter event visualization
   - Statistical analysis graphs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import logging
import librosa
import librosa.display
import seaborn as sns
from matplotlib.figure import Figure
from datetime import datetime

from src.audio.feature_extractor import SpeechFeatures
from src.audio.stutter_detector import StutterEvent, StutterType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechVisualizer:
   """
   Creates and manages visualizations for speech analysis results.
   
   This class handles the creation of various plots and visualizations
   for analyzing speech patterns and stutter events.
   """
   
   def __init__(self):
    """Initialize visualizer with default style settings."""
    # Use a built-in matplotlib style instead of seaborn
    plt.style.use('default')  # or 'classic', 'ggplot', etc.
    
    # Set custom colors
    self.colors = {
        StutterType.REPETITION: '#FF6B6B',
        StutterType.PROLONGATION: '#4ECDC4',
        StutterType.BLOCK: '#45B7D1',
        StutterType.INTERJECTION: '#96CEB4'
    }
    
    # Set default figure parameters
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100
    
    logger.info("SpeechVisualizer initialized")

   def create_analysis_dashboard(self, audio_data: np.ndarray, features: List[Dict], events: List[Dict], sample_rate: int) -> Figure:
    """
    Create comprehensive analysis dashboard with all visualizations.
    
    Args:
        audio_data: Raw audio signal
        features: List of word features with timing information
        events: List of detected events (fillers, repetitions)
        sample_rate: Audio sample rate
        
    Returns:
        matplotlib.figure.Figure: Complete dashboard figure
    """
    try:
        # Create main figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1.5, 1.5, 1])

        # 1. Waveform with events (top panel)
        ax_wave = fig.add_subplot(gs[0])
        times = np.arange(len(audio_data)) / sample_rate
        ax_wave.plot(times, audio_data, color='#2E4057', alpha=0.7)
        
        # Mark all events on waveform
        for event in events:
            if 'event_type' not in event:
                continue
                
            start = event.get('start', 0)
            end = event.get('end', start + 0.5)
            event_type = event['event_type']
            
            # Get color and create highlight
            color = self._get_event_color(event_type)
            ax_wave.axvspan(start, end, color=color, alpha=0.3)
            
            # Create detailed label
            if event_type == 'filler':
                label = f"Filler: {event.get('word', '')}"
                if 'filler_type' in event:
                    label += f"\n({event['filler_type']})"
            elif event_type == 'repetition':
                label = f"Repetition: {event.get('word', '')}x{event.get('count', 0)}"
                if 'repetition_type' in event:
                    label += f"\n({event['repetition_type']})"
            else:
                label = event_type
                
            # Add label with confidence if available
            if 'confidence' in event:
                label += f"\n{event['confidence']:.2f}"
                
            ax_wave.text(start, ax_wave.get_ylim()[1], label,
                        rotation=45, fontsize=8, verticalalignment='bottom')

        ax_wave.set_title('Speech Waveform with Detected Events')
        ax_wave.set_xlabel('Time (seconds)')
        ax_wave.set_ylabel('Amplitude')

        # 2. Spectrogram (middle panel)
        ax_spec = fig.add_subplot(gs[1])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)),
                                  ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time',
                               sr=sample_rate, ax=ax_spec)
        ax_spec.set_title('Spectrogram')

        # 3. Word timeline (third panel)
        ax_words = fig.add_subplot(gs[2])
        
        # Plot word segments
        for feature in features:
            start = feature.get('start', 0)
            end = feature.get('end', start + 0.5)
            confidence = feature.get('confidence', 0.5)
            
            # Color based on confidence
            color = plt.cm.RdYlGn(confidence)
            
            ax_words.barh(y=0, width=end-start, left=start,
                         height=0.8, alpha=0.6, color=color)
            
            # Add word labels
            if 'word' in feature:
                ax_words.text(start, 0, feature['word'],
                            rotation=45, fontsize=8,
                            verticalalignment='bottom')

        ax_words.set_title('Word Timing and Confidence')
        ax_words.set_xlabel('Time (seconds)')
        ax_words.set_yticks([])

        # 4. Event distribution (bottom panel)
        ax_dist = fig.add_subplot(gs[3])
        
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event.get('event_type', 'unknown')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        # Create bar chart
        if event_counts:
            types = list(event_counts.keys())
            counts = list(event_counts.values())
            colors = [self._get_event_color(t) for t in types]
            
            ax_dist.bar(types, counts, color=colors)
            ax_dist.set_title('Event Distribution')
            ax_dist.tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                ax_dist.text(i, count, str(count),
                           ha='center', va='bottom')
        else:
            ax_dist.text(0.5, 0.5, 'No events detected',
                        ha='center', va='center')
            ax_dist.set_xticks([])
            ax_dist.set_yticks([])

        # Adjust layout and spacing
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise

   def _get_event_color(self, event_type: str) -> str:
        """Get color for event type."""
        colors = {
            'filler': '#4ECDC4',
            'repetition': '#FF6B6B',
            'prolongation': '#45B7D1',
            'block': '#96CEB4',
            'unknown': '#95A5A6'
        }
        return colors.get(event_type.lower(), colors['unknown'])

   def _plot_waveform(self, ax: plt.Axes, audio_data: np.ndarray, 
                     sample_rate: int, events: List[StutterEvent]) -> None:
       """Plot waveform with stutter events marked."""
       times = np.arange(len(audio_data)) / sample_rate
       ax.plot(times, audio_data, color='#2E4057', alpha=0.7)
       
       # Mark stutter events
       for event in events:
           ax.axvspan(event.start_time, event.end_time, 
                     color=self.colors[event.stutter_type],
                     alpha=0.3)
       
       ax.set_title('Waveform with Stutter Events')
       ax.set_xlabel('Time (s)')
       ax.set_ylabel('Amplitude')

   def _plot_spectrogram(self, ax: plt.Axes, audio_data: np.ndarray, 
                        sample_rate: int) -> None:
       """Plot spectrogram of audio signal."""
       D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
       librosa.display.specshow(D, y_axis='log', sr=sample_rate, ax=ax)
       ax.set_title('Spectrogram')
       ax.set_xlabel('Time')
       ax.set_ylabel('Frequency (Hz)')

   def _plot_pitch_contour(self, ax: plt.Axes, pitch: np.ndarray) -> None:
       """Plot pitch contour."""
       ax.plot(pitch, color='#2E4057')
       ax.set_title('Pitch Contour')
       ax.set_xlabel('Frames')
       ax.set_ylabel('Frequency (Hz)')

   def _plot_energy_contour(self, ax: plt.Axes, energy: np.ndarray) -> None:
       """Plot energy contour."""
       ax.plot(energy, color='#2E4057')
       ax.set_title('Energy Contour')
       ax.set_xlabel('Frames')
       ax.set_ylabel('Energy')

   def _plot_stutter_statistics(self, ax: plt.Axes, events: List[StutterEvent]) -> None:
    """Plot stutter statistics."""
    try:
        # Count events by type
        event_types = [event.stutter_type for event in events]
        type_counts = {stype: event_types.count(stype) for stype in StutterType}
        
        # Convert enum types to strings for plotting
        labels = [str(stype.value) for stype in type_counts.keys()]
        values = list(type_counts.values())
        colors = [self.colors[stype] for stype in type_counts.keys()]
        
        # Create bar plot with string labels
        ax.bar(labels, values, color=colors)
        
        ax.set_title('Stutter Event Distribution')
        ax.set_xlabel('Stutter Type')
        ax.set_ylabel('Count')
        
        # Rotate labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    except Exception as e:
        logger.error(f"Error plotting statistics: {e}")
        raise

   def create_feature_plot(self, features: SpeechFeatures) -> Figure:
       """
       Create detailed feature analysis plot.
       
       Args:
           features (SpeechFeatures): Extracted speech features
           
       Returns:
           Figure: Matplotlib figure with feature plots
       """
       fig, axes = plt.subplots(3, 1, figsize=(12, 8))
       
       # Plot MFCC features
       librosa.display.specshow(features.mfcc, ax=axes[0])
       axes[0].set_title('MFCC Features')
       
       # Plot pitch and energy together
       ax2 = axes[1].twinx()
       axes[1].plot(features.pitch, color='blue', label='Pitch')
       ax2.plot(features.energy, color='red', label='Energy')
       axes[1].set_title('Pitch and Energy Contours')
       
       # Plot zero-crossing rate
       axes[2].plot(features.zero_crossing_rate)
       axes[2].set_title('Zero Crossing Rate')
       
       plt.tight_layout()
       return fig

   def create_stutter_report(self, events: List[StutterEvent]) -> Figure:
       """
       Create summary report visualization.
       
       Args:
           events (List[StutterEvent]): Detected stutter events
           
       Returns:
           Figure: Matplotlib figure with report visualizations
       """
       fig = plt.figure(figsize=(12, 8))
       gs = fig.add_gridspec(2, 2)
       
       # Event distribution
       ax1 = fig.add_subplot(gs[0, 0])
       self._plot_event_distribution(ax1, events)
       
       # Severity distribution
       ax2 = fig.add_subplot(gs[0, 1])
       self._plot_severity_distribution(ax2, events)
       
       # Timeline
       ax3 = fig.add_subplot(gs[1, :])
       self._plot_event_timeline(ax3, events)
       
       plt.tight_layout()
       return fig

   def _plot_event_distribution(self, ax: plt.Axes, 
                              events: List[StutterEvent]) -> None:
       """Plot distribution of stutter types."""
       event_types = [event.stutter_type for event in events]
       sns.countplot(x=event_types, ax=ax, palette=self.colors)
       ax.set_title('Stutter Type Distribution')
       ax.set_xlabel('Type')
       ax.set_ylabel('Count')

   def _plot_severity_distribution(self, ax: plt.Axes, 
                                 events: List[StutterEvent]) -> None:
       """Plot distribution of stutter severities."""
       severities = [event.severity for event in events]
       sns.histplot(severities, ax=ax, bins=10)
       ax.set_title('Severity Distribution')
       ax.set_xlabel('Severity Score')
       ax.set_ylabel('Count')

   def _plot_event_timeline(self, ax: plt.Axes, 
                          events: List[StutterEvent]) -> None:
       """Plot timeline of stutter events."""
       for i, event in enumerate(events):
           ax.barh(i, event.duration(), 
                  left=event.start_time,
                  color=self.colors[event.stutter_type],
                  alpha=0.6)
       
       ax.set_title('Stutter Event Timeline')
       ax.set_xlabel('Time (s)')
       ax.set_ylabel('Event Number')

   def save_visualization(self, fig: Figure, name: str) -> str:
       """
       Save visualization to file.
       
       Args:
           fig (Figure): Figure to save
           name (str): Base name for file
           
       Returns:
           str: Path to saved file
       """
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       filename = f"{name}_{timestamp}.png"
       
       try:
           fig.savefig(filename, dpi=300, bbox_inches='tight')
           logger.info(f"Visualization saved to {filename}")
           return filename
       except Exception as e:
           logger.error(f"Error saving visualization: {e}")
           raise

# Example usage
if __name__ == "__main__":
   try:
       # Create visualizer
       visualizer = SpeechVisualizer()
       
       # Generate sample data (replace with real data)
       audio_data = np.random.randn(16000)
       features = SpeechFeatures(
           mfcc=np.random.randn(13, 100),
           pitch=np.random.randn(100),
           energy=np.abs(np.random.randn(100)),
           zero_crossing_rate=np.abs(np.random.randn(100)),
           duration=1.0,
           speech_rate=4.0
       )
       events = [
           StutterEvent(
               stutter_type=StutterType.REPETITION,
               start_time=0.1,
               end_time=0.3,
               confidence=0.8,
               severity=0.7
           ),
           StutterEvent(
               stutter_type=StutterType.BLOCK,
               start_time=0.5,
               end_time=0.7,
               confidence=0.9,
               severity=0.8
           )
       ]
       
       # Create visualizations
       dashboard = visualizer.create_analysis_dashboard(
           audio_data, features, events, 16000
       )
       
       # Save visualization
       visualizer.save_visualization(dashboard, "analysis_dashboard")
       
   except Exception as e:
       print(f"Error in visualization: {e}")