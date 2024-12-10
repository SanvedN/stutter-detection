"""
Stutter Detection System
A comprehensive system for detecting and analyzing speech stutters.
"""

from .audio.audio_config import AudioConfig
from .audio.audio_recorder import AudioRecorder
from .audio.feature_extractor import FeatureExtractor
from .visualization.speech_visualizer import SpeechVisualizer

__all__ = [
    'AudioConfig',
    'AudioRecorder',
    'FeatureExtractor',
    'SpeechVisualizer'
]

__version__ = '0.1.0'
__author__ = 'Sanved'