"""
Audio processing module for the stutter detection system.
Handles all audio-related functionality including recording, processing, and analysis.
"""

from .audio_config import AudioConfig, AudioDeviceConfig, AudioFormat
from .audio_recorder import AudioRecorder, AudioRecordingError

__all__ = [
    'AudioConfig',
    'AudioDeviceConfig',
    'AudioFormat',
    'AudioRecorder',
    'AudioRecordingError'
]