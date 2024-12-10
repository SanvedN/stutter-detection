"""
audio_recorder.py

This module provides audio recording functionality for the stutter detection system.
It handles real-time audio capture, buffer management, and basic audio preprocessing.

Key Features:
    - Real-time audio recording
    - Buffer management
    - Audio level monitoring
    - Silence detection
    - Automatic segmentation
    
Example:
    >>> recorder = AudioRecorder()
    >>> recorder.start_recording()
    >>> # ... recording in progress
    >>> audio_data = recorder.stop_recording()
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Tuple, List
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path

from src.audio.audio_config import AudioConfig, AudioDeviceConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioRecordingError(Exception):
    """Custom exception for audio recording errors."""
    pass

class AudioRecorder:
    """
    Handles audio recording and basic preprocessing for the stutter detection system.
    
    This class manages audio capture from the microphone, including buffer management,
    real-time audio level monitoring, and basic preprocessing operations.
    
    Attributes:
        config (AudioConfig): Audio configuration settings
        device_config (AudioDeviceConfig): Audio device settings
        is_recording (bool): Current recording state
        is_paused (bool): Current pause state
        _audio_buffer (queue.Queue): Thread-safe buffer for audio data
        _recording_thread (threading.Thread): Thread for handling recording
    """
    
    def __init__(self, 
                 config: Optional[AudioConfig] = None,
                 device_config: Optional[AudioDeviceConfig] = None):
        """
        Initialize the audio recorder with specified configuration.
        
        Args:
            config (Optional[AudioConfig]): Audio configuration settings.
                If None, default settings will be used.
            device_config (Optional[AudioDeviceConfig]): Audio device configuration.
                If None, default device will be used.
        """
        self.config = config or AudioConfig()
        self.device_config = device_config or AudioDeviceConfig()
        
        # Validate configurations
        if not self.config.validate():
            raise AudioRecordingError("Invalid audio configuration")
        if not self.device_config.setup_devices():
            raise AudioRecordingError("Failed to setup audio device")
            
        # Initialize state
        self.is_recording = False
        self.is_paused = False
        self._audio_buffer = queue.Queue()
        self._recording_thread = None
        
        # Audio monitoring
        self._audio_levels: List[float] = []
        self._silence_counter = 0
        
        logger.info("AudioRecorder initialized successfully")

    def _audio_callback(self, indata: np.ndarray, frames: int, 
                       time_info: dict, status: sd.CallbackFlags) -> None:
        """
        Callback function for audio stream processing.
        
        Handles incoming audio data from sounddevice stream. Performs basic
        preprocessing and adds data to the buffer.
        
        Args:
            indata (np.ndarray): Input audio data
            frames (int): Number of frames
            time_info (dict): Timing information
            status (sd.CallbackFlags): Stream status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if self.is_recording and not self.is_paused:
            # Calculate audio level
            audio_level = np.max(np.abs(indata))
            self._audio_levels.append(audio_level)
            
            # Check for silence
            if audio_level < self.config.silence_threshold:
                self._silence_counter += 1
            else:
                self._silence_counter = 0
                
            # Add to buffer
            self._audio_buffer.put(indata.copy())

    def start_recording(self) -> None:
        """
        Start audio recording.
        
        Initializes audio stream and begins capturing audio data.
        
        Raises:
            AudioRecordingError: If recording cannot be started or is already in progress.
        """
        if self.is_recording:
            raise AudioRecordingError("Recording is already in progress")
            
        try:
            # Reset state
            self._audio_buffer = queue.Queue()
            self._audio_levels = []
            self._silence_counter = 0
            self.is_paused = False
            
            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype.value,
                blocksize=self.config.block_size,
                callback=self._audio_callback,
                device=self.device_config.device_id
            )
            
            self.stream.start()
            self.is_recording = True
            logger.info("Recording started")
            
        except Exception as e:
            raise AudioRecordingError(f"Failed to start recording: {e}")

    def stop_recording(self) -> Tuple[np.ndarray, float]:
        """
        Stop audio recording and return recorded data.
        
        Returns:
            Tuple[np.ndarray, float]: Recorded audio data and its duration in seconds.
            
        Raises:
            AudioRecordingError: If no recording is in progress or stopping fails.
        """
        if not self.is_recording:
            raise AudioRecordingError("No recording in progress")
            
        try:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            
            # Collect all data from buffer
            audio_chunks = []
            while not self._audio_buffer.empty():
                audio_chunks.append(self._audio_buffer.get())
                
            # Combine chunks and calculate duration
            audio_data = np.concatenate(audio_chunks, axis=0)
            duration = len(audio_data) / self.config.sample_rate
            
            logger.info(f"Recording stopped. Duration: {duration:.2f} seconds")
            return audio_data, duration
            
        except Exception as e:
            raise AudioRecordingError(f"Failed to stop recording: {e}")

    def pause_recording(self) -> None:
        """Pause current recording."""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            logger.info("Recording paused")

    def resume_recording(self) -> None:
        """Resume paused recording."""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            logger.info("Recording resumed")

    def get_audio_level(self) -> float:
        """
        Get current audio level.
        
        Returns:
            float: Current audio level (0.0 to 1.0)
        """
        if not self._audio_levels:
            return 0.0
        return float(np.mean(self._audio_levels[-10:]))

    def save_recording(self, audio_data: np.ndarray, 
                      filepath: Optional[str] = None) -> str:
        """
        Save recorded audio to file.
        
        Args:
            audio_data (np.ndarray): Audio data to save
            filepath (Optional[str]): Path to save file. If None, generates automatic name.
            
        Returns:
            str: Path to saved file
        """
        try:
            from scipy.io import wavfile
            
            if filepath is None:
                # Generate automatic filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"recording_{timestamp}.wav"
                
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            wavfile.write(
                filepath,
                self.config.sample_rate,
                audio_data
            )
            
            logger.info(f"Recording saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            raise AudioRecordingError(f"Failed to save recording: {e}")

# Example usage
if __name__ == "__main__":
    try:
        # Create recorder instance
        recorder = AudioRecorder()
        
        # Start recording
        recorder.start_recording()
        print("Recording... Press Ctrl+C to stop")
        
        # Wait for keyboard interrupt
        try:
            while True:
                level = recorder.get_audio_level()
                print(f"Audio level: {level:.2f}", end='\r')
                
        except KeyboardInterrupt:
            print("\nStopping recording...")
        
        # Stop and save recording
        audio_data, duration = recorder.stop_recording()
        filepath = recorder.save_recording(audio_data)
        print(f"Recording saved to: {filepath}")
        
    except AudioRecordingError as e:
        print(f"Recording error: {e}")