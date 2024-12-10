"""
audio_config.py

This module provides configuration management for audio processing in the stutter detection system.
It handles all audio-related settings including sampling rate, channel configuration, buffer management,
and device setup.

Key Features:
    - Audio parameter configuration
    - Device management
    - Validation checks
    - Helper utilities for audio calculations

Example:
    >>> config = AudioConfig()
    >>> if config.validate():
    >>>     device_config = AudioDeviceConfig()
    >>>     device_config.setup_devices()
"""

import dataclasses
from typing import Optional, Dict, Union, List
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFormat(Enum):
    """
    Supported audio format types.
    
    Attributes:
        FLOAT32: 32-bit floating point
        INT16: 16-bit integer
        INT32: 32-bit integer
    """
    FLOAT32 = 'float32'
    INT16 = 'int16'
    INT32 = 'int32'

@dataclasses.dataclass
class AudioConfig:
    """
    Configuration settings for audio processing in the stutter detection system.
    
    Attributes:
        sample_rate (int): Sampling frequency in Hz. Default is 16000 (16kHz),
            which is optimal for speech recognition.
        channels (int): Number of audio channels. Default is 1 (mono),
            as speech analysis typically requires single channel audio.
        dtype (AudioFormat): Data type for audio samples. Default is FLOAT32.
        block_size (int): Number of samples per processing block. Default is 1024.
            Must be a power of 2 for efficient FFT processing.
        buffer_size (int): Internal buffer size in samples. Default is 4096.
            Must be larger than block_size to prevent buffer underruns.
        max_recording_seconds (int): Maximum allowed recording duration in seconds.
        silence_threshold (float): Amplitude threshold for silence detection.
            Values below this are considered silence.
    
    Note:
        The configuration parameters are optimized for speech processing and
        stutter detection. Modifying these values may affect system performance.
    """
    
    sample_rate: int = 16000
    channels: int = 1
    dtype: AudioFormat = AudioFormat.FLOAT32
    block_size: int = 1024
    buffer_size: int = 4096
    max_recording_seconds: int = 300
    silence_threshold: float = 0.03

    def validate(self) -> bool:
        """
        Validate all configuration parameters.
        
        Performs comprehensive validation of all audio settings to ensure they
        are within acceptable ranges and maintain internal consistency.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        try:
            # Validate sample rate
            valid_rates = [8000, 16000, 22050, 44100, 48000]
            if self.sample_rate not in valid_rates:
                raise ValueError(
                    f"Sample rate must be one of {valid_rates}, "
                    f"got {self.sample_rate}"
                )
            
            # Validate channels
            if not 1 <= self.channels <= 2:
                raise ValueError(
                    f"Channel count must be 1 or 2, got {self.channels}"
                )
            
            # Validate block size
            if not (self.block_size > 0 and (self.block_size & (self.block_size - 1) == 0)):
                raise ValueError(
                    f"Block size must be a power of 2, got {self.block_size}"
                )
            
            # Validate buffer size
            if self.buffer_size < self.block_size:
                raise ValueError(
                    f"Buffer size ({self.buffer_size}) must be greater than "
                    f"block size ({self.block_size})"
                )
            
            return True
            
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_duration_samples(self, duration_ms: float) -> int:
        """
        Convert duration from milliseconds to number of samples.
        
        Args:
            duration_ms (float): Duration in milliseconds.
            
        Returns:
            int: Number of samples for the given duration.
        """
        return int(self.sample_rate * duration_ms / 1000)

    def get_bytes_per_sample(self) -> int:
        """
        Calculate number of bytes per audio sample based on data type.
        
        Returns:
            int: Number of bytes per sample.
        """
        dtype_sizes: Dict[AudioFormat, int] = {
            AudioFormat.FLOAT32: 4,
            AudioFormat.INT16: 2,
            AudioFormat.INT32: 4
        }
        return dtype_sizes.get(self.dtype, 4)

class AudioDeviceConfig:
    """
    Configuration and management of audio input/output devices.
    
    This class handles device selection, validation, and setup for audio
    input and output operations.
    
    Attributes:
        device_id (Optional[int]): ID of the audio device to use.
            If None, the system default device is used.
        input_device (dict): Information about the configured input device.
        output_device (dict): Information about the configured output device.
    """
    
    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize audio device configuration.
        
        Args:
            device_id (Optional[int]): ID of the audio device to use.
                If None, the system default device will be used.
        """
        self.device_id = device_id
        self.input_device = None
        self.output_device = None
        
    def setup_devices(self) -> bool:
        """
        Setup and validate audio devices for recording and playback.
        
        Performs device discovery, validation, and configuration. Ensures the
        selected devices are available and suitable for audio processing.
        
        Returns:
            bool: True if device setup was successful, False otherwise.
            
        Raises:
            RuntimeError: If sounddevice module is not available.
            ValueError: If specified device is not found or invalid.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice module not found")
            return False

        try:
            # Get list of available devices
            devices = sd.query_devices()
            logger.info(f"Found {len(devices)} audio devices")
            
            # Use default device if none specified
            if self.device_id is None:
                self.device_id = sd.default.device[0]
                logger.info(f"Using default device ID: {self.device_id}")
            
            # Validate device exists
            if self.device_id >= len(devices):
                raise ValueError(f"Device ID {self.device_id} not found")
            
            # Store device information
            self.input_device = devices[self.device_id]
            logger.info(f"Configured input device: {self.input_device['name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Device setup failed: {e}")
            return False

    def get_device_info(self) -> Dict[str, Union[str, int, float]]:
        """
        Get information about the currently configured audio device.
        
        Returns:
            Dict[str, Union[str, int, float]]: Dictionary containing device information.
        """
        if self.input_device is None:
            return {}
            
        return {
            'name': self.input_device['name'],
            'max_input_channels': self.input_device['max_input_channels'],
            'default_sample_rate': self.input_device['default_samplerate']
        }

# Example usage and testing
if __name__ == "__main__":
    # Create and validate configuration
    config = AudioConfig()
    if config.validate():
        logger.info("Audio configuration validated successfully")
        
        # Setup audio devices
        device_config = AudioDeviceConfig()
        if device_config.setup_devices():
            logger.info("Audio devices configured successfully")
            logger.info(f"Device info: {device_config.get_device_info()}")