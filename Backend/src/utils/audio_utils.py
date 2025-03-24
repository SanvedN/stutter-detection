"""
audio_utils.py

Utility functions for audio processing in the stutter detection system.
Provides common audio operations and helper functions.

Key Features:
    - Audio format conversion
    - Duration calculation
    - Signal processing utilities
    - Audio quality checks
    - File handling helpers
"""

import numpy as np
from typing import Tuple, Optional, Union
import librosa
import soundfile as sf
import logging
from pathlib import Path
import resampy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio(file_path: Union[str, Path], 
              target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file with automatic resampling.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sampling rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Load audio file
        audio_data, orig_sr = librosa.load(file_path, sr=None)
        
        # Resample if necessary
        if orig_sr != target_sr:
            audio_data = resampy.resample(audio_data, orig_sr, target_sr)
            logger.info(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz")
            
        return audio_data, target_sr
        
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        raise

def save_audio(audio_data: np.ndarray, 
               file_path: Union[str, Path], 
               sample_rate: int) -> None:
    """
    Save audio data to file.
    
    Args:
        audio_data: Audio signal
        file_path: Output file path
        sample_rate: Sampling rate
    """
    try:
        sf.write(file_path, audio_data, sample_rate)
        logger.info(f"Audio saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving audio file: {e}")
        raise

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1.0, 1.0] range.
    
    Args:
        audio_data: Input audio signal
        
    Returns:
        Normalized audio signal
    """
    try:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        raise

def trim_silence(audio_data: np.ndarray, 
                sample_rate: int,
                threshold_db: float = -60.0) -> np.ndarray:
    """
    Remove silence from start and end of audio.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sampling rate
        threshold_db: Silence threshold in dB
        
    Returns:
        Trimmed audio signal
    """
    try:
        trimmed, _ = librosa.effects.trim(
            audio_data, 
            top_db=-threshold_db,
            frame_length=2048,
            hop_length=512
        )
        return trimmed
    except Exception as e:
        logger.error(f"Error trimming silence: {e}")
        raise

def split_audio(audio_data: np.ndarray,
                sample_rate: int,
                segment_duration: float = 30.0) -> list[np.ndarray]:
    """
    Split audio into fixed-duration segments.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sampling rate
        segment_duration: Duration of each segment in seconds
        
    Returns:
        List of audio segments
    """
    try:
        segment_length = int(segment_duration * sample_rate)
        segments = []
        
        for start in range(0, len(audio_data), segment_length):
            end = start + segment_length
            if end <= len(audio_data):
                segments.append(audio_data[start:end])
            else:
                # Pad last segment if needed
                segment = np.zeros(segment_length)
                segment[:len(audio_data[start:])] = audio_data[start:]
                segments.append(segment)
                
        return segments
    except Exception as e:
        logger.error(f"Error splitting audio: {e}")
        raise

def get_audio_duration(audio_data: np.ndarray, 
                      sample_rate: int) -> float:
    """
    Calculate duration of audio in seconds.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sampling rate
        
    Returns:
        Duration in seconds
    """
    try:
        return len(audio_data) / sample_rate
    except Exception as e:
        logger.error(f"Error calculating duration: {e}")
        raise

def apply_noise_reduction(audio_data: np.ndarray,
                         sample_rate: int,
                         noise_reduce_amount: float = 0.75) -> np.ndarray:
    """
    Apply basic noise reduction to audio.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sampling rate
        noise_reduce_amount: Strength of noise reduction (0-1)
        
    Returns:
        Noise-reduced audio signal
    """
    try:
        # Calculate noise profile from first 1000ms
        noise_sample = audio_data[:int(sample_rate * 0.1)]
        noise_profile = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)
        
        # Apply spectral subtraction
        S = librosa.stft(audio_data)
        S_abs = np.abs(S)
        S_angle = np.angle(S)
        
        # Subtract noise profile
        S_abs_clean = np.maximum(
            S_abs - noise_reduce_amount * noise_profile.reshape(-1, 1),
            0
        )
        
        # Reconstruct signal
        S_clean = S_abs_clean * np.exp(1.0j * S_angle)
        audio_clean = librosa.istft(S_clean)
        
        return audio_clean
    except Exception as e:
        logger.error(f"Error applying noise reduction: {e}")
        raise

def check_audio_quality(audio_data: np.ndarray, 
                       sample_rate: int) -> dict[str, float]:
    """
    Check various audio quality metrics.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sampling rate
        
    Returns:
        Dictionary of quality metrics
    """
    try:
        metrics = {
            'duration': get_audio_duration(audio_data, sample_rate),
            'rms_level': float(np.sqrt(np.mean(audio_data**2))),
            'peak_level': float(np.max(np.abs(audio_data))),
            'silent_percentage': _get_silent_percentage(audio_data),
            'snr': _estimate_snr(audio_data)
        }
        return metrics
    except Exception as e:
        logger.error(f"Error checking audio quality: {e}")
        raise

def _get_silent_percentage(audio_data: np.ndarray, 
                         threshold: float = 0.001) -> float:
    """Calculate percentage of silent samples."""
    silent_samples = np.sum(np.abs(audio_data) < threshold)
    return (silent_samples / len(audio_data)) * 100

def _estimate_snr(audio_data: np.ndarray) -> float:
    """Estimate Signal-to-Noise Ratio."""
    noise_floor = np.mean(np.abs(audio_data[:1000]))  # Use first 1000 samples
    signal_level = np.mean(np.abs(audio_data))
    
    if noise_floor == 0:
        return float('inf')
        
    return 20 * np.log10(signal_level / noise_floor)

def convert_audio_format(input_path: Union[str, Path],
                        output_path: Union[str, Path],
                        target_sr: int = 16000) -> None:
    """
    Convert audio file format and sampling rate.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        target_sr: Target sampling rate
    """
    try:
        # Load audio
        audio_data, orig_sr = load_audio(input_path, target_sr)
        
        # Save in new format
        save_audio(audio_data, output_path, target_sr)
        
        logger.info(f"Converted audio from {input_path} to {output_path}")
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Load test audio file
        audio_file = "test_audio.wav"  # replace with actual file
        audio_data, sample_rate = load_audio(audio_file)
        
        # Process audio
        audio_data = normalize_audio(audio_data)
        audio_data = trim_silence(audio_data, sample_rate)
        audio_data = apply_noise_reduction(audio_data, sample_rate)
        
        # Check quality
        quality_metrics = check_audio_quality(audio_data, sample_rate)
        print("Audio Quality Metrics:", quality_metrics)
        
        # Save processed audio
        save_audio(audio_data, "processed_audio.wav", sample_rate)
        
    except Exception as e:
        print(f"Error processing audio: {e}")