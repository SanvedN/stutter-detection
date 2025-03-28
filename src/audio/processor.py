"""
processor.py

Audio preprocessing module for stutter detection system.
Handles cleaning, enhancement, and segmentation of audio signals.

Key Features:
   - Audio validation and normalization
   - Noise reduction (adaptive)
   - Speech enhancement with adjusted bandpass filter
   - Audio segmentation
   - Silence detection for block analysis
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from scipy.signal import butter, filtfilt
import librosa

from src.audio.audio_config import AudioConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Handles all preprocessing operations for audio analysis.

    This class provides methods for cleaning and preparing audio signals
    for stutter detection analysis.

    Attributes:
        config (AudioConfig): Audio configuration settings
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize preprocessor with configuration.

        Args:
            config (Optional[AudioConfig]): Audio configuration settings.
                If None, default settings will be used.
        """
        self.config = config or AudioConfig()
        logger.info("AudioPreprocessor initialized")

    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Main processing pipeline for audio data.

        Args:
            audio_data (np.ndarray): Raw audio signal

        Returns:
            np.ndarray: Processed audio signal
        """
        try:
            # Validate input
            if not self._validate_audio(audio_data):
                raise ValueError("Invalid audio data")

            # Apply processing steps
            audio = self._normalize_audio(audio_data)
            audio = self._reduce_noise(audio)
            audio = self._enhance_speech(audio)
            # Instead of removing all silences, remove only non-critical ones.
            processed_audio = self._remove_non_critical_silences(audio)

            logger.info("Audio processing completed successfully")
            return processed_audio

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise

    def _validate_audio(self, audio_data: np.ndarray) -> bool:
        """
        Validate audio data format and quality.

        Args:
            audio_data (np.ndarray): Audio data to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(audio_data, np.ndarray):
            logger.error("Audio data must be numpy array")
            return False

        if len(audio_data.shape) != 1:
            logger.error("Audio must be mono channel")
            return False

        if len(audio_data) == 0:
            logger.error("Audio data is empty")
            return False

        return True

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio volume to a consistent level using RMS normalization.

        Args:
            audio_data (np.ndarray): Audio signal to normalize

        Returns:
            np.ndarray: Normalized audio signal
        """
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            normalized = audio_data / rms
        else:
            normalized = audio_data
        return normalized

    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Reduce background noise in audio signal using adaptive noise estimation.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Noise-reduced audio signal
        """
        S = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        mag, phase = np.abs(S), np.angle(S)
        noise_est = np.percentile(mag, 20, axis=1, keepdims=True)
        factor = 1.5
        threshold = factor * noise_est
        mask = mag > threshold
        mag_denoised = mag * mask
        S_denoised = mag_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(S_denoised, hop_length=512)
        return audio_denoised

    def _enhance_speech(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance speech frequencies in audio signal.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Enhanced audio signal
        """
        nyquist = self.config.sample_rate / 2
        low_cutoff = 80 / nyquist  # 80 Hz
        high_cutoff = 4000 / nyquist  # Adjusted to 4000 Hz for broader speech energy

        b, a = butter(4, [low_cutoff, high_cutoff], btype="band")
        enhanced = filtfilt(b, a, audio_data)

        return enhanced

    def _remove_non_critical_silences(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Remove silent segments from audio while preserving natural pauses.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Audio with only non-critical silences removed
        """
        intervals = librosa.effects.split(
            audio_data,
            top_db=25,
            frame_length=2048,
            hop_length=512,
        )
        non_silent = []
        for start, end in intervals:
            non_silent.extend(audio_data[start:end])
        return np.array(non_silent)

    def detect_silences(self, audio_data: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect long silences in the audio (ignoring silences at the start and end).

        Returns:
            List[Tuple[float, float]]: List of silence intervals (start, end) in seconds.
        """
        intervals = librosa.effects.split(
            audio_data,
            top_db=25,
            frame_length=2048,
            hop_length=512,
        )
        silences = []
        if len(intervals) < 2:
            return silences

        sr = self.config.sample_rate
        for idx in range(1, len(intervals)):
            prev_end = intervals[idx - 1][1]
            current_start = intervals[idx][0]
            if prev_end == 0 or current_start == len(audio_data):
                continue
            silence_duration = (current_start - prev_end) / sr
            if silence_duration > 0.5:
                silences.append((prev_end / sr, current_start / sr))
        return silences

    def segment_audio(
        self, audio_data: np.ndarray, segment_length_ms: int = 1000
    ) -> List[np.ndarray]:
        """
        Segment audio into smaller chunks for analysis.

        Args:
            audio_data (np.ndarray): Audio signal
            segment_length_ms (int): Length of each segment in milliseconds

        Returns:
            List[np.ndarray]: List of audio segments
        """
        samples_per_segment = self.config.get_duration_samples(segment_length_ms)
        segments = []

        for start in range(0, len(audio_data), samples_per_segment):
            end = start + samples_per_segment
            if end <= len(audio_data):
                segment = audio_data[start:end]
                segments.append(segment)

        return segments

    def get_audio_quality_metrics(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate quality metrics for audio signal.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            Dict[str, float]: Dictionary of quality metrics
        """
        return {
            "snr": self._calculate_snr(audio_data),
            "peak_level": np.max(np.abs(audio_data)),
            "rms_level": np.sqrt(np.mean(audio_data**2)),
        }

    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            float: Estimated SNR in dB
        """
        noise_floor = np.mean(np.abs(audio_data[:1000]))
        signal_level = np.mean(np.abs(audio_data))

        if noise_floor == 0:
            return float("inf")

        snr = 20 * np.log10(signal_level / noise_floor)
        return snr


# Example usage
if __name__ == "__main__":
    try:
        preprocessor = AudioPreprocessor()
        test_audio = np.random.randn(16000)  # 1 second of random noise
        processed_audio = preprocessor.process_audio(test_audio)
        metrics = preprocessor.get_audio_quality_metrics(processed_audio)
        print("Audio quality metrics:", metrics)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
