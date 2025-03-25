"""
preprocessor.py

Audio preprocessing module for stutter detection system.
Handles cleaning, enhancement, and segmentation of audio signals.

Key Features:
   - Audio validation and normalization
   - Noise reduction (adaptive)
   - Speech enhancement with adjusted bandpass filter
   - Audio segmentation
   - Quality checks
"""

import numpy as np
from typing import Tuple, Optional, Dict
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
        _noise_profile (Optional[np.ndarray]): Stored noise profile for reduction
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize preprocessor with configuration.

        Args:
            config (Optional[AudioConfig]): Audio configuration settings.
                If None, default settings will be used.
        """
        self.config = config or AudioConfig()
        self._noise_profile = None
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
            audio = self._remove_silence(audio)

            logger.info("Audio processing completed successfully")
            return audio

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
        # RMS normalization to account for sustained loudness rather than transient peaks
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            normalized = audio_data / rms
        else:
            normalized = audio_data
        return normalized

    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Reduce background noise in audio signal using adaptive noise estimation.

        Uses a modified spectral subtraction method with dynamic thresholding.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Noise-reduced audio signal
        """
        # Compute STFT
        S = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        mag, phase = np.abs(S), np.angle(S)

        # Estimate noise from quiet frames (using a lower percentile)
        noise_est = np.percentile(mag, 20, axis=1, keepdims=True)

        # Adaptive threshold: noise estimate multiplied by a factor
        factor = 1.5
        threshold = factor * noise_est

        # Create mask and reduce magnitude
        mask = mag > threshold
        mag_denoised = mag * mask

        # Reconstruct signal
        S_denoised = mag_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(S_denoised, hop_length=512)

        return audio_denoised

    def _enhance_speech(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhance speech frequencies in audio signal.

        Applies an improved bandpass filter to emphasize speech frequencies.

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

    def _remove_silence(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Remove silent segments from audio while preserving natural pauses.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Audio with silences removed
        """
        # Adjusted top_db to 25 to preserve short pauses that may indicate stutter boundaries
        intervals = librosa.effects.split(
            audio_data,
            top_db=25,  # Slightly lower threshold
            frame_length=2048,
            hop_length=512,
        )

        # Concatenate non-silent parts
        non_silent = []
        for start, end in intervals:
            non_silent.extend(audio_data[start:end])

        return np.array(non_silent)

    def segment_audio(
        self, audio_data: np.ndarray, segment_length_ms: int = 1000
    ) -> list[np.ndarray]:
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

        # Split audio into segments
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
        noise_floor = np.mean(
            np.abs(audio_data[:1000])
        )  # Estimate from first 1000 samples
        signal_level = np.mean(np.abs(audio_data))

        if noise_floor == 0:
            return float("inf")

        snr = 20 * np.log10(signal_level / noise_floor)
        return snr


# Example usage
if __name__ == "__main__":
    try:
        # Create preprocessor
        preprocessor = AudioPreprocessor()

        # Load test audio (replace with actual audio data)
        test_audio = np.random.randn(16000)  # 1 second of random noise

        # Process audio
        processed_audio = preprocessor.process_audio(test_audio)

        # Get quality metrics
        metrics = preprocessor.get_audio_quality_metrics(processed_audio)
        print("Audio quality metrics:", metrics)

    except Exception as e:
        print(f"Error in preprocessing: {e}")
