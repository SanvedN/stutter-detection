"""
feature_extractor.py

Feature extraction module for stutter detection system.
Extracts and processes relevant speech features for stutter analysis.

Key Features:
    - MFCC extraction
    - Pitch tracking
    - Energy analysis
    - Duration measurements
    - Zero-crossing rate analysis
    - Speech rate calculation
    - Spectral analysis for prolonged sounds
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import scipy.signal as signal

from src.audio.audio_config import AudioConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpeechFeatures:
    """
    Container for extracted speech features.

    Attributes:
        mfcc (np.ndarray): Mel-frequency cepstral coefficients
        pitch (np.ndarray): Fundamental frequency contour
        energy (np.ndarray): Energy contour
        zero_crossing_rate (np.ndarray): Zero-crossing rates
        duration (float): Duration in seconds
        speech_rate (float): Syllables per second
        spectral_flux (np.ndarray): Spectral flux for detecting sound changes
        formants (np.ndarray): Formant frequencies for vowel analysis
    """

    mfcc: np.ndarray
    pitch: np.ndarray
    energy: np.ndarray
    zero_crossing_rate: np.ndarray
    duration: float
    speech_rate: float
    spectral_flux: Optional[np.ndarray] = None
    formants: Optional[np.ndarray] = None


class FeatureExtractor:
    """
    Extracts and processes speech features for stutter detection.

    This class handles the extraction of various acoustic features
    that are relevant for detecting different types of stutters.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize feature extractor with configuration.

        Args:
            config (Optional[AudioConfig]): Audio configuration settings.
                If None, default settings will be used.
        """
        self.config = config or AudioConfig()

        # MFCC parameters
        self.n_mfcc = 13
        self.n_mels = 40
        self.win_length = int(0.025 * self.config.sample_rate)  # 25ms window
        self.hop_length = int(0.010 * self.config.sample_rate)  # 10ms hop

        logger.info("FeatureExtractor initialized with enhanced parameters")

    def extract_features(self, audio_data: np.ndarray) -> SpeechFeatures:
        """
        Extract all relevant features from audio signal.

        Args:
            audio_data (np.ndarray): Preprocessed audio signal

        Returns:
            SpeechFeatures: Container with all extracted features
        """
        try:
            # Extract individual features
            mfcc = self._extract_mfcc(audio_data)
            pitch = self._extract_pitch(audio_data)
            energy = self._extract_energy(audio_data)
            zcr = self._extract_zero_crossing_rate(audio_data)
            duration = len(audio_data) / self.config.sample_rate
            speech_rate = self._calculate_speech_rate(audio_data)
            spectral_flux = self._extract_spectral_flux(audio_data)
            formants = self._extract_formants(audio_data)

            # Combine into feature container
            features = SpeechFeatures(
                mfcc=mfcc,
                pitch=pitch,
                energy=energy,
                zero_crossing_rate=zcr,
                duration=duration,
                speech_rate=speech_rate,
                spectral_flux=spectral_flux,
                formants=formants,
            )

            logger.info("Feature extraction completed successfully")
            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

    def _extract_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.config.sample_rate,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

        # Add delta and delta-delta features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        return np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    def _extract_pitch(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract pitch contour from audio.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Pitch contour
        """
        pitches, magnitudes = librosa.piptrack(
            y=audio_data,
            sr=self.config.sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

        # Get the most prominent pitch in each frame
        pitch_contour = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch_contour.append(pitches[index, i])

        return np.array(pitch_contour)

    def _extract_energy(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract energy contour from audio.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Energy contour
        """
        return librosa.feature.rms(
            y=audio_data, frame_length=self.win_length, hop_length=self.hop_length
        )[0]

    def _extract_zero_crossing_rate(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Calculate zero-crossing rate.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Zero-crossing rates
        """
        return librosa.feature.zero_crossing_rate(
            y=audio_data, frame_length=self.win_length, hop_length=self.hop_length
        )[0]

    def _extract_spectral_flux(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Calculate spectral flux for detecting sound changes.

        Useful for detecting prolonged sounds and transitions.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Spectral flux
        """
        # Calculate spectrogram
        D = librosa.stft(audio_data, n_fft=self.win_length, hop_length=self.hop_length)

        # Convert to magnitude
        magnitude = np.abs(D)

        # Calculate flux (difference between consecutive frames)
        flux = np.zeros(magnitude.shape[1] - 1)
        for i in range(len(flux)):
            flux[i] = np.sum((magnitude[:, i + 1] - magnitude[:, i]) ** 2)

        return flux

    def _extract_formants(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract formant frequencies for vowel analysis.

        Useful for detecting prolonged vowels.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            np.ndarray: Formant frequencies
        """
        # Pre-emphasis to amplify high frequencies
        pre_emphasis = 0.97
        emphasized_signal = np.append(
            audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1]
        )

        # Frame the signal
        frame_length = self.win_length
        hop_length = self.hop_length

        # Number of frames
        num_frames = 1 + (len(emphasized_signal) - frame_length) // hop_length

        # Initialize formants array (3 formants per frame)
        formants = np.zeros((num_frames, 3))

        # Process each frame
        for i in range(num_frames):
            # Extract frame
            start = i * hop_length
            end = start + frame_length
            frame = emphasized_signal[start:end]

            # Apply window
            frame = frame * np.hamming(frame_length)

            # LPC analysis
            try:
                A = librosa.lpc(frame, order=8)
                roots = np.roots(A)

                # Keep only roots with positive imaginary part
                roots = roots[np.imag(roots) > 0]

                # Convert to frequency
                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = angles * self.config.sample_rate / (2 * np.pi)

                # Sort by frequency
                freqs = np.sort(freqs)

                # Store first 3 formants (or fewer if not enough found)
                for j in range(min(3, len(freqs))):
                    formants[i, j] = freqs[j]
            except:
                # In case of error, leave as zeros
                pass

        return formants

    def _calculate_speech_rate(self, audio_data: np.ndarray) -> float:
        """
        Estimate speech rate in syllables per second.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            float: Estimated speech rate
        """
        # Detect peaks in energy contour
        energy = self._extract_energy(audio_data)

        # Smooth energy contour
        energy_smooth = np.convolve(energy, np.ones(5) / 5, mode="same")

        # Find peaks with dynamic threshold
        threshold = np.mean(energy_smooth) * 0.5
        peaks = []

        for i in range(1, len(energy_smooth) - 1):
            if (
                energy_smooth[i] > energy_smooth[i - 1]
                and energy_smooth[i] > energy_smooth[i + 1]
                and energy_smooth[i] > threshold
            ):
                peaks.append(i)

        # Each peak roughly corresponds to a syllable
        duration = len(audio_data) / self.config.sample_rate
        speech_rate = len(peaks) / duration if duration > 0 else 0

        return speech_rate

    def analyze_prolonged_sounds(self, audio_data: np.ndarray) -> List[Dict]:
        """
        Specialized analysis for detecting prolonged sounds.

        Args:
            audio_data (np.ndarray): Audio signal

        Returns:
            List[Dict]: Detected prolonged sounds with timing information
        """
        prolongations = []

        # Extract features specifically useful for prolongation detection
        energy = self._extract_energy(audio_data)
        zcr = self._extract_zero_crossing_rate(audio_data)
        spectral_flux = self._extract_spectral_flux(audio_data)

        # Parameters
        min_duration = int(
            0.15 / (self.hop_length / self.config.sample_rate)
        )  # 150ms minimum

        # Detect regions with stable spectral characteristics
        i = 0
        while i < len(spectral_flux) - min_duration:
            # Check for low spectral flux (stable sound)
            if np.mean(spectral_flux[i : i + min_duration]) < 0.1:
                # Determine type of sound
                if np.mean(zcr[i : i + min_duration]) > 0.3:
                    sound_type = (
                        "fricative"  # High ZCR indicates fricatives like 's', 'sh', 'f'
                    )
                else:
                    sound_type = "vowel"  # Lower ZCR indicates vowels

                # Find end of prolongation
                j = i + min_duration
                while j < len(spectral_flux) and spectral_flux[j] < 0.1:
                    j += 1

                # Convert to time
                start_time = i * (self.hop_length / self.config.sample_rate)
                end_time = j * (self.hop_length / self.config.sample_rate)

                # Add to results if duration is significant
                if end_time - start_time >= 0.15:  # At least 150ms
                    prolongations.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "duration": end_time - start_time,
                            "type": sound_type,
                            "confidence": 1.0 - np.mean(spectral_flux[i:j]),
                        }
                    )

                i = j
            else:
                i += 1

        return prolongations

    def get_feature_statistics(self, features: SpeechFeatures) -> Dict[str, float]:
        """
        Calculate statistical measures from extracted features.

        Args:
            features (SpeechFeatures): Extracted features

        Returns:
            Dict[str, float]: Statistical measures
        """
        stats = {
            "mean_pitch": (
                np.mean(features.pitch[features.pitch > 0])
                if np.any(features.pitch > 0)
                else 0
            ),
            "std_pitch": (
                np.std(features.pitch[features.pitch > 0])
                if np.any(features.pitch > 0)
                else 0
            ),
            "mean_energy": np.mean(features.energy),
            "std_energy": np.std(features.energy),
            "mean_zcr": np.mean(features.zero_crossing_rate),
            "speech_rate": features.speech_rate,
            "duration": features.duration,
        }

        # Add spectral flux statistics if available
        if features.spectral_flux is not None:
            stats["mean_spectral_flux"] = np.mean(features.spectral_flux)
            stats["std_spectral_flux"] = np.std(features.spectral_flux)

        return stats


# Example usage
if __name__ == "__main__":
    try:
        # Create feature extractor
        extractor = FeatureExtractor()

        # Generate sample audio (replace with real audio)
        sample_audio = np.random.randn(16000)  # 1 second of random noise

        # Extract features
        features = extractor.extract_features(sample_audio)

        # Get statistics
        stats = extractor.get_feature_statistics(features)
        print("Feature statistics:", stats)

    except Exception as e:
        print(f"Error in feature extraction: {e}")
