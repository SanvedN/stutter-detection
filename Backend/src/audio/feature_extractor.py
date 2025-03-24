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
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

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
    """
    mfcc: np.ndarray
    pitch: np.ndarray
    energy: np.ndarray
    zero_crossing_rate: np.ndarray
    duration: float
    speech_rate: float

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
        
        logger.info("FeatureExtractor initialized")

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
            
            # Combine into feature container
            features = SpeechFeatures(
                mfcc=mfcc,
                pitch=pitch,
                energy=energy,
                zero_crossing_rate=zcr,
                duration=duration,
                speech_rate=speech_rate
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
            hop_length=self.hop_length
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
            hop_length=self.hop_length
        )
        
        # Get the most prominent pitch in each frame
        pitch_contour = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:,i].argmax()
            pitch_contour.append(pitches[index,i])
            
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
            y=audio_data,
            frame_length=self.win_length,
            hop_length=self.hop_length
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
            y=audio_data,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )[0]

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
        peaks = librosa.util.peak_pick(
            energy,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.1,
            wait=10
        )
        
        # Each peak roughly corresponds to a syllable
        duration = len(audio_data) / self.config.sample_rate
        speech_rate = len(peaks) / duration
        
        return speech_rate

    def get_feature_statistics(self, features: SpeechFeatures) -> Dict[str, float]:
        """
        Calculate statistical measures from extracted features.
        
        Args:
            features (SpeechFeatures): Extracted features
            
        Returns:
            Dict[str, float]: Statistical measures
        """
        stats = {
            'mean_pitch': np.mean(features.pitch),
            'std_pitch': np.std(features.pitch),
            'mean_energy': np.mean(features.energy),
            'speech_rate': features.speech_rate,
            'duration': features.duration
        }
        
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