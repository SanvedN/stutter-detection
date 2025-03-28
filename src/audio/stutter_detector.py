"""
stutter_detector.py

Core stutter detection module for analyzing speech patterns and identifying stutters.
Uses extracted features to detect and classify different types of stutters.

Key Features:
   - Multiple stutter type detection
   - Pattern analysis with dynamic windowing
   - Confidence scoring with refined thresholds
   - Detailed reporting
"""

import numpy as np
from typing import Dict, List
import logging
from dataclasses import dataclass
from enum import Enum
import librosa

from src.audio.feature_extractor import SpeechFeatures

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StutterType(Enum):
    """Types of stutters that can be detected."""

    REPETITION = "repetition"
    PROLONGATION = "prolongation"
    BLOCK = "block"
    INTERJECTION = "interjection"


@dataclass
class StutterEvent:
    """
    Container for detected stutter events.

    Attributes:
        stutter_type (StutterType): Type of stutter detected
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        confidence (float): Detection confidence (0-1)
        severity (float): Estimated severity (0-1)
    """

    stutter_type: StutterType
    start_time: float
    end_time: float
    confidence: float
    severity: float

    def duration(self) -> float:
        """Calculate duration of stutter event."""
        return self.end_time - self.start_time


class StutterDetector:
    """
    Main class for detecting and analyzing stutters in speech.

    This class processes speech features to identify different types
    of stutters and provide detailed analysis.
    """

    def __init__(self):
        """Initialize stutter detector with default parameters."""
        self.repetition_threshold = 0.90  # Strict threshold for repetitions
        self.prolongation_threshold = 0.80
        self.block_threshold = 0.75
        self.window_size_repetition = 10  # Frames for sliding window
        self.step_size = 1  # Frame step for sliding window

        logger.info("StutterDetector initialized with strict thresholds")

    def analyze_speech(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Analyze speech features to detect stutters.

        Args:
            features (SpeechFeatures): Extracted speech features

        Returns:
            List[StutterEvent]: List of detected stutter events
        """
        stutter_events = []
        repetitions = self._detect_repetitions(features)
        prolongations = self._detect_prolongations(features)
        blocks = self._detect_blocks(features)
        prolonged_sounds = self._detect_prolonged_sounds(features)

        stutter_events.extend(repetitions)
        stutter_events.extend(prolongations)
        stutter_events.extend(blocks)
        stutter_events.extend(prolonged_sounds)
        stutter_events.sort(key=lambda x: x.start_time)

        logger.info(f"Detected {len(stutter_events)} stutter events")
        return stutter_events

    def _detect_repetitions(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect repetitions using a sliding window on MFCC features.
        """
        repetitions = []
        mfcc = features.mfcc  # Shape: (features, frames)
        n_frames = mfcc.shape[1]

        for i in range(0, n_frames - 2 * self.window_size_repetition, self.step_size):
            segment1 = mfcc[:, i : i + self.window_size_repetition]
            segment2 = mfcc[
                :, i + self.window_size_repetition : i + 2 * self.window_size_repetition
            ]
            corr = np.corrcoef(segment1.flatten(), segment2.flatten())[0, 1]
            corr = max(0.0, corr)

            if corr >= self.repetition_threshold:
                start_time = i * 0.01  # Assuming 10ms hop
                end_time = (i + 2 * self.window_size_repetition) * 0.01
                repetitions.append(
                    StutterEvent(
                        stutter_type=StutterType.REPETITION,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=corr,
                        severity=min(1.0, corr**2),
                    )
                )
                i += 2 * self.window_size_repetition
        return repetitions

    def _detect_prolongations(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect prolongations by evaluating stability in pitch and energy over time.
        """
        prolongations = []
        pitch = features.pitch  # In Hz; 0 indicates unvoiced
        energy = features.energy
        n_frames = len(pitch)
        window = 20  # Number of frames to evaluate

        for i in range(0, n_frames - window):
            segment_pitch = pitch[i : i + window]
            segment_energy = energy[i : i + window]

            # Consider only voiced frames for prolongation detection
            voiced = segment_pitch > 0
            if np.sum(voiced) < window * 0.8:
                continue

            pitch_std = np.std(segment_pitch[voiced])
            energy_std = np.std(segment_energy)

            if pitch_std < 5 and energy_std < 0.05:
                start_time = i * 0.01
                end_time = (i + window) * 0.01
                confidence = 1.0 - (pitch_std / 10 + energy_std / 0.1) / 2
                if confidence >= self.prolongation_threshold:
                    prolongations.append(
                        StutterEvent(
                            stutter_type=StutterType.PROLONGATION,
                            start_time=start_time,
                            end_time=end_time,
                            confidence=confidence,
                            severity=min(1.0, confidence**2),
                        )
                    )
                    i += window
        return prolongations

    def _detect_blocks(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect blocks by identifying abrupt drops in energy and low zero-crossing rate.
        """
        blocks = []
        energy = features.energy
        zcr = features.zero_crossing_rate
        n_frames = len(energy)
        window = 10

        for i in range(0, n_frames - window):
            segment_energy = energy[i : i + window]
            segment_zcr = zcr[i : i + window]

            if np.mean(segment_energy) < 0.1 and np.std(segment_energy) < 0.02:
                if np.mean(segment_zcr) < 0.05:
                    start_time = i * 0.01
                    end_time = (i + window) * 0.01
                    confidence = 1.0 - np.mean(segment_energy) / 0.1
                    if confidence >= self.block_threshold:
                        blocks.append(
                            StutterEvent(
                                stutter_type=StutterType.BLOCK,
                                start_time=start_time,
                                end_time=end_time,
                                confidence=confidence,
                                severity=min(1.0, confidence**2),
                            )
                        )
                        i += window
        return blocks

    def _detect_prolonged_sounds(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect any prolonged sound by analyzing high-resolution MFCC features.
        For each short time window (with a 5ms hop length), compute the variance of MFCCs.
        If the variance remains below a threshold for a duration longer than 300ms, flag it.
        """
        prolonged_events = []
        # Assume MFCCs are recomputed with a higher time resolution (hop_length=5ms)
        mfcc = features.mfcc
        # If original MFCC was computed with a 10ms hop, here we simulate higher resolution analysis.
        # Adjust these parameters as needed.
        n_frames = mfcc.shape[1]
        hop_time = 0.005  # 5ms per frame
        window = int(0.3 / hop_time)  # 300ms window

        for i in range(0, n_frames - window):
            segment = mfcc[:, i : i + window]
            if np.std(segment) < 0.5:  # Threshold tuned for flat spectral regions
                start_time = i * hop_time
                end_time = (i + window) * hop_time
                prolonged_events.append(
                    StutterEvent(
                        stutter_type=StutterType.PROLONGATION,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=0.95,
                        severity=0.95,
                    )
                )
                i += window
        return prolonged_events
