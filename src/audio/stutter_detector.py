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
from typing import Dict, List, Optional
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
        # Detection thresholds calibrated on development data:
        self.repetition_threshold = 0.85  # Higher threshold to reduce false positives
        self.prolongation_threshold = 0.75
        self.block_threshold = 0.70

        # Analysis window parameters
        self.window_size_repetition = 10  # Frames to compare
        self.step_size = 1  # Frame step for sliding window

        logger.info("StutterDetector initialized")

    def analyze_speech(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Analyze speech features to detect stutters.

        Args:
            features (SpeechFeatures): Extracted speech features

        Returns:
            List[StutterEvent]: List of detected stutter events
        """
        try:
            stutter_events = []

            # Detect different types of stutters
            repetitions = self._detect_repetitions(features)
            prolongations = self._detect_prolongations(features)
            blocks = self._detect_blocks(features)

            # Combine all detected events and sort by start time
            stutter_events.extend(repetitions)
            stutter_events.extend(prolongations)
            stutter_events.extend(blocks)
            stutter_events.sort(key=lambda x: x.start_time)

            logger.info(f"Detected {len(stutter_events)} stutter events")
            return stutter_events

        except Exception as e:
            logger.error(f"Error in stutter analysis: {e}")
            raise

    def _detect_repetitions(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect repetitions using a sliding window on MFCC features.

        Args:
            features (SpeechFeatures): Speech features

        Returns:
            List[StutterEvent]: Detected repetition events
        """
        repetitions = []
        mfcc = features.mfcc  # Shape: (features, frames)
        n_frames = mfcc.shape[1]

        # Slide over frames with a dynamic window comparison
        for i in range(0, n_frames - 2 * self.window_size_repetition, self.step_size):
            segment1 = mfcc[:, i : i + self.window_size_repetition]
            segment2 = mfcc[
                :, i + self.window_size_repetition : i + 2 * self.window_size_repetition
            ]

            # Compute normalized cross-correlation between flattened segments
            corr = np.corrcoef(segment1.flatten(), segment2.flatten())[0, 1]
            corr = max(0.0, corr)  # Ensure non-negative

            if corr >= self.repetition_threshold:
                start_time = i * 0.01  # Assuming 10ms hop
                end_time = (i + 2 * self.window_size_repetition) * 0.01
                event = StutterEvent(
                    stutter_type=StutterType.REPETITION,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=corr,
                    severity=self._calculate_severity(corr),
                )
                repetitions.append(event)
                # Skip ahead to avoid overlapping detections
                i += 2 * self.window_size_repetition

        return repetitions

    def _detect_prolongations(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect prolongations by evaluating stability in pitch and energy over time.

        Args:
            features (SpeechFeatures): Speech features

        Returns:
            List[StutterEvent]: Detected prolongation events
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
            if np.sum(voiced) < window * 0.8:  # Require most frames to be voiced
                continue

            # Stability: low standard deviation in pitch and energy
            pitch_std = np.std(segment_pitch[voiced])
            energy_std = np.std(segment_energy)

            if pitch_std < 5 and energy_std < 0.05:
                start_time = i * 0.01
                end_time = (i + window) * 0.01
                # Confidence based on how stable the values are
                confidence = 1.0 - (pitch_std / 10 + energy_std / 0.1) / 2
                event = StutterEvent(
                    stutter_type=StutterType.PROLONGATION,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    severity=self._calculate_severity(confidence),
                )
                prolongations.append(event)
                i += window  # Skip ahead to avoid overlapping

        return prolongations

    def _detect_blocks(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect blocks by identifying abrupt drops in energy along with low zero-crossing rate.

        Args:
            features (SpeechFeatures): Speech features

        Returns:
            List[StutterEvent]: Detected block events
        """
        blocks = []
        energy = features.energy
        zcr = features.zero_crossing_rate
        n_frames = len(energy)
        window = 10  # Short window to capture blocks

        for i in range(0, n_frames - window):
            segment_energy = energy[i : i + window]
            segment_zcr = zcr[i : i + window]

            # A block is characterized by a significant drop in energy and low variability
            if np.mean(segment_energy) < 0.1 and np.std(segment_energy) < 0.02:
                # And low zero-crossing rate indicating minimal speech movement
                if np.mean(segment_zcr) < 0.05:
                    start_time = i * 0.01
                    end_time = (i + window) * 0.01
                    # Confidence can be a function of how low the energy is compared to overall energy
                    confidence = 1.0 - np.mean(segment_energy) / 0.1
                    event = StutterEvent(
                        stutter_type=StutterType.BLOCK,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=confidence,
                        severity=self._calculate_severity(confidence),
                    )
                    blocks.append(event)
                    i += window  # Skip ahead

        return blocks

    def _calculate_severity(self, confidence: float) -> float:
        """
        Calculate severity score from confidence value using a nonlinear mapping.

        Args:
            confidence (float): Confidence score (0-1)

        Returns:
            float: Estimated severity (0-1)
        """
        # Example: quadratic mapping to emphasize high confidence events
        return min(1.0, confidence**2)

    def generate_analysis_report(self, events: List[StutterEvent]) -> Dict:
        """
        Generate detailed analysis report from detected events.

        Args:
            events (List[StutterEvent]): Detected stutter events

        Returns:
            Dict: Analysis report with statistics
        """
        total_duration = sum(event.duration() for event in events)
        report = {
            "total_events": len(events),
            "total_duration": total_duration,
            "events_by_type": {
                stype.value: len([e for e in events if e.stutter_type == stype])
                for stype in StutterType
            },
            "average_severity": (
                np.mean([e.severity for e in events]) if events else 0.0
            ),
            "average_confidence": (
                np.mean([e.confidence for e in events]) if events else 0.0
            ),
        }
        return report


# Example usage
if __name__ == "__main__":
    try:
        # Create detector
        detector = StutterDetector()

        # Generate sample features (replace with real features)
        sample_features = SpeechFeatures(
            mfcc=np.random.randn(39, 1000),
            pitch=np.abs(np.random.randn(1000)) * 100,  # Simulated pitch in Hz
            energy=np.abs(np.random.randn(1000)) * 0.1,
            zero_crossing_rate=np.abs(np.random.randn(1000)) * 0.1,
            duration=10.0,
            speech_rate=4.0,
        )

        # Detect stutters
        events = detector.analyze_speech(sample_features)

        # Generate report
        report = detector.generate_analysis_report(events)
        print("Analysis Report:", report)

    except Exception as e:
        print(f"Error in stutter detection: {e}")
