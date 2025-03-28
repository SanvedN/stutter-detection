"""
stutter_detector.py

Core stutter detection module for analyzing speech patterns and identifying stutters.
Uses extracted features to detect and classify different types of stutters.

Key Features:
   - Multiple stutter type detection
   - Pattern analysis
   - Confidence scoring
   - Detailed reporting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import librosa
import re

from src.audio.feature_extractor import SpeechFeatures

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StutterType(Enum):
    """Types of stutters that can be detected."""

    REPETITION = "repetition"  # Sound, syllable, or word repetitions
    PROLONGATION = "prolongation"  # Sound prolongations
    BLOCK = "block"  # Blocks or stops in speech
    INTERJECTION = "interjection"  # Filler words or sounds


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
        text (str): The text associated with the stutter
    """

    stutter_type: StutterType
    start_time: float
    end_time: float
    confidence: float
    severity: float
    text: str = ""

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
        # Detection thresholds
        self.repetition_threshold = 0.75
        self.prolongation_threshold = 0.65
        self.block_threshold = 0.7

        # Analysis windows
        self.min_duration = 0.1  # seconds
        self.max_gap = 0.2  # seconds

        # Prolongation detection parameters
        self.prolongation_min_duration = 0.15  # seconds
        self.prolongation_stability_threshold = 0.05  # Lower values mean more stable

        # Patterns for detecting prolonged sounds
        self.prolongation_patterns = [
            r"([a-z])\1{2,}",  # Repeated letters (e.g., "sssshort")
            r"([a-z]{1,2})-\1{1,2}(-\1{1,2})*",  # Repeated syllables (e.g., "sh-sh-short")
        ]

        # Compile patterns
        self.prolongation_regex = re.compile("|".join(self.prolongation_patterns))

        logger.info("StutterDetector initialized with enhanced detection parameters")

    def analyze_speech(
        self, features: SpeechFeatures, audio_data: np.ndarray, sample_rate: int
    ) -> List[StutterEvent]:
        """
        Analyze speech features to detect stutters.

        Args:
            features (SpeechFeatures): Extracted speech features
            audio_data (np.ndarray): Raw audio data for additional analysis
            sample_rate (int): Audio sample rate

        Returns:
            List[StutterEvent]: List of detected stutter events
        """
        try:
            stutter_events = []

            # Detect different types of stutters
            repetitions = self._detect_repetitions(features)
            prolongations = self._detect_prolongations(
                features, audio_data, sample_rate
            )
            blocks = self._detect_blocks(features)

            # Combine all detected events
            stutter_events.extend(repetitions)
            stutter_events.extend(prolongations)
            stutter_events.extend(blocks)

            # Sort events by start time
            stutter_events.sort(key=lambda x: x.start_time)

            # Merge overlapping events
            stutter_events = self._merge_overlapping_events(stutter_events)

            logger.info(f"Detected {len(stutter_events)} stutter events")
            return stutter_events

        except Exception as e:
            logger.error(f"Error in stutter analysis: {e}")
            raise

    def _detect_repetitions(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect sound, syllable, or word repetitions.

        Args:
            features (SpeechFeatures): Speech features

        Returns:
            List[StutterEvent]: Detected repetition events
        """
        repetitions = []

        # Analyze MFCC patterns for repetitions
        mfcc = features.mfcc
        frame_length = int(0.1 * 100)  # 100ms window at 10ms hop

        for i in range(0, len(mfcc[0]) - frame_length, int(frame_length / 2)):
            # Look for repetitive patterns in consecutive frames
            if i + 2 * frame_length >= len(mfcc[0]):
                break

            # Compare consecutive windows
            window1 = mfcc[:, i : i + frame_length]
            window2 = mfcc[:, i + frame_length : i + 2 * frame_length]

            similarity = self._pattern_similarity(window1, window2)

            if similarity > self.repetition_threshold:
                # Convert frame index to time
                start_time = i * 0.01  # assuming 10ms hop length
                end_time = (i + 2 * frame_length) * 0.01

                # Calculate severity based on similarity and duration
                severity = similarity * min(1.0, (end_time - start_time) / 0.5)

                event = StutterEvent(
                    stutter_type=StutterType.REPETITION,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=similarity,
                    severity=severity,
                    text="repetition",
                )
                repetitions.append(event)

        return repetitions

    def _detect_prolongations(
        self, features: SpeechFeatures, audio_data: np.ndarray, sample_rate: int
    ) -> List[StutterEvent]:
        """
        Detect sound prolongations with enhanced accuracy.

        Args:
            features (SpeechFeatures): Speech features
            audio_data (np.ndarray): Raw audio data
            sample_rate (int): Audio sample rate

        Returns:
            List[StutterEvent]: Detected prolongation events
        """
        prolongations = []

        # 1. Analyze pitch and energy stability for prolongations
        pitch = features.pitch
        energy = features.energy

        # Window size for prolongation detection (200ms)
        window_size = int(0.2 / 0.01)  # 0.01s is the hop length

        for i in range(0, len(energy) - window_size, int(window_size / 4)):
            pitch_window = pitch[i : i + window_size]
            energy_window = energy[i : i + window_size]

            # Check for stable pitch and energy (characteristic of prolongations)
            if self._is_prolongation(pitch_window, energy_window):
                start_time = i * 0.01
                end_time = (i + window_size) * 0.01
                confidence = self._calculate_prolongation_confidence(
                    pitch_window, energy_window
                )

                # Calculate severity based on duration and stability
                severity = confidence * min(1.0, (end_time - start_time) / 0.3)

                event = StutterEvent(
                    stutter_type=StutterType.PROLONGATION,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    severity=severity,
                    text="prolongation",
                )
                prolongations.append(event)

        # 2. Analyze spectral characteristics for specific prolonged sounds
        # This detects sounds like "ssss" or "ffff" that have specific spectral signatures

        # Calculate spectrogram
        D = librosa.stft(audio_data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Parameters for detection
        frame_length = int(0.025 * sample_rate)  # 25ms
        hop_length = int(0.010 * sample_rate)  # 10ms

        # Detect specific prolonged sounds (fricatives like 's', 'sh', 'f')
        for i in range(0, S_db.shape[1] - 10):
            # Check for high energy in high frequencies (characteristic of fricatives)
            high_freq_energy = np.mean(S_db[int(S_db.shape[0] / 2) :, i : i + 10])
            low_freq_energy = np.mean(S_db[: int(S_db.shape[0] / 2), i : i + 10])

            # Fricatives have more energy in high frequencies
            if high_freq_energy > low_freq_energy and high_freq_energy > -30:
                # Check for stability over time (prolongation)
                stability = 1.0 - np.std(S_db[:, i : i + 10]) / 20.0  # Normalize

                if stability > 0.7:  # High stability threshold
                    start_time = i * hop_length / sample_rate
                    end_time = (i + 10) * hop_length / sample_rate

                    # Only consider if duration is sufficient
                    if end_time - start_time >= self.prolongation_min_duration:
                        event = StutterEvent(
                            stutter_type=StutterType.PROLONGATION,
                            start_time=start_time,
                            end_time=end_time,
                            confidence=stability,
                            severity=stability * 0.8,  # Slightly lower severity
                            text="fricative_prolongation",
                        )
                        prolongations.append(event)

        return prolongations

    def _detect_blocks(self, features: SpeechFeatures) -> List[StutterEvent]:
        """
        Detect speech blocks (silent struggles).

        Args:
            features (SpeechFeatures): Speech features

        Returns:
            List[StutterEvent]: Detected block events
        """
        blocks = []

        # Analyze energy drops and zero-crossing rates for blocks
        energy = features.energy
        zcr = features.zero_crossing_rate

        # Window size for block detection (300ms)
        window_size = int(0.3 / 0.01)  # 0.01s is the hop length

        for i in range(0, len(energy) - window_size, int(window_size / 2)):
            energy_window = energy[i : i + window_size]
            zcr_window = zcr[i : i + window_size]

            # Check for blocks: low energy but some activity in ZCR
            # This indicates struggle rather than simple silence
            if self._is_block(energy_window, zcr_window):
                start_time = i * 0.01
                end_time = (i + window_size) * 0.01
                confidence = self._calculate_block_confidence(energy_window, zcr_window)

                # Calculate severity based on duration and confidence
                severity = confidence * min(1.0, (end_time - start_time) / 0.5)

                event = StutterEvent(
                    stutter_type=StutterType.BLOCK,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    severity=severity,
                    text="block",
                )
                blocks.append(event)

        return blocks

    def _pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns."""
        try:
            # Reshape if needed
            p1_flat = pattern1.flatten()
            p2_flat = pattern2.flatten()

            # Calculate correlation
            correlation = np.corrcoef(p1_flat, p2_flat)[0, 1]
            return max(0, correlation)
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0

    def _is_prolongation(self, pitch: np.ndarray, energy: np.ndarray) -> bool:
        """
        Check if segment contains a prolongation.

        Prolongations are characterized by stable pitch and energy.
        """
        if len(pitch) < 3 or len(energy) < 3:
            return False

        # Remove zeros and NaNs
        valid_pitch = pitch[~np.isnan(pitch) & (pitch > 0)]
        if len(valid_pitch) < 3:
            return False

        # Calculate stability metrics
        pitch_stability = np.std(valid_pitch) / (np.mean(valid_pitch) + 1e-10)
        energy_stability = np.std(energy) / (np.mean(energy) + 1e-10)

        # Prolongations have stable pitch and energy
        return (
            pitch_stability < self.prolongation_stability_threshold
            and energy_stability < self.prolongation_stability_threshold
            and np.mean(energy) > 0.05
        )  # Ensure there's actual sound

    def _is_block(self, energy: np.ndarray, zcr: np.ndarray) -> bool:
        """
        Check if segment contains a block.

        Blocks are characterized by low energy but some articulatory movement (ZCR).
        """
        # Calculate metrics
        mean_energy = np.mean(energy)
        mean_zcr = np.mean(zcr)

        # Blocks have very low energy but some ZCR activity
        # This differentiates blocks from silence (low energy, low ZCR)
        return (
            mean_energy < 0.1  # Low energy
            and mean_zcr > 0.1  # Some ZCR activity
            and mean_zcr < 0.5
        )  # But not too high (which would be fricatives)

    def _calculate_severity(self, confidence: float, duration: float) -> float:
        """Calculate severity score from confidence and duration."""
        # Severity increases with confidence and duration
        base_severity = min(1.0, confidence * 1.2)
        duration_factor = min(1.0, duration / 0.5)  # Normalize to 0.5s

        return base_severity * (0.7 + 0.3 * duration_factor)

    def _calculate_prolongation_confidence(
        self, pitch: np.ndarray, energy: np.ndarray
    ) -> float:
        """Calculate confidence score for prolongation detection."""
        # Remove zeros and NaNs from pitch
        valid_pitch = pitch[~np.isnan(pitch) & (pitch > 0)]
        if len(valid_pitch) < 3:
            return 0.0

        # Calculate stability metrics
        pitch_stability = 1.0 - min(
            1.0, np.std(valid_pitch) / (np.mean(valid_pitch) + 1e-10) / 0.2
        )
        energy_stability = 1.0 - min(
            1.0, np.std(energy) / (np.mean(energy) + 1e-10) / 0.2
        )

        # Combine metrics with higher weight on pitch stability
        return pitch_stability * 0.7 + energy_stability * 0.3

    def _calculate_block_confidence(self, energy: np.ndarray, zcr: np.ndarray) -> float:
        """Calculate confidence score for block detection."""
        # Calculate metrics
        mean_energy = np.mean(energy)
        mean_zcr = np.mean(zcr)

        # Confidence based on how well the metrics match block characteristics
        energy_score = 1.0 - min(1.0, mean_energy / 0.1)
        zcr_score = max(0.0, min(1.0, (mean_zcr - 0.1) / 0.4))

        return energy_score * 0.7 + zcr_score * 0.3

    def _merge_overlapping_events(
        self, events: List[StutterEvent]
    ) -> List[StutterEvent]:
        """Merge overlapping stutter events of the same type."""
        if not events:
            return []

        # Sort by start time
        sorted_events = sorted(events, key=lambda e: e.start_time)
        merged = [sorted_events[0]]

        for event in sorted_events[1:]:
            prev = merged[-1]

            # Check if events overlap and are of the same type
            if (
                event.start_time <= prev.end_time
                and event.stutter_type == prev.stutter_type
            ):
                # Merge events
                prev.end_time = max(prev.end_time, event.end_time)
                prev.confidence = max(prev.confidence, event.confidence)
                prev.severity = max(prev.severity, event.severity)
            else:
                merged.append(event)

        return merged

    def analyze_transcription(self, transcription: str) -> List[Dict]:
        """
        Analyze transcription text for stutter patterns.

        Args:
            transcription (str): Transcribed text

        Returns:
            List[Dict]: Detected stutter patterns in text
        """
        results = []

        # 1. Check for prolonged sounds (e.g., "ssssshort")
        for match in self.prolongation_regex.finditer(transcription.lower()):
            results.append(
                {
                    "text": match.group(0),
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "stutter_type": "prolongation",
                }
            )

        # 2. Check for part-word repetitions
        words = transcription.lower().split()
        for i, word in enumerate(words):
            # Check for hyphenated repetitions (e.g., "w-w-word")
            if "-" in word:
                parts = word.split("-")
                if len(parts) >= 2:
                    # Check if parts are repetitive
                    if len(set(parts[:-1])) == 1:
                        results.append(
                            {
                                "text": word,
                                "word_position": i,
                                "stutter_type": "part_word_repetition",
                            }
                        )

        # 3. Check for word repetitions in sequence
        i = 0
        while i < len(words) - 1:
            j = i + 1
            repetition_count = 1

            while j < len(words) and words[j] == words[i]:
                repetition_count += 1
                j += 1

            if repetition_count > 1:
                results.append(
                    {
                        "text": " ".join([words[i]] * repetition_count),
                        "word_position": i,
                        "repetition_count": repetition_count,
                        "stutter_type": "word_repetition",
                    }
                )
                i = j
            else:
                i += 1

        return results

    def generate_analysis_report(self, events: List[StutterEvent]) -> Dict:
        """
        Generate detailed analysis report from detected events.

        Args:
            events (List[StutterEvent]): Detected stutter events

        Returns:
            Dict: Analysis report
        """
        if not events:
            return {
                "total_events": 0,
                "total_duration": 0,
                "events_by_type": {stype.value: 0 for stype in StutterType},
                "average_severity": 0.0,
                "average_confidence": 0.0,
                "stutter_rate": 0.0,
            }

        total_duration = sum(event.duration() for event in events)
        speech_duration = max(event.end_time for event in events) - min(
            event.start_time for event in events
        )

        # Count events by type
        events_by_type = {}
        for stype in StutterType:
            type_events = [e for e in events if e.stutter_type == stype]
            events_by_type[stype.value] = {
                "count": len(type_events),
                "total_duration": sum(e.duration() for e in type_events),
                "average_severity": (
                    np.mean([e.severity for e in type_events]) if type_events else 0.0
                ),
            }

        report = {
            "total_events": len(events),
            "total_duration": total_duration,
            "speech_duration": speech_duration,
            "events_by_type": events_by_type,
            "average_severity": np.mean([e.severity for e in events]),
            "average_confidence": np.mean([e.confidence for e in events]),
            "stutter_rate": (
                len(events) / (speech_duration / 60) if speech_duration > 0 else 0
            ),  # Events per minute
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
            pitch=np.random.randn(1000),
            energy=np.abs(np.random.randn(1000)),
            zero_crossing_rate=np.abs(np.random.randn(1000)),
            duration=10.0,
            speech_rate=4.0,
        )

        # Sample audio data
        sample_audio = np.random.randn(16000)
        sample_rate = 16000

        # Detect stutters
        events = detector.analyze_speech(sample_features, sample_audio, sample_rate)

        # Generate report
        report = detector.generate_analysis_report(events)
        print("Analysis Report:", report)

    except Exception as e:
        print(f"Error in stutter detection: {e}")
