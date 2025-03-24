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
       # Detection thresholds
       self.repetition_threshold = 0.8
       self.prolongation_threshold = 0.7
       self.block_threshold = 0.75
       
       # Analysis windows
       self.min_duration = 0.1  # seconds
       self.max_gap = 0.2      # seconds
       
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
           
           # Combine all detected events
           stutter_events.extend(repetitions)
           stutter_events.extend(prolongations)
           stutter_events.extend(blocks)
           
           # Sort events by start time
           stutter_events.sort(key=lambda x: x.start_time)
           
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
       for i in range(len(mfcc[0]) - 1):
           similarity = self._pattern_similarity(
               mfcc[:, i:i+10],
               mfcc[:, i+10:i+20]
           )
           
           if similarity > self.repetition_threshold:
               # Convert frame index to time
               start_time = i * 0.01  # assuming 10ms hop length
               end_time = (i + 20) * 0.01
               
               event = StutterEvent(
                   stutter_type=StutterType.REPETITION,
                   start_time=start_time,
                   end_time=end_time,
                   confidence=similarity,
                   severity=self._calculate_severity(similarity)
               )
               repetitions.append(event)
       
       return repetitions

   def _detect_prolongations(self, features: SpeechFeatures) -> List[StutterEvent]:
       """
       Detect sound prolongations.
       
       Args:
           features (SpeechFeatures): Speech features
           
       Returns:
           List[StutterEvent]: Detected prolongation events
       """
       prolongations = []
       
       # Analyze pitch and energy stability for prolongations
       pitch = features.pitch
       energy = features.energy
       
       for i in range(len(energy) - 1):
           if self._is_prolongation(pitch[i:i+20], energy[i:i+20]):
               start_time = i * 0.01
               end_time = (i + 20) * 0.01
               confidence = self._calculate_prolongation_confidence(
                   pitch[i:i+20],
                   energy[i:i+20]
               )
               
               event = StutterEvent(
                   stutter_type=StutterType.PROLONGATION,
                   start_time=start_time,
                   end_time=end_time,
                   confidence=confidence,
                   severity=self._calculate_severity(confidence)
               )
               prolongations.append(event)
       
       return prolongations

   def _detect_blocks(self, features: SpeechFeatures) -> List[StutterEvent]:
       """
       Detect speech blocks.
       
       Args:
           features (SpeechFeatures): Speech features
           
       Returns:
           List[StutterEvent]: Detected block events
       """
       blocks = []
       
       # Analyze energy drops and zero-crossing rates for blocks
       energy = features.energy
       zcr = features.zero_crossing_rate
       
       for i in range(len(energy) - 1):
           if self._is_block(energy[i:i+10], zcr[i:i+10]):
               start_time = i * 0.01
               end_time = (i + 10) * 0.01
               confidence = self._calculate_block_confidence(
                   energy[i:i+10],
                   zcr[i:i+10]
               )
               
               event = StutterEvent(
                   stutter_type=StutterType.BLOCK,
                   start_time=start_time,
                   end_time=end_time,
                   confidence=confidence,
                   severity=self._calculate_severity(confidence)
               )
               blocks.append(event)
       
       return blocks

   def _pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
       """Calculate similarity between two patterns."""
       try:
           correlation = np.corrcoef(pattern1.flat, pattern2.flat)[0,1]
           return max(0, correlation)
       except:
           return 0.0

   def _is_prolongation(self, pitch: np.ndarray, energy: np.ndarray) -> bool:
       """Check if segment contains a prolongation."""
       pitch_stability = np.std(pitch) < 10  # Hz
       energy_stability = np.std(energy) < 0.1
       return pitch_stability and energy_stability

   def _is_block(self, energy: np.ndarray, zcr: np.ndarray) -> bool:
       """Check if segment contains a block."""
       energy_drop = np.mean(energy) < 0.1
       low_zcr = np.mean(zcr) < 0.1
       return energy_drop and low_zcr

   def _calculate_severity(self, confidence: float) -> float:
       """Calculate severity score from confidence value."""
       # Simple linear mapping, can be made more sophisticated
       return min(1.0, confidence * 1.2)

   def _calculate_prolongation_confidence(self, pitch: np.ndarray, 
                                       energy: np.ndarray) -> float:
       """Calculate confidence score for prolongation detection."""
       pitch_stability = 1.0 - min(1.0, np.std(pitch) / 10)
       energy_stability = 1.0 - min(1.0, np.std(energy) / 0.1)
       return (pitch_stability + energy_stability) / 2

   def _calculate_block_confidence(self, energy: np.ndarray, 
                                 zcr: np.ndarray) -> float:
       """Calculate confidence score for block detection."""
       energy_score = 1.0 - min(1.0, np.mean(energy) / 0.1)
       zcr_score = 1.0 - min(1.0, np.mean(zcr) / 0.1)
       return (energy_score + zcr_score) / 2

   def generate_analysis_report(self, events: List[StutterEvent]) -> Dict:
       """
       Generate detailed analysis report from detected events.
       
       Args:
           events (List[StutterEvent]): Detected stutter events
           
       Returns:
           Dict: Analysis report
       """
       total_duration = sum(event.duration() for event in events)
       
       report = {
           'total_events': len(events),
           'total_duration': total_duration,
           'events_by_type': {
               stype: len([e for e in events if e.stutter_type == stype])
               for stype in StutterType
           },
           'average_severity': np.mean([e.severity for e in events])
           if events else 0.0,
           'average_confidence': np.mean([e.confidence for e in events])
           if events else 0.0
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
           speech_rate=4.0
       )
       
       # Detect stutters
       events = detector.analyze_speech(sample_features)
       
       # Generate report
       report = detector.generate_analysis_report(events)
       print("Analysis Report:", report)
       
   except Exception as e:
       print(f"Error in stutter detection: {e}")