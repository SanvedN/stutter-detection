"""
transcription_analyzer.py

Enhanced speech transcription and analysis module with high accuracy detection.
Handles transcription, filler detection, and exports in multiple formats.

Features:
   - High-accuracy speech-to-text conversion
   - Advanced stutter detection (repetitions, prolongations, blocks)
   - Multiple export formats (TXT, VTT, TextGrid, JSON)
   - Comprehensive analysis reports
   - Reference text comparison
"""

import whisper
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import torch
import re
from datetime import datetime
import tgt
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
import string
import Levenshtein
import librosa

# Download required NLTK data
try:
    nltk.download("punkt")
    nltk.download("words")
except Exception as e:
    logging.warning(f"Failed to download NLTK data: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Grandfather's passage for reference
GRANDFATHERS_PASSAGE = """You wish to know about my grandfather. Well, he is nearly 93 years old, yet he still thinks as swiftly as ever. He dresses himself in an old black frock coat, usually several buttons missing. A long beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. When he speaks, his voice is just a bit cracked and quivers a bit. Twice each day he plays skillfully and with zest upon a small organ. Except in the winter when the snow or ice prevents, he slowly takes a short walk in the open air each day. We have often urged him to walk more and smoke less, but he always answers, “Banana oil!”. Grandfather likes to be modern in his language."""


@dataclass
class TranscriptionResult:
    """Container for transcription analysis results."""

    text: str
    segments: List[Dict]
    word_timings: List[Dict]
    fillers: List[Dict]
    repetitions: List[Dict]
    pronunciation_errors: List[Dict]
    confidence: float
    duration: float
    speech_rate: float
    language_score: float
    silences: List[Dict]


class TranscriptionAnalyzer:
    def __init__(self, model_size: str = "large"):
        """
        Initialize with high-accuracy speech recognition.

        Args:
            model_size: Whisper model size (recommended: "large" for best accuracy)
        """
        try:
            # Load Whisper model
            self.model = whisper.load_model(model_size)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            # Comprehensive filler and speech disfluency patterns
            self.speech_patterns = {
                "hesitation": {
                    "single": ["uh", "um", "er", "ah", "eh", "hm", "hmm", "erm"],
                    "compound": ["uh uh", "um um", "er er", "ah ah"],
                },
                "discourse": {
                    "single": ["like", "well", "so", "right", "okay", "see"],
                    "compound": ["you know", "i mean", "kind of", "sort of", "you see"],
                },
                "pause_fillers": ["mm", "uh-huh", "mhm", "yeah"],
                "starters": [
                    "basically",
                    "actually",
                    "literally",
                    "obviously",
                    "frankly",
                ],
                "repetition_markers": ["th-th", "st-st", "wh-wh", "b-b"],
            }

            # Compile regex patterns
            self._compile_patterns()

            # Prepare reference text
            self.reference_words = word_tokenize(GRANDFATHERS_PASSAGE.lower())

            logger.info(
                f"TranscriptionAnalyzer initialized with {model_size} model on {self.device}"
            )

        except Exception as e:
            logger.error(f"Error initializing TranscriptionAnalyzer: {e}")
            raise

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.pattern_regexes = {}
        for category, patterns in self.speech_patterns.items():
            if isinstance(patterns, dict):
                self.pattern_regexes[category] = {
                    "single": re.compile(
                        r"\b(" + "|".join(patterns["single"]) + r")\b", re.IGNORECASE
                    ),
                    "compound": re.compile(
                        r"\b(" + "|".join(patterns["compound"]) + r")\b", re.IGNORECASE
                    ),
                }
            else:
                self.pattern_regexes[category] = re.compile(
                    r"\b(" + "|".join(patterns) + r")\b", re.IGNORECASE
                )

    def analyze_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_dir: Path
    ) -> TranscriptionResult:
        """
        Perform complete audio analysis with enhanced accuracy.

        Args:
            audio_data: Audio signal
            sample_rate: Audio sample rate
            output_dir: Directory for output files

        Returns:
            TranscriptionResult containing detailed analysis
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Perform enhanced transcription
            result = self.transcribe_with_enhanced_detection(audio_data, sample_rate)

            # Save results in multiple formats
            # self.save_all_formats(result, output_dir)

            return result

        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            raise

    def transcribe_with_enhanced_detection(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> TranscriptionResult:
        """Perform high-accuracy transcription with enhanced detection."""
        try:
            # Initial transcription with optimized settings
            result = self.model.transcribe(
                audio_data,
                language="en",
                word_timestamps=True,
                condition_on_previous_text=True,
                initial_prompt="Include hesitations, fillers, repetitions, and partial words exactly as spoken. This is the Grandfather's Passage.",
                temperature=0.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                beam_size=5,
            )

            # Enhanced processing
            segments = self._post_process_segments(result["segments"])
            word_timings = self._extract_enhanced_word_timings(segments)
            fillers = self._detect_fillers_with_context(word_timings)
            repetitions = self._enhanced_repetition_detection(word_timings)
            pronunciation_errors = self._detect_pronunciation_errors(word_timings)
            silences = self._detect_silences(audio_data, sample_rate, segments)

            # Calculate metrics
            confidence = np.mean([segment.get("confidence", 0) for segment in segments])
            duration = segments[-1]["end"] if segments else 0
            speech_rate = self._calculate_speech_rate(word_timings, duration)
            language_score = self._calculate_language_score(result["text"])

            return TranscriptionResult(
                text=result["text"],
                segments=segments,
                word_timings=word_timings,
                fillers=fillers,
                repetitions=repetitions,
                pronunciation_errors=pronunciation_errors,
                confidence=confidence,
                duration=duration,
                speech_rate=speech_rate,
                language_score=language_score,
                silences=silences,
            )

        except Exception as e:
            logger.error(f"Error in enhanced transcription: {e}")
            raise

    def _extract_enhanced_word_timings(self, segments: List[Dict]) -> List[Dict]:
        """Extract detailed word timings with confidence scores."""
        word_timings = []

        for segment in segments:
            words = segment.get("words", [])
            for i, word_info in enumerate(words):
                if not isinstance(word_info, dict):
                    continue

                word = word_info.get("word", "").strip().lower()
                if not word:
                    continue

                # Check for partial words and stutters
                is_partial = bool(re.search(r"-", word))
                is_stutter = bool(re.search(r"([a-z])\1+", word))

                timing = {
                    "word": word,
                    "start": word_info.get("start", segment["start"]),
                    "end": word_info.get("end", segment["end"]),
                    "confidence": word_info.get("confidence", 0.0),
                    "is_partial": is_partial,
                    "is_stutter": is_stutter,
                    "segment_id": segment["id"],
                }

                word_timings.append(timing)

        return word_timings

    def _detect_fillers_with_context(self, word_timings: List[Dict]) -> List[Dict]:
        """Detect filler words with context analysis."""
        fillers = []
        window_size = 3

        for i, word_info in enumerate(word_timings):
            word = word_info["word"].lower()

            # Get context window
            start_idx = max(0, i - window_size)
            end_idx = min(len(word_timings), i + window_size + 1)
            context = word_timings[start_idx:end_idx]

            # Check for single-word fillers
            for category, patterns in self.pattern_regexes.items():
                if isinstance(patterns, dict):
                    if patterns["single"].search(word):
                        if self._validate_filler_context(word, context):
                            fillers.append(
                                self._create_filler_entry(word_info, category, context)
                            )

                    # Check for compound fillers
                    if i < len(word_timings) - 1:
                        compound = f"{word} {word_timings[i+1]['word']}".lower()
                        if patterns["compound"].search(compound):
                            fillers.append(
                                self._create_filler_entry(
                                    word_info,
                                    category,
                                    context,
                                    end_time=word_timings[i + 1]["end"],
                                    compound=True,
                                )
                            )

        return fillers

    def _create_filler_entry(
        self,
        word_info: Dict,
        category: str,
        context: List[Dict],
        end_time: float = None,
        compound: bool = False,
    ) -> Dict:
        """Create standardized filler entry."""
        return {
            "word": word_info["word"],
            "start": word_info["start"],
            "end": end_time or word_info["end"],
            "event_type": "filler",  # Standardized event type
            "filler_type": category,  # Specific filler category
            "compound": compound,
            "confidence": word_info["confidence"],
            "context": " ".join(w["word"] for w in context),
        }

    def _enhanced_repetition_detection(self, word_timings: List[Dict]) -> List[Dict]:
        """Improved repetition detection focusing on short interval repetitions."""
        repetitions = []
        max_time_between_repetitions = 1.0  # Maximum 1 second between repetitions

        # Process words in sequence
        i = 0
        while i < len(word_timings) - 1:
            current_word = word_timings[i]["word"].lower().strip(string.punctuation)
            if (
                not current_word or len(current_word) < 2
            ):  # Skip empty or very short words
                i += 1
                continue

            # Look for repetitions in a short time window
            repetition_sequence = [
                {
                    "word": current_word,
                    "start": word_timings[i]["start"],
                    "end": word_timings[i]["end"],
                }
            ]
            j = i + 1

            while j < len(word_timings):
                next_word = word_timings[j]["word"].lower().strip(string.punctuation)
                if not next_word:
                    j += 1
                    continue

                time_gap = word_timings[j]["start"] - word_timings[j - 1]["end"]

                # Check if this is a repetition within the time threshold
                if (
                    self._is_repetition(current_word, next_word)
                    and time_gap <= max_time_between_repetitions
                ):
                    repetition_sequence.append(
                        {
                            "word": next_word,
                            "start": word_timings[j]["start"],
                            "end": word_timings[j]["end"],
                        }
                    )
                    j += 1
                else:
                    break

            # If we found repetitions (more than 1 occurrence)
            if len(repetition_sequence) > 1:
                repetitions.append(
                    {
                        "word": current_word,
                        "pattern": repetition_sequence,
                        "count": len(repetition_sequence),
                        "start": repetition_sequence[0]["start"],
                        "end": repetition_sequence[-1]["end"],
                        "event_type": "repetition",
                        "repetition_type": (
                            "continuous" if len(repetition_sequence) > 2 else "simple"
                        ),
                        "confidence": 0.9,  # High confidence for detected repetitions
                    }
                )
                i = j  # Skip the words we've already processed
            else:
                i += 1

        return repetitions

    def _is_repetition(self, word1: str, word2: str) -> bool:
        """Check if two words represent a repetition."""
        # Exact match
        if word1 == word2:
            return True

        # Check for partial word repetitions (e.g., "st-stutter")
        if len(word1) >= 2 and len(word2) >= 2:
            # Check if they share the same beginning
            if word1.startswith(word2[:2]) or word2.startswith(word1[:2]):
                # Check if one is a partial form of the other (e.g., "st-" and "stutter")
                if "-" in word1 or "-" in word2:
                    return True

                # Check if they're very similar
                similarity = Levenshtein.ratio(word1, word2)
                if similarity > 0.8:
                    return True

        return False

    def _detect_pronunciation_errors(self, word_timings: List[Dict]) -> List[Dict]:
        """Detect pronunciation errors and prolongations."""
        errors = []

        # Extract words from transcription
        transcribed_words = [
            w["word"].lower().strip(string.punctuation)
            for w in word_timings
            if w["word"].strip() and not self._is_filler(w["word"])
        ]

        # First pass: detect prolongations directly from word timings
        for i, word_info in enumerate(word_timings):
            word = word_info["word"].lower()

            # Check for prolonged sounds (repeated characters)
            if re.search(r"([a-z])\1{2,}", word):  # e.g., "sssshort" or "mmmmom"
                errors.append(
                    {
                        "word": word,
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "event_type": "prolongation",
                        "confidence": 0.9,
                        "severity": 0.7,
                    }
                )
                continue

            # Check for hyphenated prolongations (e.g., "s-s-sam")
            if "-" in word and len(word) >= 3:
                parts = word.split("-")
                if len(parts) >= 2 and len(set(parts[:-1])) == 1:
                    errors.append(
                        {
                            "word": word,
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "event_type": "prolongation",
                            "confidence": 0.9,
                            "severity": 0.8,
                        }
                    )
                    continue

        # Second pass: compare with reference text
        alignment = self._align_texts(transcribed_words, self.reference_words)

        for i, (trans_idx, ref_idx) in enumerate(alignment):
            if trans_idx is not None and ref_idx is not None:
                trans_word = transcribed_words[trans_idx]
                ref_word = self.reference_words[ref_idx]

                # Skip words already identified as prolongations
                if any(
                    e.get("word", "") == trans_word
                    and e.get("event_type") == "prolongation"
                    for e in errors
                ):
                    continue

                # Check for pronunciation errors
                if trans_word != ref_word:
                    # Calculate edit distance to determine severity
                    distance = Levenshtein.distance(trans_word, ref_word)
                    similarity = Levenshtein.ratio(trans_word, ref_word)

                    # Only report if words are somewhat similar but not identical
                    if 0.5 < similarity < 0.9:
                        word_info = word_timings[
                            self._find_word_timing_index(trans_word, word_timings)
                        ]
                        errors.append(
                            {
                                "word": trans_word,
                                "reference": ref_word,
                                "start": word_info["start"],
                                "end": word_info["end"],
                                "event_type": "pronunciation_error",
                                "confidence": 1.0 - similarity,
                                "severity": distance
                                / max(len(trans_word), len(ref_word)),
                            }
                        )

        return errors

    def _find_word_timing_index(self, word: str, word_timings: List[Dict]) -> int:
        """Find the index of a word in word_timings."""
        for i, timing in enumerate(word_timings):
            if timing["word"].lower().strip(string.punctuation) == word:
                return i
        return 0  # Default to first word if not found

    def _align_texts(
        self, transcribed: List[str], reference: List[str]
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """Align transcribed text with reference text using dynamic programming."""
        # Create a matrix of edit distances
        m, n = len(transcribed), len(reference)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if transcribed[i - 1] == reference[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1,  # substitution
                        dp[i - 1][j] + 1,  # deletion
                        dp[i][j - 1] + 1,
                    )  # insertion

        # Backtrack to find alignment
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and transcribed[i - 1] == reference[j - 1]:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                alignment.append((i - 1, j - 1))  # substitution
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                alignment.append((i - 1, None))  # deletion
                i -= 1
            else:
                alignment.append((None, j - 1))  # insertion
                j -= 1

        return list(reversed(alignment))

    def _detect_silences(
        self, audio_data: np.ndarray, sample_rate: int, segments: List[Dict]
    ) -> List[Dict]:
        """
        Detect silences with improved accuracy, distinguishing between natural pauses and blocks.

        Args:
            audio_data: Audio signal
            sample_rate: Audio sample rate
            segments: Transcription segments

        Returns:
            List of silence events with classification
        """
        silences = []

        try:
            # Parameters for silence detection - refined thresholds
            min_silence_duration = (
                0.3  # Minimum silence duration in seconds to consider
            )
            natural_pause_threshold = 0.6  # Threshold for natural pauses (seconds)
            block_threshold = 0.8  # Threshold for blocks (seconds)
            threshold_db = -40  # Silence threshold in dB

            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Get non-silent intervals using librosa's split function
            intervals = librosa.effects.split(
                audio_data,
                top_db=-threshold_db,
                frame_length=int(0.025 * sample_rate),
                hop_length=int(0.010 * sample_rate),
            )

            # Convert frame indices to seconds
            intervals_sec = [
                (start / sample_rate, end / sample_rate) for start, end in intervals
            ]

            # Get total audio duration
            total_duration = len(audio_data) / sample_rate

            # Find silences between speech segments
            if len(intervals_sec) > 1:
                for i in range(len(intervals_sec) - 1):
                    silence_start = intervals_sec[i][1]
                    silence_end = intervals_sec[i + 1][0]
                    silence_duration = silence_end - silence_start

                    # Skip very short silences
                    if silence_duration < min_silence_duration:
                        continue

                    # Skip silences at the very beginning or end
                    if silence_start < 0.3 or silence_end > total_duration - 0.3:
                        continue

                    # Analyze the silence
                    is_block = False
                    confidence = 0.5  # Default confidence

                    # Check if this is likely a block (longer than block threshold)
                    if silence_duration >= block_threshold:
                        # For very long silences, almost certainly a block
                        if silence_duration > 3:
                            is_block = True
                            confidence = 0.9
                        else:
                            # For borderline cases, analyze context
                            is_block = self._analyze_silence_context(
                                silence_start, silence_end, segments
                            )
                            confidence = min(
                                0.9, 0.7 + (silence_duration - block_threshold) * 0.5
                            )
                    # Check if this might be a block (in the gray area)
                    elif silence_duration >= natural_pause_threshold:
                        # Look at surrounding context to determine if this is a block
                        is_block = self._analyze_silence_context(
                            silence_start, silence_end, segments
                        )
                        confidence = (
                            0.5 + (silence_duration - natural_pause_threshold) * 0.5
                        )

                    # Add to results
                    silences.append(
                        {
                            "start": silence_start,
                            "end": silence_end,
                            "duration": silence_duration,
                            "event_type": "silence",
                            "position": "middle",
                            "is_block": is_block,
                            "confidence": confidence,
                        }
                    )

            return silences

        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
            return []

    def _analyze_silence_context(
        self, silence_start: float, silence_end: float, segments: List[Dict]
    ) -> bool:
        """
        Analyze the context around a silence to determine if it's likely a block.

        Args:
            silence_start: Start time of silence
            silence_end: End time of silence
            segments: Transcription segments

        Returns:
            True if likely a block, False if likely a natural pause
        """
        try:
            # Find segments before and after the silence
            segment_before = None
            segment_after = None

            for segment in segments:
                if abs(segment["end"] - silence_start) < 0.2:
                    segment_before = segment
                if abs(segment["start"] - silence_end) < 0.2:
                    segment_after = segment

            # If we can't find surrounding segments, default to not a block
            if not segment_before or not segment_after:
                return False

            # Check if the silence occurs in the middle of a sentence
            # (This would be unusual and might indicate a block)
            text_before = segment_before["text"].strip()
            text_after = segment_after["text"].strip()

            # Check if text before ends with sentence-ending punctuation
            ends_sentence = bool(re.search(r"[.!?]$", text_before))

            # Check if text after starts with a capital letter
            starts_sentence = bool(re.search(r"^[A-Z]", text_after))

            # If the silence doesn't align with sentence boundaries, it's more likely a block
            if not ends_sentence and not starts_sentence:
                return True

            # Check for grammatical continuity across the silence
            # If the text flows naturally across the silence, it's less likely to be a block
            combined_text = text_before + " " + text_after
            if self._is_grammatical_continuation(text_before, text_after):
                return True

            # Otherwise, it's probably a natural pause between sentences
            return False

        except Exception as e:
            logger.error(f"Error analyzing silence context: {e}")
            return False

    def _is_grammatical_continuation(self, text_before: str, text_after: str) -> bool:
        """
        Check if text_after is a grammatical continuation of text_before.

        Returns:
            True if it appears to be a continuation, False otherwise
        """
        # Check if text_before ends with a word that typically continues
        continuation_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "when",
            "while",
            "because",
            "since",
            "as",
            "that",
            "which",
            "who",
            "whom",
            "whose",
        ]

        # Get the last word of text_before
        last_word = text_before.split()[-1].lower() if text_before.split() else ""

        # If the last word suggests continuation, this might be a block
        if last_word in continuation_words:
            return True

        # Check if text_after starts with a word that typically follows
        following_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "when",
            "while",
            "because",
            "since",
            "as",
            "that",
            "which",
            "who",
            "whom",
            "whose",
        ]

        # Get the first word of text_after
        first_word = text_after.split()[0].lower() if text_after.split() else ""

        # If the first word suggests continuation, this might be a block
        if first_word in following_words:
            return True

        return False

    def _post_process_segments(self, segments: List[Dict]) -> List[Dict]:
        """Enhanced post-processing of segments."""
        processed_segments = []

        for segment in segments:
            # Handle timing adjustments
            start_time = segment["start"]
            end_time = segment["end"]

            # Clean text while preserving speech patterns
            text = segment["text"]

            # Process words with confidence scores
            words = segment.get("words", [])
            if words:
                words = [self._process_word_info(w) for w in words]

            processed_segments.append(
                {
                    "id": segment.get("id", len(processed_segments)),
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "words": words,
                    "confidence": segment.get("confidence", 0.0),
                }
            )

        return processed_segments

    def _process_word_info(self, word_info: Dict) -> Dict:
        """Process individual word information."""
        if not isinstance(word_info, dict):
            return {}

        return {
            "word": word_info.get("word", "").strip(),
            "start": word_info.get("start", 0),
            "end": word_info.get("end", 0),
            "confidence": word_info.get("confidence", 0),
            "probability": word_info.get("probability", 0),
        }

    def _validate_filler_context(self, word: str, context: List[Dict]) -> bool:
        """Validate if word is used as filler based on context."""
        context_words = [w["word"].lower() for w in context]

        # Hesitations are always fillers
        if any(word == p for p in self.speech_patterns["hesitation"]["single"]):
            return True

        # Check discourse markers
        if word in self.speech_patterns["discourse"]["single"]:
            prev_words = context_words[: context_words.index(word)]
            next_words = context_words[context_words.index(word) + 1 :]

            # Check if word breaks natural sentence flow
            if not self._is_grammatical_usage(word, prev_words, next_words):
                return True

        return False

    def _is_grammatical_usage(
        self, word: str, prev_words: List[str], next_words: List[str]
    ) -> bool:
        """Check if word is used grammatically in context."""
        # Simple heuristic for grammatical usage
        if not prev_words and not next_words:
            return False

        # Check if the word is surrounded by proper context
        if prev_words and next_words:
            return True

        return False

    def _is_filler(self, word: str) -> bool:
        """Check if word is in filler patterns."""
        word = word.lower()
        for patterns in self.speech_patterns.values():
            if isinstance(patterns, dict):
                if word in patterns["single"] or word in patterns["compound"]:
                    return True
            elif word in patterns:
                return True
        return False

    def _calculate_speech_rate(
        self, word_timings: List[Dict], duration: float
    ) -> float:
        """Calculate speech rate in words per minute."""
        if duration <= 0:
            return 0.0

        # Count content words (excluding fillers and partial words)
        content_words = [
            w
            for w in word_timings
            if not w.get("is_partial") and not self._is_filler(w["word"])
        ]

        return len(content_words) / (duration / 60)

    def _calculate_language_score(self, text: str) -> float:
        """Calculate overall language quality score."""
        # Simple language quality score based on word count and diversity
        words = word_tokenize(text.lower())
        if not words:
            return 0.0

        # Calculate word diversity
        unique_words = set(words)
        diversity = len(unique_words) / len(words)

        # Calculate average word length
        avg_word_length = sum(len(w) for w in words) / len(words)

        # Combine metrics
        return (diversity * 0.5) + (min(1.0, avg_word_length / 8) * 0.5)

    def save_all_formats(self, result: TranscriptionResult, output_dir: Path) -> None:
        """Save analysis results in multiple formats."""
        try:
            # Save plain text with detailed analysis
            self._save_txt(result, output_dir / "transcription.txt")

            # Save VTT with timing information
            self._save_vtt(result, output_dir / "transcription.vtt")

            # Save TextGrid for Praat analysis
            self._save_textgrid(result, output_dir / "transcription.TextGrid")

            # Save detailed JSON analysis
            self._save_analysis_json(result, output_dir / "analysis.json")

            # Save summary report
            self._save_summary_report(result, output_dir / "summary_report.txt")

            logger.info(f"All analysis files saved to {output_dir}")

        except Exception as e:
            logger.error(f"Error saving analysis files: {e}")
            raise

    def _save_analysis_json(
        self, result: TranscriptionResult, output_path: Path
    ) -> None:
        """Save detailed analysis in JSON format."""
        try:
            analysis = {
                "text": result.text,
                "duration": result.segments[-1]["end"] if result.segments else 0,
                "word_count": len(result.word_timings),
                "fillers": {"count": len(result.fillers), "details": result.fillers},
                "repetitions": {
                    "count": len(result.repetitions),
                    "details": result.repetitions,
                },
                "pronunciation_errors": {
                    "count": len(result.pronunciation_errors),
                    "details": result.pronunciation_errors,
                },
                "silences": {
                    "count": len(
                        [s for s in result.silences if s.get("is_block", False)]
                    ),
                    "details": result.silences,
                },
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving analysis JSON: {e}")
            raise

    def _save_textgrid(self, result: TranscriptionResult, output_path: Path) -> None:
        """Save TextGrid format for Praat analysis."""
        try:
            # Create TextGrid
            textgrid_content = 'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
            textgrid_content += f"xmin = 0\nxmax = {result.duration}\n"
            textgrid_content += "tiers? <exists>\nsize = 4\nitem []:\n"

            # Words tier
            textgrid_content += self._create_textgrid_tier(
                "words", result.word_timings, 1
            )

            # Fillers tier
            textgrid_content += self._create_textgrid_tier("fillers", result.fillers, 2)

            # Repetitions tier
            textgrid_content += self._create_textgrid_tier(
                "repetitions", result.repetitions, 3
            )

            # Pronunciation errors tier
            textgrid_content += self._create_textgrid_tier(
                "pronunciation", result.pronunciation_errors, 4
            )

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(textgrid_content)

        except Exception as e:
            logger.error(f"Error saving TextGrid: {e}")
            raise

    def _create_textgrid_tier(self, name: str, items: List[Dict], tier_num: int) -> str:
        """Create a tier for TextGrid."""
        content = f"    item [{tier_num}]:\n"
        content += f'        class = "IntervalTier"\n'
        content += f'        name = "{name}"\n'
        content += f"        xmin = 0\n"

        if not items:
            content += f"        xmax = 0\n"
            content += f"        intervals: size = 0\n"
            return content

        xmax = max(item.get("end", 0) for item in items)
        content += f"        xmax = {xmax}\n"
        content += f"        intervals: size = {len(items)}\n"

        for i, item in enumerate(items, 1):
            content += f"        intervals [{i}]:\n"
            content += f"            xmin = {item.get('start', 0)}\n"
            content += f"            xmax = {item.get('end', 0)}\n"

            if "word" in item:
                text = item["word"]
            elif "pattern" in item:
                text = f"REP:{','.join(p['word'] for p in item['pattern'])}"
            elif "reference" in item:
                text = f"PRON:{item['word']}->{item['reference']}"
            else:
                text = ""

            content += f'            text = "{text}"\n'

        return content

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to VTT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _save_vtt(self, result: TranscriptionResult, output_path: Path) -> None:
        """Save WebVTT format with enhanced timing."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")

                for i, segment in enumerate(result.segments):
                    start = self._format_timestamp(segment["start"])
                    end = self._format_timestamp(segment["end"])
                    f.write(f"{i+1}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{segment['text']}\n\n")

        except Exception as e:
            logger.error(f"Error saving VTT: {e}")
            raise

    def _save_txt(self, result: TranscriptionResult, output_path: Path) -> None:
        """Save detailed text transcription."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # Write header
                f.write("SPEECH ANALYSIS TRANSCRIPT\n")
                f.write("=" * 50 + "\n\n")

                # Write metadata
                f.write(f"Duration: {result.duration:.2f} seconds\n")
                f.write(f"Speech Rate: {result.speech_rate:.1f} words per minute\n")
                f.write(f"Language Score: {result.language_score:.2f}/1.0\n")
                f.write(f"Overall Confidence: {result.confidence:.2f}\n\n")

                # Write full text
                f.write("FULL TRANSCRIPTION:\n")
                f.write("-" * 20 + "\n")
                f.write(result.text + "\n\n")

                # Write segments with timestamps
                f.write("TIMESTAMPED SEGMENTS:\n")
                f.write("-" * 20 + "\n")
                for segment in result.segments:
                    start = self._format_timestamp(segment["start"])
                    end = self._format_timestamp(segment["end"])
                    f.write(f"[{start} --> {end}] {segment['text']}\n")

                # Write analysis
                self._write_analysis_section(f, result)

        except Exception as e:
            logger.error(f"Error saving TXT: {e}")
            raise

    def _write_analysis_section(self, file, result: TranscriptionResult) -> None:
        """Write detailed analysis section to text file."""
        file.write("\nDETAILED ANALYSIS:\n")
        file.write("-" * 20 + "\n\n")

        # Write filler analysis
        file.write("Filler Words:\n")
        for filler in result.fillers:
            start = self._format_timestamp(filler["start"])
            file.write(f"- '{filler['word']}' at {start} ({filler['filler_type']})\n")

        file.write("\nRepetitions:\n")
        for rep in result.repetitions:
            start = self._format_timestamp(rep["start"])
            file.write(f"- '{rep['word']}' repeated {rep['count']} times at {start}\n")

        file.write("\nPronunciation Errors and Prolongations:\n")
        for error in result.pronunciation_errors:
            start = self._format_timestamp(error["start"])
            if "prolongation" in error.get("event_type", ""):
                file.write(f"- Prolongation: '{error['word']}' at {start}\n")
            else:
                file.write(
                    f"- '{error['word']}' should be '{error.get('reference', '')}' at {start}\n"
                )

        file.write("\nSilences and Blocks:\n")
        for silence in result.silences:
            start = self._format_timestamp(silence["start"])
            end = self._format_timestamp(silence["end"])
            if silence.get("is_block", False):
                file.write(
                    f"- Block: {silence['duration']:.2f}s from {start} to {end}\n"
                )
            else:
                file.write(
                    f"- Silence: {silence['duration']:.2f}s from {start} to {end}\n"
                )

    def _save_summary_report(
        self, result: TranscriptionResult, output_path: Path
    ) -> None:
        """Save concise summary report."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("SPEECH ANALYSIS SUMMARY\n")
                f.write("=" * 30 + "\n\n")

                # Key metrics
                f.write("Key Metrics:\n")
                f.write(f"- Duration: {result.duration:.2f} seconds\n")
                f.write(f"- Speech Rate: {result.speech_rate:.1f} words/minute\n")
                f.write(f"- Language Score: {result.language_score:.2f}/1.0\n")
                f.write(f"- Confidence: {result.confidence:.2f}/1.0\n\n")

                # Statistics
                f.write("Statistics:\n")
                f.write(f"- Total Words: {len(result.word_timings)}\n")
                f.write(f"- Filler Words: {len(result.fillers)}\n")
                f.write(f"- Repetitions: {len(result.repetitions)}\n")
                f.write(
                    f"- Prolongations: {len([e for e in result.pronunciation_errors if 'prolongation' in e.get('event_type', '')])}\n"
                )
                f.write(
                    f"- Pronunciation Errors: {len([e for e in result.pronunciation_errors if 'prolongation' not in e.get('event_type', '')])}\n"
                )
                f.write(
                    f"- Blocks: {len([s for s in result.silences if s.get('is_block', False)])}\n"
                )
                f.write(
                    f"- Natural Pauses: {len([s for s in result.silences if not s.get('is_block', False)])}\n"
                )

        except Exception as e:
            logger.error(f"Error saving summary report: {e}")
            raise
