import os
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import Levenshtein
import re
import inflect
from src.utils.audio_utils import load_audio, normalize_audio, apply_noise_reduction
from src.audio.transcription_analyzer import TranscriptionAnalyzer, TranscriptionResult
from src.visualization.speech_visualizer import SpeechVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize inflect engine for number conversion
p = inflect.engine()

# Grandfather's passage for reference
GRANDFATHERS_PASSAGE = """You wish to know about my grandfather. Well, he is nearly 93 years old, yet he still thinks as swiftly as ever. He dresses himself in an old black frock coat, usually several buttons missing. A long beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. When he speaks, his voice is just a bit cracked and quivers a bit. Twice each day he plays skillfully and with zest upon a small organ. Except in the winter when the snow or ice prevents, he slowly takes a short walk in the open air each day. We have often urged him to walk more and smoke less, but he always answers, “Banana oil!”. Grandfather likes to be modern in his language."""


class SpeechAnalyzer:
    def __init__(self):
        """Initialize all analysis components."""
        try:
            self.transcriber = TranscriptionAnalyzer(model_size="medium")
            self.visualizer = SpeechVisualizer()
            logger.info("Speech Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Speech Analyzer: {e}")
            raise

    def analyze_audio_file(self, file_path: str) -> dict:
        """Perform full speech analysis and return the results."""
        try:
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"results/{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)

            viz_dir = output_dir / "visualizations"
            transcripts_dir = output_dir / "transcripts"

            for directory in [viz_dir, transcripts_dir]:
                directory.mkdir(exist_ok=True)

            # Load and preprocess audio
            logger.info("Loading and preprocessing audio...")
            audio_data, sample_rate = self._load_and_preprocess_audio(file_path)

            # Perform transcription and analysis
            logger.info("Performing transcription and analysis...")
            result = self.transcriber.analyze_audio(
                audio_data, sample_rate, transcripts_dir
            )

            # Compare with reference passage
            logger.info("Comparing with reference passage...")
            passage_comparison = self._compare_with_reference(result.text)

            # Compute fluency score
            fluency_score, severity = self._calculate_fluency_score(
                result, passage_comparison
            )

            # Generate visualizations
            logger.info("Generating visualizations...")
            self._generate_visualizations(audio_data, result, viz_dir)

            # Return results as a dictionary
            return {
                "transcription": result.text,
                "stutter_events": self._format_stutter_events(result),
                "fluency_score": fluency_score,
                "num_repetitions": len(result.repetitions),
                "num_fillers": len(result.fillers),
                "num_prolongations": len(
                    [
                        e
                        for e in result.pronunciation_errors
                        if "prolongation" in e.get("event_type", "")
                    ]
                ),
                "num_blocks": len(
                    [s for s in result.silences if s.get("is_block", False)]
                ),
                "passage_comparison": passage_comparison,
                "severity": severity,
                "visualization_path": str(viz_dir / "waveform_analysis.png"),
            }

        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"error": str(e)}

    def _load_and_preprocess_audio(self, file_path: str) -> tuple:
        """Load and preprocess audio file."""
        audio_data, sample_rate = load_audio(file_path, target_sr=16000)
        audio_data = normalize_audio(audio_data)
        audio_data = apply_noise_reduction(audio_data, sample_rate)
        return audio_data, sample_rate

    def _generate_visualizations(self, audio_data: np.ndarray, result, viz_dir: Path):
        """Generate visualizations for analysis."""
        # Combine all events for visualization
        all_events = (
            result.repetitions
            + result.fillers
            + [e for e in result.pronunciation_errors if e.get("confidence", 0) > 0.6]
            + [s for s in result.silences if s.get("is_block", False)]
        )

        fig_wave = self.visualizer.create_analysis_dashboard(
            audio_data=audio_data,
            features=result.word_timings,
            events=all_events,
            sample_rate=16000,
        )
        self.visualizer.save_visualization(fig_wave, viz_dir / "waveform_analysis.png")

    def _compare_with_reference(self, transcription: str) -> dict:
        """
        Compare transcription with the Grandfather's Passage to identify discrepancies.

        Returns:
            dict: Comparison metrics and identified discrepancies
        """
        # Normalize texts for comparison
        transcription_norm = self._normalize_text_for_comparison(transcription)
        reference_norm = self._normalize_text_for_comparison(GRANDFATHERS_PASSAGE)

        reference_norm = " ".join(
            reference_norm.split()[: len(transcription_norm.split())]
        )

        # Calculate Levenshtein distance and similarity ratio
        distance = Levenshtein.distance(transcription_norm, reference_norm)
        similarity = Levenshtein.ratio(transcription_norm, reference_norm)

        # Identify specific discrepancies
        discrepancies = self._identify_discrepancies(transcription_norm, reference_norm)

        # Filter out likely false positives
        filtered_discrepancies = self._filter_false_positives(discrepancies)

        return {
            "distance": distance,
            "spoken_word_count": len(transcription_norm.split()),
            "reference_word_count": len(reference_norm.split()),
            "discrepancies": filtered_discrepancies,
            "discrepancy_count": len(filtered_discrepancies),
            "raw_discrepancy_count": len(discrepancies),
        }

    def _normalize_text_for_comparison(self, text: str) -> str:
        """
        Normalize text for more accurate comparison.

        - Convert to lowercase
        - Normalize numbers (convert digits to words)
        - Remove punctuation
        - Standardize whitespace
        - Remove common filler words
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation except hyphens (important for stutters)
        text = re.sub(r"[^\w\s\-]", "", text)

        # Standardize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Normalize numbers (convert digits to words)
        words = []
        for word in text.split():
            if word.isdigit():
                try:
                    # Convert number to words (e.g., "93" to "ninety-three")
                    word = p.number_to_words(word).replace(" and ", " ")
                except:
                    pass  # Keep original if conversion fails
            words.append(word)

        text = " ".join(words)

        # Remove common filler words that don't affect meaning
        fillers = ["um", "uh", "er", "ah", "like", "you know"]
        for filler in fillers:
            text = re.sub(r"\b" + filler + r"\b", "", text)

        # Clean up any double spaces created by filler removal
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _identify_discrepancies(self, transcription: str, reference: str) -> list:
        """
        Identify specific discrepancies between transcription and reference.

        Uses a more sophisticated alignment algorithm to identify true discrepancies.

        Returns:
            list: List of discrepancy objects with type and details
        """
        discrepancies = []

        # Split into words
        trans_words = transcription.split()
        ref_words = reference.split()

        # Use dynamic programming to align words
        alignment = self._align_texts(trans_words, ref_words)

        # Track consecutive repetitions
        repetition_sequence = []
        last_word = None

        for i, (trans_idx, ref_idx) in enumerate(alignment):
            # Handle repetitions
            if trans_idx is not None:
                current_word = trans_words[trans_idx]

                # Check for repetition
                if last_word == current_word:
                    repetition_sequence.append(current_word)
                else:
                    # Process any completed repetition sequence
                    if len(repetition_sequence) > 1:
                        discrepancies.append(
                            {
                                "type": "repetition",
                                "words": repetition_sequence.copy(),
                                "count": len(repetition_sequence),
                                "position": trans_idx - len(repetition_sequence),
                            }
                        )
                    repetition_sequence = [current_word]

                last_word = current_word

            # Process regular alignment discrepancies
            if trans_idx is not None and ref_idx is not None:
                # Both words exist - check for mismatch
                trans_word = trans_words[trans_idx]
                ref_word = ref_words[ref_idx]

                # Skip if words are equivalent after normalization
                if self._are_words_equivalent(trans_word, ref_word):
                    continue

                if trans_word != ref_word:
                    # Calculate similarity
                    word_similarity = Levenshtein.ratio(trans_word, ref_word)

                    # Check for prolongation (repeated characters)
                    has_prolongation = bool(re.search(r"([a-z])\1{2,}", trans_word))

                    # Check for partial word (hyphenated)
                    is_partial = "-" in trans_word

                    if word_similarity < 0.7:  # Significant difference
                        discrepancies.append(
                            {
                                "type": "substitution",
                                "transcribed": trans_word,
                                "reference": ref_word,
                                "position": trans_idx,
                                "similarity": word_similarity,
                                "has_prolongation": has_prolongation,
                                "is_partial": is_partial,
                            }
                        )
                    elif has_prolongation:
                        discrepancies.append(
                            {
                                "type": "prolongation",
                                "transcribed": trans_word,
                                "reference": ref_word,
                                "position": trans_idx,
                            }
                        )
                    elif is_partial:
                        discrepancies.append(
                            {
                                "type": "partial_word",
                                "transcribed": trans_word,
                                "reference": ref_word,
                                "position": trans_idx,
                            }
                        )
            elif trans_idx is not None and ref_idx is None:
                # Word in transcription but not in reference
                # Check if it's a common variation or synonym
                trans_word = trans_words[trans_idx]
                if not self._is_common_variation(trans_word, ref_words):
                    discrepancies.append(
                        {
                            "type": "insertion",
                            "transcribed": trans_word,
                            "position": trans_idx,
                            "has_prolongation": bool(
                                re.search(r"([a-z])\1{2,}", trans_word)
                            ),
                            "is_partial": "-" in trans_word,
                        }
                    )
            elif trans_idx is None and ref_idx is not None:
                # Word in reference but not in transcription
                ref_word = ref_words[ref_idx]
                if not self._is_common_variation(ref_word, trans_words):
                    discrepancies.append(
                        {"type": "omission", "reference": ref_word, "position": ref_idx}
                    )

        # Check for any final repetition sequence
        if len(repetition_sequence) > 1:
            discrepancies.append(
                {
                    "type": "repetition",
                    "words": repetition_sequence,
                    "count": len(repetition_sequence),
                    "position": len(trans_words) - len(repetition_sequence),
                }
            )

        return discrepancies

    def _are_words_equivalent(self, word1: str, word2: str) -> bool:
        """
        Check if two words are semantically equivalent despite different forms.

        Handles:
        - Numbers in digit vs word form
        - Common contractions
        - Spelling variations
        """
        # Check for exact match
        if word1 == word2:
            return True

        # Check for number equivalence
        if word1.isdigit() or word2.isdigit():
            try:
                # Convert digit to word form if needed
                word1_norm = (
                    p.number_to_words(word1).replace(" and ", " ")
                    if word1.isdigit()
                    else word1
                )
                word2_norm = (
                    p.number_to_words(word2).replace(" and ", " ")
                    if word2.isdigit()
                    else word2
                )

                # Compare normalized forms
                if word1_norm == word2_norm:
                    return True

                # Handle hyphenated numbers
                word1_norm = word1_norm.replace("-", " ")
                word2_norm = word2_norm.replace("-", " ")
                if word1_norm == word2_norm:
                    return True
            except:
                pass

        # Check for high similarity (spelling variations)
        similarity = Levenshtein.ratio(word1, word2)
        if similarity > 0.85:
            return True

        # Check for common synonyms and variations
        synonyms = {
            "renounced": ["pronounced"],
            "pronounced": ["renounced"],
            "jest": ["zest"],
            "zest": ["jest"],
            "addresses": ["dresses"],
            "dresses": ["addresses"],
            "bit": ["trifle"],
            "trifle": ["bit"],
        }

        if word1 in synonyms and word2 in synonyms.get(word1, []):
            return True

        return False

    def _is_common_variation(self, word: str, word_list: list) -> bool:
        """
        Check if a word is a common variation of any word in the list.

        Used to reduce false positives in insertions/omissions.
        """
        # Common words that might be added/omitted in natural speech
        common_variations = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "so",
            "very",
            "quite",
            "just",
            "really",
            "actually",
            "basically",
            "well",
            "now",
            "then",
            "you",
            "know",
            "see",
            "like",
        ]

        # Check if it's a common filler/connector word
        if word.lower() in common_variations:
            return True

        # Check if it's a close match to any word in the list
        for other_word in word_list:
            if self._are_words_equivalent(word, other_word):
                return True

        return False

    def _filter_false_positives(self, discrepancies: list) -> list:
        """
        Filter out likely false positives from the discrepancies list.

        Returns:
            list: Filtered discrepancies
        """
        filtered = []

        # Words that are commonly misinterpreted or have multiple valid forms
        common_variations = {
            "ninety-three": ["93", "ninety three", "93 years"],
            "93": ["ninety-three", "ninety three"],
            "dresses": ["addresses", "dressed"],
            "addresses": ["dresses", "dressed"],
            "pronounced": ["renounced"],
            "renounced": ["pronounced"],
            "zest": ["jest"],
            "jest": ["zest"],
            "trifle": ["bit", "little"],
            "bit": ["trifle", "little"],
        }

        for disc in discrepancies:
            # Keep all repetitions, prolongations, and partial words as they're likely real stutters
            if disc["type"] in ["repetition", "prolongation", "partial_word"]:
                filtered.append(disc)
                continue

            # For substitutions, check if they're common variations
            if disc["type"] == "substitution":
                # If it has prolongation or is partial, it's likely a stutter
                if disc.get("has_prolongation", False) or disc.get("is_partial", False):
                    filtered.append(disc)
                    continue

                # Check if it's a common variation
                ref_word = disc.get("reference", "").lower()
                trans_word = disc.get("transcribed", "").lower()

                if (
                    ref_word in common_variations
                    and trans_word in common_variations.get(ref_word, [])
                ):
                    # This is a common variation, not a true discrepancy
                    continue

                if (
                    trans_word in common_variations
                    and ref_word in common_variations.get(trans_word, [])
                ):
                    # This is a common variation, not a true discrepancy
                    continue

                # If similarity is very low, it might be a real substitution error
                if disc.get("similarity", 1.0) < 0.4:
                    filtered.append(disc)
                    continue

                # Otherwise, it might be a normal speech variation
                continue

            # For insertions, check if they're likely stutters or common variations
            if disc["type"] == "insertion":
                if disc.get("has_prolongation", False) or disc.get("is_partial", False):
                    filtered.append(disc)
                    continue

                # Check if it's a common filler word
                if "transcribed" in disc:
                    word = disc["transcribed"].lower()
                    if word in ["um", "uh", "er", "ah", "like", "you know"]:
                        # This is a filler, not a true discrepancy
                        continue

                    # Check if it's a common word that might be added in natural speech
                    if word in [
                        "the",
                        "a",
                        "an",
                        "and",
                        "or",
                        "but",
                        "so",
                        "very",
                        "quite",
                        "just",
                    ]:
                        continue

                # Otherwise include it
                filtered.append(disc)
                continue

            # For omissions, check if they're common words that might be omitted
            if disc["type"] == "omission":
                if "reference" in disc:
                    word = disc["reference"].lower()
                    if word in [
                        "the",
                        "a",
                        "an",
                        "and",
                        "or",
                        "but",
                        "so",
                        "very",
                        "quite",
                        "just",
                    ]:
                        continue

                filtered.append(disc)
                continue

            # Default: include the discrepancy
            filtered.append(disc)

        return filtered

    def _align_texts(self, transcribed: list, reference: list) -> list:
        """
        Align transcribed text with reference text using dynamic programming.

        Improved to handle stutters and repetitions better.

        Returns:
            list: List of tuples (trans_idx, ref_idx) representing alignment
        """
        # Create a matrix of edit distances
        m, n = len(transcribed), len(reference)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill the matrix with improved scoring for stutters
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Check for exact match or equivalent words
                if self._are_words_equivalent(transcribed[i - 1], reference[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Check for potential stutter (repetition or partial word)
                    is_potential_stutter = False

                    # Check for repetition (current word same as previous)
                    if i > 1 and transcribed[i - 1] == transcribed[i - 2]:
                        is_potential_stutter = True

                    # Check for partial word (hyphenated)
                    if "-" in transcribed[i - 1]:
                        is_potential_stutter = True

                    # Check for prolongation (repeated characters)
                    if re.search(r"([a-z])\1{2,}", transcribed[i - 1]):
                        is_potential_stutter = True

                    # If it's a potential stutter, favor deletion (keep reference word)
                    if is_potential_stutter:
                        dp[i][j] = min(
                            dp[i - 1][j - 1] + 1,  # substitution
                            dp[i - 1][j] + 0.5,  # deletion (lower cost for stutters)
                            dp[i][j - 1] + 1,  # insertion
                        )
                    else:
                        # Standard edit distance
                        dp[i][j] = min(
                            dp[i - 1][j - 1] + 1,  # substitution
                            dp[i - 1][j] + 1,  # deletion
                            dp[i][j - 1] + 1,  # insertion
                        )

        # Backtrack to find alignment
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if (
                i > 0
                and j > 0
                and self._are_words_equivalent(transcribed[i - 1], reference[j - 1])
            ):
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                alignment.append((i - 1, j - 1))  # substitution
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + (
                0.5 if self._is_potential_stutter(transcribed[i - 1]) else 1
            ):
                alignment.append((i - 1, None))  # deletion
                i -= 1
            else:
                alignment.append((None, j - 1))  # insertion
                j -= 1

        return list(reversed(alignment))

    def _is_potential_stutter(self, word: str) -> bool:
        """Check if a word is potentially a stutter."""
        # Check for repetition, partial word, or prolongation
        if "-" in word:
            return True

        if re.search(r"([a-z])\1{2,}", word):
            return True

        return False

    def _calculate_fluency_score(self, result, passage_comparison) -> tuple:
        """
        Calculate stutter fluency score and severity with enhanced accuracy.

        Incorporates passage comparison results for more accurate scoring.
        """
        try:
            # Count total syllables (approximation)
            total_syllables = max(1, len(result.text.split()))  # Avoid division by zero

            # Count all stutter events
            repetitions_count = len(result.repetitions)
            prolongations_count = len(
                [
                    e
                    for e in result.pronunciation_errors
                    if "prolongation" in e.get("event_type", "")
                ]
            )
            blocks_count = len([s for s in result.silences if s.get("is_block", False)])
            fillers_count = len(result.fillers)

            # Count discrepancies from reference passage
            discrepancy_count = passage_comparison["discrepancy_count"]

            # Calculate weights for different stutter types
            repetition_weight = 1.0
            prolongation_weight = 1.2
            block_weight = 1.5
            filler_weight = 0.5
            discrepancy_weight = (
                0.2  # Further reduced weight as this is a secondary indicator
            )

            # Calculate weighted stutter events
            weighted_stutters = (
                repetitions_count * repetition_weight
                + prolongations_count * prolongation_weight
                + blocks_count * block_weight
                + fillers_count * filler_weight
                + discrepancy_count * discrepancy_weight
            )

            # Compute %SS (Percentage of Syllables Stuttered)
            percent_ss = min(100, (weighted_stutters / total_syllables) * 100)

            # Extract durations for severity calculation
            all_events = (
                result.repetitions
                + [
                    e
                    for e in result.pronunciation_errors
                    if "prolongation" in e.get("event_type", "")
                ]
                + [s for s in result.silences if s.get("is_block", False)]
                + result.fillers
            )

            # Find longest stutter duration
            longest_stutter = 0
            if all_events:
                try:
                    longest_stutter = max(
                        [self._get_event_duration(event) for event in all_events]
                    )
                except (ValueError, TypeError):
                    longest_stutter = 0

            # Calculate duration score
            duration_score = self._get_duration_score(longest_stutter)

            # Calculate frequency score
            frequency_score = self._get_frequency_score(
                weighted_stutters, total_syllables
            )

            # Calculate passage similarity penalty
            similarity_penalty = self._get_similarity_penalty(
                passage_comparison["spoken_word_count"]
                / passage_comparison["reference_word_count"]
            )

            # Compute final fluency score (lower is better)
            fluency_score = min(
                100,
                max(
                    0,
                    int(percent_ss)
                    + duration_score
                    + frequency_score
                    + similarity_penalty,
                ),
            )

            # Determine severity level
            severity = self._get_severity_level(fluency_score)

            return 100 - fluency_score, severity

        except Exception as e:
            logger.error(f"Error calculating fluency score: {e}")
            # Return default values in case of error
            return 50, "Moderate"

    def _get_event_duration(self, event):
        """Safely extract duration from an event."""
        if "duration" in event:
            return event["duration"]
        elif "start" in event and "end" in event:
            return event["end"] - event["start"]
        return 0

    def _get_duration_score(self, duration) -> int:
        """Assigns duration score based on the longest stuttering event."""
        if duration < 0.3:
            return 0
        elif duration < 0.7:
            return 2
        elif duration < 1.5:
            return 4
        elif duration < 2.5:
            return 6
        elif duration < 4.0:
            return 8
        else:
            return 10

    def _get_frequency_score(self, stutter_count, total_syllables) -> int:
        """Assigns frequency score based on stutter frequency."""
        frequency = (stutter_count / total_syllables) * 100

        if frequency < 1:
            return 0
        elif frequency < 2:
            return 2
        elif frequency < 5:
            return 4
        elif frequency < 8:
            return 6
        elif frequency < 12:
            return 8
        else:
            return 10

    def _get_similarity_penalty(self, similarity) -> int:
        """
        Calculate penalty based on similarity to reference passage.
        Lower similarity = higher penalty
        """
        if similarity > 0.95:
            return 0
        elif similarity > 0.9:
            return 1
        elif similarity > 0.8:
            return 2
        elif similarity > 0.7:
            return 4
        elif similarity > 0.6:
            return 6
        else:
            return 8

    def _get_severity_level(self, score) -> str:
        """Determines severity level based on enhanced scoring."""
        if score <= 10:
            return "Very Mild"
        elif score <= 20:
            return "Mild"
        elif score <= 30:
            return "Moderate"
        elif score <= 40:
            return "Severe"
        else:
            return "Very Severe"

    def _format_stutter_events(self, result) -> list:
        """
        Format all stutter events into a consistent structure for the API response.
        """
        formatted_events = []

        # Process repetitions
        for rep in result.repetitions:
            formatted_events.append(
                {
                    "type": "repetition",
                    "subtype": rep.get("repetition_type", "simple"),
                    "start": rep.get("start", 0),
                    "end": rep.get("end", 0),
                    "duration": rep.get("end", 0) - rep.get("start", 0),
                    "text": rep.get("word", ""),
                    "count": rep.get("count", 1),
                    "confidence": rep.get("confidence", 0.0),
                }
            )

        # Process fillers
        for filler in result.fillers:
            formatted_events.append(
                {
                    "type": "filler",
                    "subtype": filler.get("filler_type", "hesitation"),
                    "start": filler.get("start", 0),
                    "end": filler.get("end", 0),
                    "duration": filler.get("end", 0) - filler.get("start", 0),
                    "text": filler.get("word", ""),
                    "confidence": filler.get("confidence", 0.0),
                }
            )

        # Process pronunciation errors (prolongations)
        for error in result.pronunciation_errors:
            if "prolongation" in error.get("event_type", ""):
                formatted_events.append(
                    {
                        "type": "prolongation",
                        "subtype": "sound_prolongation",
                        "start": error.get("start", 0),
                        "end": error.get("end", 0),
                        "duration": error.get("end", 0) - error.get("start", 0),
                        "text": error.get("word", ""),
                        "confidence": error.get("confidence", 0.0),
                    }
                )

        # Process silences (blocks)
        for silence in result.silences:
            if silence.get("is_block", False):  # Only include blocks
                formatted_events.append(
                    {
                        "type": "block",
                        "subtype": "silence",
                        "start": silence.get("start", 0),
                        "end": silence.get("end", 0),
                        "duration": silence.get("duration", 0),
                        "confidence": silence.get(
                            "confidence", 0.9
                        ),  # High confidence for silence detection
                    }
                )

        return formatted_events
