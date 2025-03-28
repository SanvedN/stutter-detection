import os
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import Levenshtein
from src.utils.audio_utils import load_audio, normalize_audio, apply_noise_reduction
from src.audio.transcription_analyzer import TranscriptionAnalyzer, TranscriptionResult
from src.visualization.speech_visualizer import SpeechVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Grandfather's passage for reference
GRANDFATHERS_PASSAGE = """You wished to know all about my grandfather. Well, he is nearly ninety-three years old. He dresses himself in an ancient black frock coat, usually minus several buttons; yet he still thinks as swiftly as ever. A long, flowing beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. When he speaks his voice is just a bit cracked and quivers a trifle. Twice each day he plays skillfully and with zest upon our small organ. Except in the winter when the ooze or snow or ice prevents, he slowly takes a short walk in the open air each day. We have often urged him to walk more and smoke less, but he always answers, "Banana Oil!" Grandfather likes to be modern in his language."""


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
                "num_blocks": len(result.silences),
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
            + result.pronunciation_errors
            + result.silences
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
        transcription_norm = transcription.lower().strip()
        reference_norm = GRANDFATHERS_PASSAGE.lower().strip()

        # Calculate Levenshtein distance and similarity ratio
        distance = Levenshtein.distance(transcription_norm, reference_norm)
        similarity = Levenshtein.ratio(transcription_norm, reference_norm)

        # Identify specific discrepancies
        discrepancies = self._identify_discrepancies(transcription_norm, reference_norm)

        return {
            "distance": distance,
            "similarity": similarity,
            "discrepancies": discrepancies,
            "discrepancy_count": len(discrepancies),
        }

    def _identify_discrepancies(self, transcription: str, reference: str) -> list:
        """
        Identify specific discrepancies between transcription and reference.

        Returns:
            list: List of discrepancy objects with type and details
        """
        discrepancies = []

        # Split into words
        trans_words = transcription.split()
        ref_words = reference.split()

        # Use dynamic programming to align words
        alignment = self._align_texts(trans_words, ref_words)

        for i, (trans_idx, ref_idx) in enumerate(alignment):
            if trans_idx is not None and ref_idx is not None:
                # Both words exist - check for mismatch
                trans_word = trans_words[trans_idx]
                ref_word = ref_words[ref_idx]

                if trans_word != ref_word:
                    # Calculate similarity
                    word_similarity = Levenshtein.ratio(trans_word, ref_word)

                    if word_similarity < 0.8:  # Significant difference
                        discrepancies.append(
                            {
                                "type": "substitution",
                                "transcribed": trans_word,
                                "reference": ref_word,
                                "position": trans_idx,
                                "similarity": word_similarity,
                            }
                        )
            elif trans_idx is not None and ref_idx is None:
                # Word in transcription but not in reference - possible insertion/repetition
                discrepancies.append(
                    {
                        "type": "insertion",
                        "transcribed": trans_words[trans_idx],
                        "position": trans_idx,
                    }
                )
            elif trans_idx is None and ref_idx is not None:
                # Word in reference but not in transcription - possible omission
                discrepancies.append(
                    {
                        "type": "omission",
                        "reference": ref_words[ref_idx],
                        "position": ref_idx,
                    }
                )

        return discrepancies

    def _align_texts(self, transcribed: list, reference: list) -> list:
        """
        Align transcribed text with reference text using dynamic programming.

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

    def _calculate_fluency_score(self, result, passage_comparison) -> tuple:
        """
        Calculate stutter fluency score and severity with enhanced accuracy.

        Incorporates passage comparison results for more accurate scoring.
        """
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
        blocks_count = len(result.silences)
        fillers_count = len(result.fillers)

        # Count discrepancies from reference passage
        discrepancy_count = passage_comparison["discrepancy_count"]

        # Calculate weights for different stutter types
        repetition_weight = 1.0
        prolongation_weight = 1.2
        block_weight = 1.5
        filler_weight = 0.5
        discrepancy_weight = 0.8

        # Calculate weighted stutter events
        weighted_stutters = (
            repetitions_count * repetition_weight
            + prolongations_count * prolongation_weight
            + blocks_count * block_weight
            + fillers_count * filler_weight
            + discrepancy_count * discrepancy_weight
        )

        # Compute %SS (Percentage of Syllables Stuttered)
        percent_ss = (weighted_stutters / total_syllables) * 100

        # Extract durations for severity calculation
        all_events = (
            result.repetitions
            + result.fillers
            + result.pronunciation_errors
            + result.silences
        )

        try:
            # Find longest stutter duration
            longest_stutter = max(
                [
                    event.get("duration", event.get("end", 0) - event.get("start", 0))
                    for event in all_events
                ],
                default=0,
            )
        except (AttributeError, TypeError):
            longest_stutter = 0  # Default if duration is missing

        # Calculate duration score
        duration_score = self._get_duration_score(longest_stutter)

        # Calculate frequency score
        frequency_score = self._get_frequency_score(weighted_stutters, total_syllables)

        # Calculate passage similarity penalty
        similarity_penalty = self._get_similarity_penalty(
            passage_comparison["similarity"]
        )

        # Compute final fluency score (lower is better)
        fluency_score = min(
            100,
            max(
                0,
                int(percent_ss) + duration_score + frequency_score + similarity_penalty,
            ),
        )

        # Determine severity level
        severity = self._get_severity_level(fluency_score)

        return fluency_score, severity

    def _get_duration_score(self, duration) -> int:
        """Assigns duration score based on the longest stuttering event."""
        if duration < 0.5:
            return 0
        elif duration < 1.0:
            return 2
        elif duration < 2.0:
            return 4
        elif duration < 3.0:
            return 6
        elif duration < 5.0:
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
            return 2
        elif similarity > 0.8:
            return 5
        elif similarity > 0.7:
            return 8
        elif similarity > 0.6:
            return 12
        else:
            return 15

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
            if silence.get("position", "") == "middle":  # Only include middle silences
                formatted_events.append(
                    {
                        "type": "block",
                        "subtype": "silence",
                        "start": silence.get("start", 0),
                        "end": silence.get("end", 0),
                        "duration": silence.get("duration", 0),
                        "confidence": 0.9,  # High confidence for silence detection
                    }
                )

        return formatted_events
