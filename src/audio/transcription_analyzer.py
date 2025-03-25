"""
transcription_analyzer.py

Enhanced speech transcription and analysis module with high accuracy detection.
Handles transcription, filler detection, and exports in multiple formats.

Features:
   - High-accuracy speech-to-text conversion using Whisper
   - Advanced filler word and repetition detection with context analysis
   - Multiple export formats (TXT, VTT, TextGrid, JSON)
   - Comprehensive analysis reports with grammar checking
   - Improved error handling and logging for production readiness
"""

import whisper
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import torch
import re
from datetime import datetime
import json
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import string

# Download required NLTK data
try:
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("maxent_ne_chunker")
    nltk.download("words")
except Exception as e:
    logging.warning(f"Failed to download NLTK data: {e}")

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logging.warning(f"Failed to load spaCy model: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Container for transcription analysis results."""

    text: str
    segments: List[Dict]
    word_timings: List[Dict]
    fillers: List[Dict]
    repetitions: List[Dict]
    grammar_errors: List[Dict]
    confidence: float
    duration: float
    speech_rate: float
    language_score: float


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

            # Compile regex patterns for filler detection
            self._compile_patterns()

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

            # Transcribe with enhanced detection
            result = self.transcribe_with_enhanced_detection(audio_data, sample_rate)

            # Save results in multiple formats
            self.save_all_formats(result, output_dir)

            return result

        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            raise

    def transcribe_with_enhanced_detection(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> TranscriptionResult:
        """Perform high-accuracy transcription with enhanced detection."""
        try:
            # Perform initial transcription with optimized settings
            result = self.model.transcribe(
                audio_data,
                language="en",
                word_timestamps=True,
                condition_on_previous_text=True,
                initial_prompt="Include hesitations, fillers, repetitions, and partial words exactly as spoken.",
                temperature=0.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                beam_size=5,
            )

            # Enhanced post-processing on segments
            segments = self._post_process_segments(result["segments"])
            word_timings = self._extract_enhanced_word_timings(segments)
            fillers = self._detect_fillers_with_context(word_timings)
            repetitions = self._enhanced_repetition_detection(word_timings)
            grammar_errors = self._analyze_grammar(result["text"])

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
                grammar_errors=grammar_errors,
                confidence=confidence,
                duration=duration,
                speech_rate=speech_rate,
                language_score=language_score,
            )

        except Exception as e:
            logger.error(f"Error in enhanced transcription: {e}")
            raise

    def _post_process_segments(self, segments: List[Dict]) -> List[Dict]:
        """Enhanced post-processing of segments."""
        processed_segments = []

        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
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
            "is_partial": bool(re.search(r"-", word_info.get("word", ""))),
        }

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
                timing = {
                    "word": word,
                    "start": word_info.get("start", segment["start"]),
                    "end": word_info.get("end", segment["end"]),
                    "confidence": word_info.get("confidence", 0.0),
                    "is_partial": word_info.get("is_partial", False),
                    "segment_id": segment["id"],
                }
                word_timings.append(timing)
        return word_timings

    def _detect_fillers_with_context(self, word_timings: List[Dict]) -> List[Dict]:
        """Detect filler words using context analysis."""
        fillers = []
        window_size = 3

        for i, word_info in enumerate(word_timings):
            word = word_info["word"].lower()
            # Get context window
            start_idx = max(0, i - window_size)
            end_idx = min(len(word_timings), i + window_size + 1)
            context = word_timings[start_idx:end_idx]

            # Check for single-word fillers using compiled regex
            for category, patterns in self.pattern_regexes.items():
                if isinstance(patterns, dict) and patterns["single"].search(word):
                    if self._validate_filler_context(word, context):
                        fillers.append(
                            self._create_filler_entry(word_info, category, context)
                        )
                # Check for compound fillers
                if i < len(word_timings) - 1:
                    compound = f"{word} {word_timings[i+1]['word']}".lower()
                    if isinstance(patterns, dict) and patterns["compound"].search(
                        compound
                    ):
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
            "event_type": "filler",
            "filler_type": category,
            "compound": compound,
            "confidence": word_info["confidence"],
            "context": " ".join(w["word"] for w in context),
        }

    def _enhanced_repetition_detection(self, word_timings: List[Dict]) -> List[Dict]:
        """Improved repetition detection with context-aware analysis."""
        repetitions = []
        i = 0
        while i < len(word_timings) - 1:
            current_word = word_timings[i]["word"].lower()
            pattern = []
            pattern_start = word_timings[i]["start"]
            j = i
            # Accumulate repetitions within a short window
            while j < len(word_timings) and j < i + 5:
                next_word = word_timings[j]["word"].lower()
                if current_word == next_word or self._is_stutter_pattern(
                    current_word, next_word
                ):
                    pattern.append(
                        {
                            "word": next_word,
                            "start": word_timings[j]["start"],
                            "end": word_timings[j]["end"],
                        }
                    )
                    j += 1
                else:
                    break
            if len(pattern) > 1:
                repetitions.append(
                    {
                        "word": current_word,
                        "pattern": pattern,
                        "count": len(pattern),
                        "start": pattern_start,
                        "end": pattern[-1]["end"],
                        "event_type": "repetition",
                        "repetition_type": self._classify_repetition_type(pattern),
                        "confidence": np.mean(
                            [w.get("confidence", 0) for w in pattern]
                        ),
                    }
                )
                i = j
            else:
                i += 1
        return repetitions

    def _is_stutter_pattern(self, word1: str, word2: str) -> bool:
        """Check if two words form a stutter pattern."""
        word1 = word1.strip(string.punctuation)
        word2 = word2.strip(string.punctuation)
        if word1 == word2:
            return True
        if len(word1) >= 2 and len(word2) >= 2:
            if word1.startswith(word2[:2]) or word2.startswith(word1[:2]):
                return True
        stutter_patterns = [
            r"([a-z])\1+",  # Repeated letters
            r"([a-z]{1,2})-\1",  # Word part repetitions
            r"([a-z]{1,2})-([a-z]{1,2})",  # Partial word breaks
        ]
        for pattern in stutter_patterns:
            if re.search(pattern, word1) or re.search(pattern, word2):
                return True
        return False

    def _classify_repetition_type(self, pattern: List[Dict]) -> str:
        """Classify the type of repetition pattern."""
        words = [p["word"] for p in pattern]
        if all(w == words[0] for w in words):
            return "exact_repetition"
        if all(len(w) >= 2 and w.startswith(words[0][:2]) for w in words):
            return "partial_repetition"
        if any(re.search(r"([a-z])\1{2,}", w) for w in words):
            return "prolongation"
        return "block"

    def _validate_filler_context(self, word: str, context: List[Dict]) -> bool:
        """Validate if word is used as filler based on context."""
        context_words = [w["word"].lower() for w in context]
        if any(word == p for p in self.speech_patterns["hesitation"]["single"]):
            return True
        if word in self.speech_patterns["discourse"]["single"]:
            try:
                index = context_words.index(word)
                prev_words = context_words[:index]
                next_words = context_words[index + 1 :]
                if not self._is_grammatical_usage(word, prev_words, next_words):
                    return True
            except ValueError:
                pass
        return False

    def _is_grammatical_usage(
        self, word: str, prev_words: List[str], next_words: List[str]
    ) -> bool:
        """Check if word is used grammatically in context."""
        try:
            context_text = " ".join(prev_words + [word] + next_words)
            doc = nlp(context_text)
            for token in doc:
                if token.text.lower() == word.lower():
                    if token.dep_ in ["discourse", "intj"]:
                        return False
                    return any(token in chunk for chunk in doc.noun_chunks)
        except Exception:
            pass
        return True

    def _analyze_grammar(self, text: str) -> List[Dict]:
        """Perform comprehensive grammar analysis using spaCy and NLTK."""
        errors = []
        try:
            doc = nlp(text)
            for sent in doc.sents:
                self._check_subject_verb_agreement(sent, errors)
                self._check_article_usage(sent, errors)
                self._check_verb_tense(sent, errors)
            sentences = sent_tokenize(text)
            for sent in sentences:
                tokens = word_tokenize(sent)
                tagged = nltk.pos_tag(tokens)
                self._check_double_negatives(tagged, errors)
                self._check_word_order(tagged, errors)
        except Exception as e:
            logger.error(f"Error in grammar analysis: {e}")
        return errors

    def _check_subject_verb_agreement(self, sent, errors: List[Dict]) -> None:
        try:
            for token in sent:
                if token.dep_ == "nsubj":
                    subject = token
                    verb = token.head
                    if subject.tag_.startswith("NN") and verb.tag_.startswith("VB"):
                        if subject.tag_ == "NNS" and verb.tag_ == "VBZ":
                            errors.append(
                                {
                                    "text": f"{subject.text} {verb.text}",
                                    "type": "subject-verb disagreement",
                                    "start_pos": subject.idx,
                                    "end_pos": verb.idx + len(verb.text),
                                }
                            )
        except Exception as e:
            logger.error(f"Error checking subject-verb agreement: {e}")

    def _check_article_usage(self, sent, errors: List[Dict]) -> None:
        try:
            for token in sent:
                if token.text.lower() in ["a", "an"]:
                    next_token = token.nbor() if token.i + 1 < len(sent) else None
                    if (
                        next_token
                        and token.text.lower() == "a"
                        and next_token.text[0].lower() in "aeiou"
                    ):
                        errors.append(
                            {
                                "text": f"{token.text} {next_token.text}",
                                "type": "incorrect article",
                                "start_pos": token.idx,
                                "end_pos": next_token.idx + len(next_token.text),
                            }
                        )
        except Exception as e:
            logger.error(f"Error checking article usage: {e}")

    def _check_verb_tense(self, sent, errors: List[Dict]) -> None:
        try:
            verbs = [token for token in sent if token.pos_ == "VERB"]
            if len(verbs) > 1:
                main_tense = verbs[0].tag_
                for verb in verbs[1:]:
                    if verb.tag_ != main_tense:
                        errors.append(
                            {
                                "text": f"{verbs[0].text}...{verb.text}",
                                "type": "inconsistent verb tense",
                                "start_pos": verbs[0].idx,
                                "end_pos": verb.idx + len(verb.text),
                            }
                        )
        except Exception as e:
            logger.error(f"Error checking verb tense: {e}")

    def _check_double_negatives(self, tagged_tokens, errors: List[Dict]) -> None:
        try:
            negatives = ["not", "n't", "no", "never", "none", "nothing", "nowhere"]
            neg_positions = [
                i
                for i, (word, _) in enumerate(tagged_tokens)
                if word.lower() in negatives
            ]
            for i in range(len(neg_positions) - 1):
                if neg_positions[i + 1] - neg_positions[i] < 5:
                    errors.append(
                        {
                            "text": " ".join(
                                word
                                for word, _ in tagged_tokens[
                                    neg_positions[i] : neg_positions[i + 1] + 1
                                ]
                            ),
                            "type": "double negative",
                            "start_pos": neg_positions[i],
                            "end_pos": neg_positions[i + 1],
                        }
                    )
        except Exception as e:
            logger.error(f"Error checking double negatives: {e}")

    def _check_word_order(self, tagged_tokens, errors: List[Dict]) -> None:
        try:
            for i, (word, tag) in enumerate(tagged_tokens[:-1]):
                next_word, next_tag = tagged_tokens[i + 1]
                if tag == "JJ" and next_tag.startswith("NN"):
                    continue
                elif tag.startswith("NN") and next_tag == "JJ":
                    errors.append(
                        {
                            "text": f"{word} {next_word}",
                            "type": "incorrect word order",
                            "start_pos": i,
                            "end_pos": i + 2,
                        }
                    )
        except Exception as e:
            logger.error(f"Error checking word order: {e}")

    def save_all_formats(self, result: TranscriptionResult, output_dir: Path) -> None:
        """Save analysis results in multiple formats."""
        try:
            self._save_txt(result, output_dir / "transcription.txt")
            self._save_vtt(result, output_dir / "transcription.vtt")
            self._save_textgrid(result, output_dir / "transcription.TextGrid")
            self._save_analysis_json(result, output_dir / "analysis.json")
            self._save_summary_report(result, output_dir / "summary_report.txt")
            logger.info(f"All analysis files saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving analysis files: {e}")
            raise

    def _save_analysis_json(
        self, result: TranscriptionResult, output_path: Path
    ) -> None:
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
                "grammar_errors": {
                    "count": len(result.grammar_errors),
                    "details": result.grammar_errors,
                },
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving analysis JSON: {e}")
            raise

    def _save_textgrid(self, result: TranscriptionResult, output_path: Path) -> None:
        try:
            textgrid_content = 'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
            textgrid_content += f"xmin = 0\nxmax = {result.duration}\n"
            textgrid_content += "tiers? <exists>\nsize = 3\nitem []:\n"
            textgrid_content += self._create_textgrid_tier(
                "words", result.word_timings, 1
            )
            textgrid_content += self._create_textgrid_tier("fillers", result.fillers, 2)
            textgrid_content += self._create_textgrid_tier(
                "repetitions", result.repetitions, 3
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(textgrid_content)
        except Exception as e:
            logger.error(f"Error saving TextGrid: {e}")
            raise

    def _create_textgrid_tier(self, name: str, items: List[Dict], tier_num: int) -> str:
        content = f"    item [{tier_num}]:\n"
        content += f'        class = "IntervalTier"\n'
        content += f'        name = "{name}"\n'
        content += "        xmin = 0\n"
        if not items:
            content += "        xmax = 0\n"
            content += "        intervals: size = 0\n"
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
                text = f"REP:{','.join(item['pattern'])}"
            else:
                text = ""
            content += f'            text = "{text}"\n'
        return content

    def _calculate_speech_rate(
        self, word_timings: List[Dict], duration: float
    ) -> float:
        if duration <= 0:
            return 0.0
        content_words = [
            w
            for w in word_timings
            if not w.get("is_partial") and not self._is_filler(w["word"])
        ]
        return len(content_words) / (duration / 60)

    def _save_vtt(self, result: TranscriptionResult, output_path: Path) -> None:
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

    def _format_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _save_txt(self, result: TranscriptionResult, output_path: Path) -> None:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("SPEECH ANALYSIS TRANSCRIPT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Duration: {result.duration:.2f} seconds\n")
                f.write(f"Speech Rate: {result.speech_rate:.1f} words per minute\n")
                f.write(f"Language Score: {result.language_score:.2f}/1.0\n")
                f.write(f"Overall Confidence: {result.confidence:.2f}\n\n")
                f.write("FULL TRANSCRIPTION:\n")
                f.write("-" * 20 + "\n")
                f.write(result.text + "\n\n")
                f.write("TIMESTAMPED SEGMENTS:\n")
                f.write("-" * 20 + "\n")
                for segment in result.segments:
                    start = self._format_timestamp(segment["start"])
                    end = self._format_timestamp(segment["end"])
                    f.write(f"[{start} --> {end}] {segment['text']}\n")
                self._write_analysis_section(f, result)
        except Exception as e:
            logger.error(f"Error saving TXT: {e}")
            raise

    def _write_analysis_section(self, file, result: TranscriptionResult) -> None:
        file.write("\nDETAILED ANALYSIS:\n")
        file.write("-" * 20 + "\n\n")
        file.write("Filler Words:\n")
        for filler in result.fillers:
            start = self._format_timestamp(filler["start"])
            file.write(
                f"- '{filler['word']}' at {start} ({filler.get('filler_type', '')})\n"
            )
        file.write("\nRepetitions:\n")
        for rep in result.repetitions:
            start = self._format_timestamp(rep["start"])
            file.write(
                f"- '{rep['word']}' repeated {rep.get('count', 0)} times at {start}\n"
            )
        file.write("\nGrammar Issues:\n")
        for error in result.grammar_errors:
            file.write(f"- {error['text']}: {error['type']}\n")

    def _save_summary_report(
        self, result: TranscriptionResult, output_path: Path
    ) -> None:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("SPEECH ANALYSIS SUMMARY\n")
                f.write("=" * 30 + "\n\n")
                f.write("Key Metrics:\n")
                f.write(f"- Duration: {result.duration:.2f} seconds\n")
                f.write(f"- Speech Rate: {result.speech_rate:.1f} words/minute\n")
                f.write(f"- Language Score: {result.language_score:.2f}/1.0\n")
                f.write(f"- Confidence: {result.confidence:.2f}/1.0\n\n")
                f.write("Statistics:\n")
                f.write(f"- Total Words: {len(result.word_timings)}\n")
                f.write(f"- Filler Words: {len(result.fillers)}\n")
                f.write(f"- Repetitions: {len(result.repetitions)}\n")
                f.write(f"- Grammar Errors: {len(result.grammar_errors)}\n")
        except Exception as e:
            logger.error(f"Error saving summary report: {e}")
            raise

    def _is_filler(self, word: str) -> bool:
        """Check if word is considered a filler."""
        word = word.lower()
        for patterns in self.speech_patterns.values():
            if isinstance(patterns, dict):
                if word in patterns["single"] or word in patterns["compound"]:
                    return True
            elif word in patterns:
                return True
        return False

    def _calculate_language_score(self, text: str) -> float:
        """Calculate overall language quality score."""
        try:
            doc = nlp(text)
            grammar_score = self._calculate_grammar_score(doc)
            fluency_score = self._calculate_fluency_score(doc)
            complexity_score = self._calculate_complexity_score(doc)
            return grammar_score * 0.4 + fluency_score * 0.3 + complexity_score * 0.3
        except Exception:
            return 0.0

    def _calculate_grammar_score(self, doc) -> float:
        error_count = len([token for token in doc if token.dep_ == "ROOT"])
        return max(0, 1 - (error_count / len(doc)))

    def _calculate_fluency_score(self, doc) -> float:
        filler_count = len(
            [
                token
                for token in doc
                if token.text.lower() in self.speech_patterns["hesitation"]["single"]
            ]
        )
        return max(0, 1 - (filler_count / len(doc)))

    def _calculate_complexity_score(self, doc) -> float:
        sent_lengths = [len(sent) for sent in doc.sents]
        if not sent_lengths:
            return 0.0
        avg_length = sum(sent_lengths) / len(sent_lengths)
        return min(1.0, avg_length / 20)


# Example usage
if __name__ == "__main__":
    try:
        analyzer = TranscriptionAnalyzer(model_size="large")
        audio_data = np.zeros(16000)  # Dummy audio (replace with actual audio data)
        sample_rate = 16000
        output_dir = Path("transcription_output")
        result = analyzer.analyze_audio(audio_data, sample_rate, output_dir)
        print(f"Analysis complete. Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error in transcription analysis: {e}")
