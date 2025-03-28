"""
transcription_analyzer.py

Enhanced speech transcription and analysis module with high accuracy detection.
Handles transcription, filler detection, repetition detection (with reference-based checking),
and exports in multiple formats.

Modifications:
   - Removed grammar checking.
   - Added reference-based discrepancy check using the provided "grandfather's passage".
   - Incorporated forced alignment for phoneme-level analysis (dummy implementation provided).
"""

import difflib
import numpy as np
import whisper
import torch
import re
import json
import spacy
import nltk
from pathlib import Path
from dataclasses import dataclass
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
except Exception as e:
    print(f"Warning: {e}")

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Warning: {e}")

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Dummy Forced Alignment Module ---
# In a production system, replace this with a proper forced alignment library.
class ForcedAlignmentModule:
    @staticmethod
    def align(audio_data: np.ndarray, transcript: str, sample_rate: int) -> list:
        """
        Dummy forced alignment: returns a list of phoneme alignments.
        For each word, splits it into letters and assigns arbitrary timings.
        """
        alignments = []
        words = transcript.split()
        total_duration = len(audio_data) / sample_rate
        time_per_word = total_duration / max(len(words), 1)
        current_time = 0.0
        for word in words:
            phonemes = list(word)  # Dummy: each letter as a phoneme
            phoneme_duration = time_per_word / max(len(phonemes), 1)
            phoneme_list = []
            for phon in phonemes:
                phoneme_list.append(
                    {
                        "phoneme": phon,
                        "start": current_time,
                        "end": current_time + phoneme_duration,
                    }
                )
                current_time += phoneme_duration
            alignments.append({"word": word, "phonemes": phoneme_list})
        return alignments


# Use the dummy forced alignment module
forced_alignment_module = ForcedAlignmentModule()

# Grandfather's passage is defined here as a constant.
GRANDFATHERS_PASSAGE = (
    "You wished to know all about my grandfather. Well, he is nearly ninety-three years old. "
    "He dresses himself in an ancient black frock coat, usually minus several buttons; yet he still thinks as swiftly as ever. "
    "A long, flowing beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. "
    "When he speaks his voice is just a bit cracked and quivers a trifle. "
    "Twice each day he plays skillfully and with zest upon our small organ. "
    "Except in the winter when the ooze or snow or ice prevents, he slowly takes a short walk in the open air each day. "
    "We have often urged him to walk more and smoke less, but he always answers, “Banana Oil!” "
    "Grandfather likes to be modern in his language."
)


@dataclass
class TranscriptionResult:
    """
    Container for transcription analysis results.
    Fields related to grammar errors have been removed.
    """

    text: str
    segments: list
    word_timings: list
    fillers: list
    repetitions: list
    confidence: float
    duration: float
    speech_rate: float
    language_score: float
    phoneme_alignments: list  # Forced alignment output


class TranscriptionAnalyzer:
    def __init__(self, model_size: str = "large"):
        """
        Initialize with high-accuracy speech recognition.
        Args:
            model_size: Whisper model size (recommended: "large" for best accuracy)
        """
        try:
            self.model = whisper.load_model(model_size)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            # Define filler and repetition patterns as needed.
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

    def transcribe_with_enhanced_detection(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> TranscriptionResult:
        """
        Perform high-accuracy transcription with enhanced detection.
        """
        try:
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
            segments = self._post_process_segments(result["segments"])
            word_timings = self._extract_enhanced_word_timings(segments)
            fillers = self._detect_fillers_with_context(word_timings)
            repetitions = self._enhanced_repetition_detection(word_timings)
            phoneme_alignments = forced_alignment_module.align(
                audio_data, result["text"], sample_rate
            )
            confidence = np.mean([seg.get("confidence", 0) for seg in segments])
            duration = segments[-1]["end"] if segments else 0
            speech_rate = self._calculate_speech_rate(word_timings, duration)
            language_score = self._calculate_language_score(result["text"])
            return TranscriptionResult(
                text=result["text"],
                segments=segments,
                word_timings=word_timings,
                fillers=fillers,
                repetitions=repetitions,
                confidence=confidence,
                duration=duration,
                speech_rate=speech_rate,
                language_score=language_score,
                phoneme_alignments=phoneme_alignments,
            )
        except Exception as e:
            logger.error(f"Error in enhanced transcription: {e}")
            raise

    def check_transcription_against_reference(self, transcription: str) -> list:
        """
        Compare the transcription against the known grandfather's passage
        to identify potential mis-transcriptions.
        """
        discrepancies = []
        ref_words = GRANDFATHERS_PASSAGE.lower().split()
        trans_words = transcription.lower().split()
        matcher = difflib.SequenceMatcher(None, ref_words, trans_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                discrepancies.append(
                    {
                        "ref_segment": " ".join(ref_words[i1:i2]),
                        "trans_segment": " ".join(trans_words[j1:j2]),
                        "discrepancy_type": tag,
                        "ref_indices": (i1, i2),
                        "trans_indices": (j1, j2),
                    }
                )
        return discrepancies

    def _post_process_segments(self, segments: list) -> list:
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

    def _process_word_info(self, word_info: dict) -> dict:
        if not isinstance(word_info, dict):
            return {}
        return {
            "word": word_info.get("word", "").strip(),
            "start": word_info.get("start", 0),
            "end": word_info.get("end", 0),
            "confidence": word_info.get("confidence", 0),
            "is_partial": bool(re.search(r"-", word_info.get("word", ""))),
        }

    def _extract_enhanced_word_timings(self, segments: list) -> list:
        word_timings = []
        for segment in segments:
            words = segment.get("words", [])
            for word_info in words:
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

    def _detect_fillers_with_context(self, word_timings: list) -> list:
        fillers = []
        window_size = 3
        for i, word_info in enumerate(word_timings):
            word = word_info["word"].lower()
            start_idx = max(0, i - window_size)
            end_idx = min(len(word_timings), i + window_size + 1)
            context = word_timings[start_idx:end_idx]
            for category, patterns in self.pattern_regexes.items():
                if isinstance(patterns, dict) and patterns["single"].search(word):
                    if self._validate_filler_context(word, context):
                        fillers.append(
                            self._create_filler_entry(word_info, category, context)
                        )
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
        word_info: dict,
        category: str,
        context: list,
        end_time: float = None,
        compound: bool = False,
    ) -> dict:
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

    def _enhanced_repetition_detection(self, word_timings: list) -> list:
        repetitions = []
        i = 0
        max_gap = 0.5  # seconds
        while i < len(word_timings) - 1:
            current_word = word_timings[i]["word"].lower()
            pattern = []
            pattern_start = word_timings[i]["start"]
            j = i
            while j < len(word_timings):
                next_word = word_timings[j]["word"].lower()
                if current_word == next_word or self._is_stutter_pattern(
                    current_word, next_word
                ):
                    if j > i:
                        gap = word_timings[j]["start"] - word_timings[j - 1]["end"]
                        if gap > max_gap:
                            break
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
        word1 = word1.strip(".,!?")
        word2 = word2.strip(".,!?")
        if word1 == word2:
            return True
        if (
            len(word1) >= 2
            and len(word2) >= 2
            and (word1.startswith(word2[:2]) or word2.startswith(word1[:2]))
        ):
            return True
        return False

    def _classify_repetition_type(self, pattern: list) -> str:
        words = [p["word"] for p in pattern]
        if all(w == words[0] for w in words):
            return "exact_repetition"
        if all(len(w) >= 2 and w.startswith(words[0][:2]) for w in words):
            return "partial_repetition"
        return "complex_repetition"

    def _validate_filler_context(self, word: str, context: list) -> bool:
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
        self, word: str, prev_words: list, next_words: list
    ) -> bool:
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

    def _calculate_speech_rate(self, word_timings: list, duration: float) -> float:
        if duration <= 0:
            return 0.0
        content_words = [
            w
            for w in word_timings
            if not w.get("is_partial") and not self._is_filler(w["word"])
        ]
        return len(content_words) / (duration / 60)

    def _is_filler(self, word: str) -> bool:
        word = word.lower()
        for patterns in self.speech_patterns.values():
            if isinstance(patterns, dict):
                if word in patterns["single"] or word in patterns["compound"]:
                    return True
            elif word in patterns:
                return True
        return False

    def _calculate_language_score(self, text: str) -> float:
        try:
            doc = nlp(text)
            grammar_score = self._calculate_grammar_score(doc)
            fluency_score = self._calculate_fluency_score(doc)
            complexity_score = self._calculate_complexity_score(doc)
            return grammar_score * 0.4 + fluency_score * 0.3 + complexity_score * 0.3
        except Exception:
            return 0.0

    def _calculate_grammar_score(self, doc) -> float:
        # Simplified grammar score (could be replaced with more detailed rules)
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

    def save_all_formats(self, result: TranscriptionResult, output_dir: Path) -> None:
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
        analysis = {
            "text": result.text,
            "duration": result.segments[-1]["end"] if result.segments else 0,
            "word_count": len(result.word_timings),
            "fillers": {"count": len(result.fillers), "details": result.fillers},
            "repetitions": {
                "count": len(result.repetitions),
                "details": result.repetitions,
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

    def _save_textgrid(self, result: TranscriptionResult, output_path: Path) -> None:
        textgrid_content = 'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
        textgrid_content += f"xmin = 0\nxmax = {result.duration}\n"
        textgrid_content += "tiers? <exists>\nsize = 3\nitem []:\n"
        textgrid_content += self._create_textgrid_tier("words", result.word_timings, 1)
        textgrid_content += self._create_textgrid_tier("fillers", result.fillers, 2)
        textgrid_content += self._create_textgrid_tier(
            "repetitions", result.repetitions, 3
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(textgrid_content)

    def _create_textgrid_tier(self, name: str, items: list, tier_num: int) -> str:
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

    def _save_vtt(self, result: TranscriptionResult, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, segment in enumerate(result.segments):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                f.write(f"{i+1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment['text']}\n\n")

    def _format_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def _save_txt(self, result: TranscriptionResult, output_path: Path) -> None:
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

    def _save_summary_report(
        self, result: TranscriptionResult, output_path: Path
    ) -> None:
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

    # End of TranscriptionAnalyzer


# Example usage
if __name__ == "__main__":
    try:
        analyzer = TranscriptionAnalyzer(model_size="large")
        # Dummy audio for testing (replace with actual audio)
        audio_data = np.zeros(16000)
        sample_rate = 16000
        output_dir = Path("transcription_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = analyzer.transcribe_with_enhanced_detection(audio_data, sample_rate)
        discrepancies = analyzer.check_transcription_against_reference(result.text)
        print("Discrepancies with reference passage:")
        print(discrepancies)
        analyzer.save_all_formats(result, output_dir)
        print(f"Analysis complete. Results saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error in transcription analysis: {e}")
