import os
import shutil
import threading
import time
from collections import deque
from typing import Callable, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

WHISPER_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


# ---------------------------------------------------------------------------
# ASREngine
# ---------------------------------------------------------------------------


class ASREngine:
    """Loads a Whisper model and transcribes speech chunks.

    Owns the audio buffer, VAD state machine, silence timer thread,
    and chunk-split logic. Has zero ROS2 dependencies.

    Communicates outward via one callback:
      - on_transcript_ready(transcript, speaker_id, language_code):
          Called after a successful transcription. The node stamps and
          publishes the result.
    """

    # Minimum audio duration in seconds before attempting transcription
    _MIN_TRANSCRIPTION_DURATION = 0.01
    # Rolling audio buffer kept in memory (seconds)
    _AUDIO_BUFFER_DURATION = 35.0
    # Rolling speaker timeline kept in memory (seconds)
    _SPEAKER_TIMELINE_DURATION = 120.0

    def __init__(
        self,
        *,
        model_size: str,
        compute_type: str,
        language: str,
        use_batched_inference: bool,
        batch_size: int,
        vad_threshold: float,
        min_silence_duration: float,
        max_chunk_duration: float,
        silence_detection_threshold: float,
        pre_buffer_duration: float,
        diarization_offset: float = 0.0,
        min_speaker_run_tokens: int = 2,
        min_speaker_run_duration: float = 0.4,
        snap_splits_to_sentences: bool = True,
        speaker_interval_tolerance: float = 3.0,
        min_speaker_chunk_duration: float = 0.3,
        weights_dir: str,
        on_transcript_ready: Callable[[str, str, str, int, int, float], None],
        logger,
    ):
        self._logger = logger
        self.language = language
        self.use_batched_inference = use_batched_inference
        self.batch_size = batch_size
        self.vad_threshold = vad_threshold
        self.min_silence_duration = min_silence_duration
        self.max_chunk_duration = max_chunk_duration
        self.silence_detection_threshold = silence_detection_threshold
        self.pre_buffer_duration = pre_buffer_duration
        self.diarization_offset = diarization_offset
        self.min_speaker_run_tokens = min_speaker_run_tokens
        self.min_speaker_run_duration = min_speaker_run_duration
        self.snap_splits_to_sentences = snap_splits_to_sentences
        self.speaker_interval_tolerance = speaker_interval_tolerance
        self.min_speaker_chunk_duration = min_speaker_chunk_duration
        self._on_transcript_ready = on_transcript_ready

        self.sample_rate: Optional[int] = None
        self.speaker_id: Optional[str] = None

        # Speaker timeline: timestamped "who was speaking when" segments fed by
        # SpeechActivityDetection. Lets a finished audio chunk be attributed to
        # the speaker that was actually active during the chunk instead of
        # whichever speaker happened to be reported last (diarization runs
        # asynchronously, 1-3s behind the audio).
        # Each entry: {"start": float, "end": float, "speaker": str, "active": bool}
        self.speaker_timeline: deque = deque()
        self._speaker_lock = threading.RLock()

        # Audio buffer: deque of {"audio": np.ndarray, "timestamp": float}
        self.audio_buffer: deque = deque()
        self._buffer_lock = threading.RLock()

        # VAD state machine
        self.vad_state: bool = False
        self.last_vad_change_time: float = 0.0
        self.speech_start_time: float = 0.0
        self.last_silence_time: float = 0.0
        self.speech_interrupted: bool = False
        self._speech_cancelled = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        # Set once a silence timer commits its segment to transcription, so a
        # resumed utterance after that point starts a fresh segment instead of
        # being merged into the one already dispatched to Whisper.
        self._segment_dispatched: bool = False
        self._segment_lock = threading.Lock()
        self.should_stop: bool = False
        # Whisper calls are serialised: a speaker-change split and the silence
        # timer can both want to transcribe at the same moment.
        self._transcribe_lock = threading.Lock()

        # Load model
        self.model_size = model_size
        self.model, self.batched_model = self._load_model(model_size, compute_type, weights_dir)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def validate_model_size(model_size: str) -> str:
        """Raise ValueError if model_size is unknown, else return the HF repo id."""
        if model_size not in WHISPER_MODELS:
            raise ValueError(
                f"Invalid model_size: '{model_size}'. Available: {list(WHISPER_MODELS.keys())}"
            )
        return WHISPER_MODELS[model_size]

    def _run_transcription(
        self, audio_data: np.ndarray, language: str
    ) -> tuple[List[dict], str]:
        """Factory hook used by selectable ASR backends.

        Runs the model on one chunk and returns (segments, detected_language),
        where segments has the structure _collect_segments() produces. Word
        timings are required: _group_segments_by_speaker() uses them to split a
        chunk across speakers.
        """
        if self.use_batched_inference and self.batched_model:
            segments, info = self.batched_model.transcribe(
                audio_data,
                batch_size=self.batch_size,
                vad_filter=False,  # external VAD already gates audio; skip redundant internal pass
                word_timestamps=True,  # needed to split a chunk across speakers
                language=language,
            )
        else:
            segments, info = self.model.transcribe(
                audio_data,
                vad_filter=False,  # external VAD already gates audio; skip redundant internal pass
                word_timestamps=True,  # needed to split a chunk across speakers
                language=language,
            )

        collected = self._collect_segments(segments)
        detected = info.language if hasattr(info, "language") else language
        return collected, detected

    def _load_model(self, model_size: str, compute_type: str, weights_dir: str):
        # Optional backend dependencies must not enter the Parakeet process.
        import ctranslate2
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        self._logger.info(f"Using device on ASR: {device}")

        effective_compute_type = compute_type
        if device == "cpu" and "float16" in compute_type:
            # CTranslate2 on CPU does not support efficient float16 inference.
            self._logger.warn(
                f"compute_type '{compute_type}' is not supported on CPU. Falling back to 'float32'."
            )
            effective_compute_type = "float32"

        os.makedirs(weights_dir, exist_ok=True)

        model_dir_name = "models--" + WHISPER_MODELS[model_size].replace("/", "--")
        resolved_path = self._resolve_local_snapshot(weights_dir, model_dir_name)

        def _build_model(selected_compute_type: str):
            if resolved_path:
                self._logger.info(f"Using local snapshot: {resolved_path}")
                return WhisperModel(
                    resolved_path,
                    device=device,
                    compute_type=selected_compute_type,
                )

            self._logger.info(
                f"No valid local snapshot found — downloading '{model_size}' to {weights_dir}"
            )
            return WhisperModel(
                model_size,
                device=device,
                compute_type=selected_compute_type,
                download_root=weights_dir,
            )

        try:
            model = _build_model(effective_compute_type)
        except ValueError as e:
            fallback_compute_type = "float32"
            self._logger.warn(
                f"ASR model load failed with compute_type '{effective_compute_type}': {e}. "
                f"Retrying with '{fallback_compute_type}'."
            )
            model = _build_model(fallback_compute_type)
            effective_compute_type = fallback_compute_type

        device_label = "GPU" if device == "cuda" else "CPU"
        self._logger.info(
            f"Faster-Whisper model '{model_size}' loaded on {device_label} "
            f"with compute_type '{effective_compute_type}'."
        )

        batched_model = (
            BatchedInferencePipeline(model=model) if self.use_batched_inference else None
        )
        return model, batched_model

    def _resolve_local_snapshot(self, weights_dir: str, model_dir_name: str) -> Optional[str]:
        """Walk weights_dir looking for a complete snapshot (has model.bin).

        Returns the path to the snapshot directory, or None if not found.
        """
        candidate = os.path.join(weights_dir, model_dir_name)
        if not os.path.exists(candidate):
            return None

        for root, _, files in os.walk(candidate):
            if "model.bin" in files:
                return root

        # Snapshot exists but is incomplete — remove and re-download
        self._logger.warn(f"Incomplete model snapshot at {candidate} (no model.bin). Removing...")
        try:
            shutil.rmtree(candidate)
        except Exception as e:
            self._logger.error(f"Failed to remove incomplete snapshot: {e}")
        return None

    def set_sample_rate(self, sample_rate: int) -> None:
        """Called once on the first audio message."""
        self.sample_rate = sample_rate

    def push_audio(self, audio_data: np.ndarray) -> None:
        """Append a new chunk to the rolling audio buffer."""
        current_time = time.time()
        with self._buffer_lock:
            self.audio_buffer.append({"audio": audio_data, "timestamp": current_time})
            cutoff = current_time - self._AUDIO_BUFFER_DURATION
            while self.audio_buffer and self.audio_buffer[0]["timestamp"] < cutoff:
                self.audio_buffer.popleft()

    def update_vad(self, probability: float) -> None:
        """Update VAD state from the latest probability. Triggers processing threads."""
        current_time = time.time()
        new_state = probability > self.vad_threshold

        if new_state != self.vad_state:
            self._logger.debug(f"VAD state changed: {self.vad_state} -> {new_state}")
            if new_state:
                self._speech_cancelled.set()
                with self._segment_lock:
                    dispatched = self._segment_dispatched
                    self._segment_dispatched = False
                if self.speech_start_time == 0 or dispatched:
                    self.speech_start_time = current_time
                    self._logger.debug("Speech started.")
                else:
                    self._logger.debug("Speech continued from previous segment.")
                self.speech_interrupted = False
            else:
                self.last_silence_time = current_time
                self._logger.debug("Speech ended, starting silence timer.")
                self._speech_cancelled.clear()
                # Always spawn a fresh timer thread. Whisper calls are already
                # serialised by _transcribe_lock; gating this on the previous
                # thread's aliveness blocked a new segment's silence timer
                # from ever starting while the previous segment was still
                # inside its (slow) Whisper call, silently dropping it.
                self._processing_thread = threading.Thread(
                    target=self._process_speech_end, daemon=True
                )
                self._processing_thread.start()

        self.vad_state = new_state
        self.last_vad_change_time = current_time

        # Force chunk split on long speech
        if self.vad_state and self.speech_start_time > 0:
            if current_time - self.speech_start_time >= self.max_chunk_duration:
                self._logger.debug(
                    f"Speech duration exceeded {self.max_chunk_duration}s — forcing split."
                )
                self._force_chunk_split()

    def update_speaker(self, speaker_id: Optional[str], active: bool = True) -> None:
        """Called on every SpeechActivityDetection message.

        Keeps a hold timeline (a speaker stays current until a different id is
        reported) and, importantly, when the speaker changes *while speech is
        still going* it flushes the audio accumulated so far. Without that the
        whole exchange would only be published once silence finally arrives.
        """
        self.speaker_id = speaker_id
        if not speaker_id:
            return

        # Signed correction applied to diarization event arrival times so they
        # line up with the audio they describe.
        now = time.time() + self.diarization_offset
        previous_speaker: Optional[str] = None
        with self._speaker_lock:
            if self.speaker_timeline:
                last = self.speaker_timeline[-1]
                if last["speaker"] == speaker_id:
                    last["end"] = now
                    return
                previous_speaker = last["speaker"]
                last["end"] = now

            self.speaker_timeline.append(
                {"start": now, "end": now, "speaker": speaker_id, "active": bool(active)}
            )

            cutoff = now - self._SPEAKER_TIMELINE_DURATION
            while self.speaker_timeline and self.speaker_timeline[0]["end"] < cutoff:
                self.speaker_timeline.popleft()

        if previous_speaker is not None and previous_speaker != speaker_id:
            self._flush_chunk_on_speaker_change(now)

    def _flush_chunk_on_speaker_change(self, change_time: float) -> None:
        """Publish the utterance built so far, then keep accumulating.

        Called the moment diarization reports a different speaker mid-speech,
        so the previous speaker's transcript is published immediately instead
        of waiting for the end of the whole multi-speaker stretch.
        """
        if not self.vad_state or self.speech_start_time <= 0:
            return
        if self.sample_rate is None:
            return

        with self._buffer_lock:
            audio_data, start_time, stop_time = self._extract_audio_data(change_time)
        if audio_data is None:
            return
        if len(audio_data) < int(self.sample_rate * self.min_speaker_chunk_duration):
            return

        # Everything before the change belongs to the previous speaker; carry
        # on accumulating from the change point.
        self.speech_start_time = change_time
        threading.Thread(
            target=self._transcribe_with_data,
            args=(audio_data, start_time, stop_time),
            kwargs={"reset_timing": False},
            daemon=True,
        ).start()

    def _resolve_speaker_for_interval(
        self, start_time: Optional[float], end_time: Optional[float]
    ) -> str:
        """Return the single most likely speaker for a chunk of audio.

        Picks the speaker that was current for most of the chunk, then holds the
        last speaker reported before it, then the nearest one.
        """
        fallback = self.speaker_id or "unknown"
        if start_time is None or end_time is None or end_time < start_time:
            return fallback

        with self._speaker_lock:
            segments = list(self.speaker_timeline)
        if not segments:
            return fallback

        best_speaker, _best_overlap = self._max_overlap_speaker(segments, start_time, end_time)
        if best_speaker is not None:
            return best_speaker

        # Nothing overlapped: hold the last speaker reported before the chunk
        # instead of giving up (diarization lags the audio by ~1s).
        candidates = [s for s in segments if s["start"] <= end_time]
        if candidates:
            return candidates[-1]["speaker"]

        nearest = min(
            segments,
            key=lambda s: self._interval_distance(s, start_time, end_time),
        )
        if self._interval_distance(nearest, start_time, end_time) <= self.speaker_interval_tolerance:
            return nearest["speaker"]
        return fallback

    @staticmethod
    def _max_overlap_speaker(segments, start_time: float, end_time: float):
        """Return (speaker, overlap) of the segment overlapping most; else (None, 0)."""
        best_speaker: Optional[str] = None
        best_overlap = 0.0
        for seg in segments:
            overlap = min(end_time, seg["end"]) - max(start_time, seg["start"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, seg["speaker"]
        return best_speaker, best_overlap

    @staticmethod
    def _interval_distance(segment, start_time: float, end_time: float) -> float:
        if segment["end"] < start_time:
            return start_time - segment["end"]
        if segment["start"] > end_time:
            return segment["start"] - end_time
        return 0.0

    def _timeline_summary(self, start_time: Optional[float], end_time: Optional[float]) -> str:
        """Human-readable current-speaker timeline relative to a chunk (for tuning)."""
        if start_time is None or end_time is None:
            return ""
        with self._speaker_lock:
            segments = list(self.speaker_timeline)

        parts = []
        for seg in segments:
            if seg["end"] < start_time or seg["start"] > end_time:
                continue
            parts.append(
                f"{seg['speaker']}"
                f"[{seg['start'] - start_time:+.2f},{seg['end'] - start_time:+.2f}]"
            )
        return " ".join(parts)

    def stop(self) -> None:
        self.should_stop = True
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def _detect_language(
        self, audio_data: np.ndarray, allowed_languages: Optional[List[str]] = None
    ) -> str:
        if allowed_languages is None:
            allowed_languages = ["en", "es", "ca"]
        try:
            language, _, all_probs = self.model.detect_language(audio_data)
            filtered = [(lang, prob) for lang, prob in all_probs if lang in allowed_languages]
            self._logger.info(
                f"Detected languages (filtered): {[(l, f'{p:.4f}') for l, p in filtered]}"
            )
            if allowed_languages:
                best_lang, best_score = "en", 0.0
                for lang, prob in all_probs:
                    if lang in allowed_languages and prob > best_score:
                        best_score, best_lang = prob, lang
                return best_lang
            return language
        except Exception as e:
            self._logger.warn(f"Language detection failed: {e}. Falling back to 'en'.")
            return "en"

    def _resolve_language(self, audio_data: np.ndarray) -> str:
        if self.language == "auto":
            return self._detect_language(audio_data)
        if "," in self.language:
            allowed = [lang.strip() for lang in self.language.split(",")]
            return self._detect_language(audio_data, allowed)
        if self.model_size.endswith(".en"):
            return "en"
        return self.language

    # ------------------------------------------------------------------
    # VAD state machine internals
    # ------------------------------------------------------------------

    def _process_speech_end(self) -> None:
        """Background thread: wait for silence timeout then transcribe."""
        self._logger.debug(f"Silence timer started, waiting {self.min_silence_duration}s...")
        cancelled = self._speech_cancelled.wait(timeout=self.min_silence_duration)
        if cancelled:
            self._logger.debug("VAD reactivated — cancelling transcription.")
            self.speech_interrupted = False
            return
        if not self.vad_state and self.last_silence_time > 0:
            self._logger.debug("Silence timeout reached — transcribing.")
            expected_start_time = self.speech_start_time
            with self._segment_lock:
                self._segment_dispatched = True
            self._transcribe_speech_chunk(expected_start_time=expected_start_time)
            self.speech_interrupted = False
        else:
            self._logger.debug("VAD state changed before processing — skipping.")

    def _force_chunk_split(self) -> None:
        """Split at the quietest point in the middle 50% of the current chunk."""
        audio_data = None
        split_time = None

        with self._buffer_lock:
            if not self.audio_buffer:
                return

            current_time = time.time()
            chunk_start = self.speech_start_time
            middle_start = chunk_start + (current_time - chunk_start) * 0.25
            middle_end = chunk_start + (current_time - chunk_start) * 0.75

            best_split_time = None
            min_rms = float("inf")
            for chunk in self.audio_buffer:
                t = chunk["timestamp"]
                if middle_start <= t <= middle_end:
                    rms = float(np.sqrt(np.mean(chunk["audio"] ** 2)))
                    if rms < min_rms:
                        min_rms, best_split_time = rms, t

            if best_split_time and min_rms < self.silence_detection_threshold:
                split_time = best_split_time
                self._logger.debug(f"Splitting at quietest point: {best_split_time:.3f}")
            else:
                split_time = current_time
                self._logger.debug("No silence found — splitting at current time.")

            audio_data, start_time, stop_time = self._extract_audio_data(split_time)

        if audio_data is not None:
            self._transcribe_with_data(audio_data, start_time, stop_time)
            self.speech_start_time = split_time

    def _transcribe_speech_chunk(
        self, end_time: Optional[float] = None, expected_start_time: Optional[float] = None
    ) -> None:
        """Extract audio from the buffer then transcribe."""
        if end_time is None:
            end_time = self.last_silence_time if self.last_silence_time > 0 else time.time()
        with self._buffer_lock:
            audio_data, start_time, stop_time = self._extract_audio_data(end_time)
        if audio_data is not None:
            self._transcribe_with_data(
                audio_data, start_time, stop_time, expected_start_time=expected_start_time
            )

    def _extract_audio_data(
        self, end_time: float
    ) -> tuple[Optional[np.ndarray], float, float]:
        """Collect audio chunks between speech_start_time and end_time.

        Returns ``(audio, start_time, end_time)`` where the times are wall-clock
        timestamps covering the extracted samples (used for speaker resolution).
        Must be called inside self._buffer_lock.
        """
        if self.speech_start_time <= 0:
            return None, 0.0, 0.0
        actual_start = self.speech_start_time - self.pre_buffer_duration
        chunks = [
            c["audio"] for c in self.audio_buffer if actual_start <= c["timestamp"] <= end_time
        ]
        if not chunks:
            return None, 0.0, 0.0
        return np.concatenate(chunks), actual_start, end_time

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_segments(segments) -> List[dict]:
        """Materialise Whisper segments (a generator) with their word timings."""
        collected = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            words = getattr(seg, "words", None) or []
            collected.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                    "words": [
                        (float(w.start), float(w.end), w.word)
                        for w in words
                        if getattr(w, "word", "").strip()
                    ],
                }
            )
        return collected

    @staticmethod
    def _append_token(
        groups: List[dict], speaker: str, token: str, start: float, end: float
    ) -> None:
        if groups and groups[-1]["speaker"] == speaker:
            groups[-1]["spans"].append((start, end, token))
        else:
            groups.append({"speaker": speaker, "spans": [(start, end, token)]})

    @staticmethod
    def _span_bounds(span: dict) -> tuple[float, float]:
        return span["spans"][0][0], span["spans"][-1][1]

    @staticmethod
    def _is_sentence_end(token: str) -> bool:
        return token.strip().endswith((".", "!", "?", "…"))

    def _smooth_spans(self, spans: List[dict]) -> List[dict]:
        """Drop speaker runs that are too short to be credible diarization.

        A one-word flicker caused by word timestamp jitter would otherwise split
        an utterance into two speakers.
        """
        if len(spans) < 2:
            return spans

        smoothed: List[dict] = []
        for span in spans:
            if smoothed and span["speaker"] != smoothed[-1]["speaker"]:
                prev = smoothed[-1]
                start, end = self._span_bounds(span)
                too_short = (
                    len(span["spans"]) < self.min_speaker_run_tokens
                    or (end - start) < self.min_speaker_run_duration
                ) and len(prev["spans"]) >= self.min_speaker_run_tokens
                if too_short:
                    # Absorb the flicker into the previous run.
                    prev["spans"].extend(span["spans"])
                    continue
            smoothed.append(span)
        return smoothed

    def _group_tokens_by_sentence(self, tokens: List[tuple]) -> List[dict]:
        """Label whole sentences with their majority speaker.

        Tokens are ``(start, end, text, speaker)``. Sentence units are formed at
        punctuation, so a noisy diarization boundary can never split a sentence
        in half (which reads as a mid-word/mid-phrase bug) nor make one sentence
        absorb part of the next one.
        """
        groups: List[dict] = []
        unit: List[tuple] = []

        sentence_labels: List[tuple] = []
        for token in tokens:
            unit.append(token)
            if self._is_sentence_end(token[2]):
                sentence_labels.append((round(unit[-1][1], 2), self._majority_speaker(unit)))
                self._append_sentence_group(groups, unit)
                unit = []
        if unit:
            sentence_labels.append((round(unit[-1][1], 2), self._majority_speaker(unit)))
            self._append_sentence_group(groups, unit)
        self._last_sentence_labels = sentence_labels
        return groups

    def _append_sentence_group(self, groups: List[dict], unit: List[tuple]) -> None:
        speaker = self._majority_speaker(unit)
        spans = [(start, end, text) for start, end, text, _ in unit]
        if groups and groups[-1]["speaker"] == speaker:
            groups[-1]["spans"].extend(spans)
        else:
            groups.append({"speaker": speaker, "spans": spans})

    @staticmethod
    def _majority_speaker(unit: List[tuple]) -> str:
        """Speaker with the most speaking time inside a sentence unit."""
        durations: dict = {}
        for start, end, _text, speaker in unit:
            durations[speaker] = durations.get(speaker, 0.0) + max(0.0, end - start)
        return max(durations, key=durations.get)

    def _group_segments_by_speaker(
        self,
        segments: List[dict],
        start_time: Optional[float],
        end_time: Optional[float],
    ) -> List[dict]:
        """Split a transcribed chunk into runs of the same speaker.

        Word-level timings are used so that a single Whisper segment containing
        several speakers is still split correctly. Offsets are relative to the
        chunk, so wall-clock times are rebuilt from ``start_time`` before
        consulting the speaker timeline.
        """
        tokens: List[tuple] = []

        for seg in segments:
            words = seg["words"]
            if words and start_time is not None:
                for w_start, w_end, token in words:
                    speaker = self._resolve_speaker_for_interval(
                        start_time + w_start, start_time + w_end
                    )
                    tokens.append((w_start, w_end, token, speaker))
            else:
                if start_time is not None:
                    speaker = self._resolve_speaker_for_interval(
                        start_time + seg["start"], start_time + seg["end"]
                    )
                else:
                    speaker = self.speaker_id or "unknown"
                tokens.append((seg["start"], seg["end"], seg["text"], speaker))

        if self.snap_splits_to_sentences:
            groups = self._group_tokens_by_sentence(tokens)
        else:
            groups = []
            for tok_start, tok_end, token, speaker in tokens:
                self._append_token(groups, speaker, token, tok_start, tok_end)
            groups = self._smooth_spans(groups)

        for group in groups:
            group["text"] = "".join(t for _, _, t in group["spans"]).strip()
            group["start_offset"], group["end_offset"] = self._span_bounds(group)
        return [g for g in groups if g["text"]]

    def _transcribe_with_data(
        self,
        audio_data: np.ndarray,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        reset_timing: bool = True,
        expected_start_time: Optional[float] = None,
    ) -> None:
        """Run Whisper on pre-extracted audio and fire on_transcript_ready."""
        if audio_data is None or len(audio_data) == 0:
            self._logger.warn("Empty audio data — skipping transcription.")
            return

        if self.sample_rate is None:
            self._logger.warn("Sample rate not set — skipping transcription.")
            return

        min_samples = int(self.sample_rate * self._MIN_TRANSCRIPTION_DURATION)
        if len(audio_data) < min_samples:
            self._logger.warn("Audio chunk too short — skipping transcription.")
            return

        duration = len(audio_data) / self.sample_rate
        self._logger.info(f"Transcribing {duration:.2f}s of audio...")
        transcribe_start = time.time()

        transcription_language = self._resolve_language(audio_data)

        try:
            with self._transcribe_lock:
                collected, detected_language = self._run_transcription(
                    audio_data, transcription_language
                )

            transcript = " ".join(seg["text"] for seg in collected).strip()

            if transcript:
                model_processing_ms = int((time.time() - transcribe_start) * 1000)
                if self.last_silence_time > 0:
                    processing_ms = int((time.time() - self.last_silence_time) * 1000)
                else:
                    processing_ms = model_processing_ms
                # A single VAD chunk can contain several speakers (e.g. fast
                # dialogue over background music where no silence is detected).
                # Attribute every Whisper segment with the speaker active during
                # that segment and publish one result per speaker run.
                groups = self._group_segments_by_speaker(collected, start_time, end_time)
                self._logger.info(
                    f"Transcribed {duration:.2f}s -> {len(groups)} speaker group(s)"
                )
                if len(groups) > 1 or duration >= 3.0:
                    self._logger.info(
                        f"Speaker timeline: {self._timeline_summary(start_time, end_time)}"
                    )
                    self._logger.info(
                        "Sentence speakers (end, speaker): "
                        f"{getattr(self, '_last_sentence_labels', [])}"
                    )

                for group in groups:
                    group_text = group["text"]
                    if not group_text:
                        continue
                    group_audio_ms = max(
                        1, int((group["end_offset"] - group["start_offset"]) * 1000)
                    )
                    realtime_factor = (
                        float(processing_ms) / float(group_audio_ms)
                        if group_audio_ms > 0
                        else 0.0
                    )
                    self._logger.info(
                        f"Transcript: '{group_text}' (lang: {detected_language}, "
                        f"speaker: {group['speaker']}, "
                        f"seg={group['start_offset']:.2f}-{group['end_offset']:.2f}s, "
                        f"proc={processing_ms}ms, model={model_processing_ms}ms, "
                        f"audio={group_audio_ms}ms, x{realtime_factor:.2f})"
                    )
                    self._on_transcript_ready(
                        group_text,
                        group["speaker"],
                        detected_language,
                        processing_ms,
                        group_audio_ms,
                        realtime_factor,
                    )
            else:
                self._logger.info("Empty transcript — not publishing.")

        except Exception as e:
            self._logger.error(f"Transcription failed: {e}")

        # Reset speech timing only when this chunk really ended. A speaker-change
        # flush keeps accumulating, so its start time must survive. Also skip the
        # reset if a newer segment has already started since this call began
        # (its own silence-timer thread committed late, e.g. Whisper was still
        # running when the next utterance started) — resetting now would zero
        # out the *new* segment's start time and silently drop it.
        if reset_timing and (
            expected_start_time is None or self.speech_start_time == expected_start_time
        ):
            self.speech_start_time = 0.0
            self.last_silence_time = 0.0
