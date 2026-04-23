import os
import shutil
import threading
import time
from collections import deque
from typing import Callable, List, Optional

import numpy as np
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel

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
        weights_dir: str,
        on_transcript_ready: Callable[[str, str, str], None],
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
        self._on_transcript_ready = on_transcript_ready

        self.sample_rate: Optional[int] = None
        self.speaker_id: Optional[str] = None

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
        self.should_stop: bool = False

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

    def _load_model(self, model_size: str, compute_type: str, weights_dir: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.info(f"Using device on ASR: {device}")

        os.makedirs(weights_dir, exist_ok=True)

        model_dir_name = "models--" + WHISPER_MODELS[model_size].replace("/", "--")
        resolved_path = self._resolve_local_snapshot(weights_dir, model_dir_name)

        if resolved_path:
            self._logger.info(f"Using local snapshot: {resolved_path}")
            model = WhisperModel(resolved_path, device=device, compute_type=compute_type)
        else:
            self._logger.info(
                f"No valid local snapshot found — downloading '{model_size}' to {weights_dir}"
            )
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=weights_dir,
            )

        device_label = "GPU" if device == "cuda" else "CPU"
        self._logger.info(
            f"Faster-Whisper model '{model_size}' loaded on {device_label} "
            f"with compute_type '{compute_type}'."
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
                if self.speech_start_time == 0:
                    self.speech_start_time = current_time
                    self._logger.debug("Speech started.")
                else:
                    self._logger.debug("Speech continued from previous segment.")
                self.speech_interrupted = False
            else:
                self.last_silence_time = current_time
                self._logger.debug("Speech ended, starting silence timer.")
                self._speech_cancelled.clear()
                if self._processing_thread is None or not self._processing_thread.is_alive():
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

    def update_speaker(self, speaker_id: Optional[str]) -> None:
        """Called by the node when SpeechActivityDetection arrives."""
        self.speaker_id = speaker_id

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
            self._transcribe_speech_chunk()
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

            audio_data = self._extract_audio_data(split_time)

        if audio_data is not None:
            self._transcribe_with_data(audio_data)
            self.speech_start_time = split_time

    def _transcribe_speech_chunk(self, end_time: Optional[float] = None) -> None:
        """Extract audio from the buffer then transcribe."""
        if end_time is None:
            end_time = self.last_silence_time if self.last_silence_time > 0 else time.time()
        with self._buffer_lock:
            audio_data = self._extract_audio_data(end_time)
        if audio_data is not None:
            self._transcribe_with_data(audio_data)

    def _extract_audio_data(self, end_time: float) -> Optional[np.ndarray]:
        """Collect audio chunks between speech_start_time and end_time.
        Must be called inside self._buffer_lock.
        """
        if self.speech_start_time <= 0:
            return None
        actual_start = self.speech_start_time - self.pre_buffer_duration
        chunks = [
            c["audio"] for c in self.audio_buffer if actual_start <= c["timestamp"] <= end_time
        ]
        return np.concatenate(chunks) if chunks else None

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe_with_data(self, audio_data: np.ndarray) -> None:
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

        transcription_language = self._resolve_language(audio_data)

        try:
            if self.use_batched_inference and self.batched_model:
                segments, info = self.batched_model.transcribe(
                    audio_data,
                    batch_size=self.batch_size,
                    vad_filter=True,
                    word_timestamps=False,
                    language=transcription_language,
                )
            else:
                segments, info = self.model.transcribe(
                    audio_data,
                    vad_filter=True,
                    word_timestamps=False,
                    language=transcription_language,
                )

            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            detected_language = (
                info.language if hasattr(info, "language") else transcription_language
            )

            if transcript:
                speaker = self.speaker_id or "unknown"
                self._logger.info(
                    f"Transcript: '{transcript}' (lang: {detected_language}, speaker: {speaker})"
                )
                self._on_transcript_ready(transcript, speaker, detected_language)
            else:
                self._logger.info("Empty transcript — not publishing.")

        except Exception as e:
            self._logger.error(f"Transcription failed: {e}")

        # Reset speech timing
        self.speech_start_time = 0.0
        self.last_silence_time = 0.0
