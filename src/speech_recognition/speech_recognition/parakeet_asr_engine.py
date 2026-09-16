"""NVIDIA Parakeet TDT backend for the ASR node.

Selected with `asr_backend: "parakeet"` in asr_params.yaml. Everything except
the model is inherited from ASREngine (the Whisper backend): audio buffering,
VAD state machine, speaker-change flushes, per-speaker grouping and the
on_transcript_ready callback that publishes /speech_result. Only model loading
and the per-chunk transcription call are replaced here.

Language note: parakeet-tdt-0.6b-v3 covers 25 European languages. Spanish is
included, Catalan is not. Keep asr_backend: "whisper" where 'ca' matters.
The model never reports the language it heard, so a separate LangID model
(language_id.SpokenLanguageIdentifier, AmberNet) picks the published language
code from the same allow-list the Whisper backend uses.
"""

from __future__ import annotations

import logging
import os
from math import gcd
from typing import List, Optional

import numpy as np

from speech_recognition.asr_engine import DEFAULT_DETECTION_LANGUAGES, ASREngine
from speech_recognition.language_id import (
    DEFAULT_LANGUAGE_ID_MODEL,
    LANGUAGE_ID_SAMPLE_RATE,
    SpokenLanguageIdentifier,
)

DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

# Parakeet models are trained on 16 kHz mono audio.
PARAKEET_SAMPLE_RATE = 16000

# 25 languages per the v3 model card. Used to warn about a misconfigured stack,
# not to reject audio: the model does its own language identification.
PARAKEET_LANGUAGES = frozenset(
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()
)


class ParakeetASREngine(ASREngine):
    """ASREngine that transcribes with a NeMo TDT model instead of Whisper."""

    # Audio used to warm up the model once at load time (seconds).
    _WARMUP_DURATION = 1.0
    # Shorter chunks keep the previous language: LangID is unreliable there and
    # AmberNet cannot process less than ~50 ms (seconds).
    _MIN_LANGUAGE_ID_DURATION = 0.25

    def __init__(
        self,
        *,
        parakeet_model_name: str,
        language_id_model: str = DEFAULT_LANGUAGE_ID_MODEL,
        language_id_min_confidence: float = 0.9,
        **kwargs,
    ) -> None:
        self.parakeet_model_name = parakeet_model_name or DEFAULT_PARAKEET_MODEL
        self.language_id_model = language_id_model
        self.language_id_min_confidence = language_id_min_confidence
        self._device = "cpu"
        self._language_id: Optional[SpokenLanguageIdentifier] = None
        self._last_language: Optional[str] = None
        super().__init__(**kwargs)
        self._warn_unsupported_languages()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def validate_model_size(model_size: str) -> str:
        """model_size names a Whisper checkpoint, so Parakeet ignores it.

        Overridden so the node validates uniformly whichever engine it selected.
        """
        return model_size

    def _load_model(self, model_size: str, compute_type: str, weights_dir: str):
        """Load the NeMo model.

        Returns (model, batched_model) for parity with the Whisper backend.
        Parakeet has no batched variant, so use_batched_inference and
        batch_size are ignored.
        """
        import torch

        try:
            # Imported lazily: a Whisper-only image does not need NeMo.
            import nemo.collections.asr as nemo_asr
            from nemo.utils import logging as nemo_logging
        except ImportError as e:
            raise RuntimeError(
                "asr_backend=parakeet needs nemo_toolkit[asr] in this image "
                f"({e}). Rebuild the image or select asr_backend: whisper."
            ) from e

        # NeMo dumps the full training config at WARNING level on every restore.
        nemo_logging.setLevel(logging.ERROR)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._logger.info(f"Using device on ASR: {self._device}")
        if self._device == "cpu":
            self._logger.warn("CUDA not available: Parakeet runs on CPU and will be slow.")

        checkpoint = self._resolve_checkpoint(weights_dir)
        self._logger.info(f"Loading Parakeet model from: {checkpoint}")
        # restore_from() instantiates the concrete class named in the checkpoint
        # config (EncDecRNNTBPEModel for TDT), so no model class is hardcoded.
        model = nemo_asr.models.ASRModel.restore_from(checkpoint, map_location=self._device)
        model.eval()
        self._configure_decoding(model)

        if compute_type != "float32":
            # Half precision gives no speed-up here and dropped punctuation in
            # tests, which breaks sentence-based speaker splitting.
            self._logger.info(f"compute_type '{compute_type}' is Whisper-only; Parakeet runs float32.")
        self.model = model
        self._warm_up()

        self._logger.info(
            f"Parakeet model '{self.parakeet_model_name}' loaded on "
            f"{'GPU' if self._device == 'cuda' else 'CPU'} with compute_type 'float32'."
        )
        self._language_id = self._load_language_id(weights_dir)
        return model, None

    def _load_language_id(self, weights_dir: str) -> Optional[SpokenLanguageIdentifier]:
        """Load the LangID model when there is more than one language to choose from.

        A failure (e.g. no network on first start) is not fatal: the node keeps
        transcribing and publishes the first candidate language instead.
        """
        candidates = self._candidate_languages()
        if len(candidates) < 2:
            self._logger.info(
                f"Single language {candidates}: language identification not needed."
            )
            return None
        if not self.language_id_model:
            self._logger.warn(
                "parakeet_language_id_model is empty: publishing the first configured "
                f"language '{candidates[0]}' for every transcript."
            )
            return None
        try:
            identifier = SpokenLanguageIdentifier(
                model_name=self.language_id_model,
                weights_dir=weights_dir,
                device=self._device,
                logger=self._logger,
            )
            unknown = [code for code in candidates if code not in identifier.labels]
            if unknown:
                self._logger.warn(f"Language identification model does not know {unknown}.")
            identifier.probabilities(
                np.zeros(int(LANGUAGE_ID_SAMPLE_RATE * self._WARMUP_DURATION), dtype=np.float32),
                candidates,
            )
        except Exception as e:
            self._logger.error(
                f"Language identification model '{self.language_id_model}' failed to load: {e}. "
                f"Publishing the first configured language '{candidates[0]}'."
            )
            return None
        self._logger.info(
            f"Language identification '{self.language_id_model}' loaded; choosing from "
            f"{candidates} with min confidence {self.language_id_min_confidence}."
        )
        return identifier

    def _resolve_checkpoint(self, weights_dir: str) -> str:
        """Return a local .nemo path, downloading it into weights_dir if needed.

        parakeet_model_name is either a path to a .nemo file or a Hugging Face
        repo id. Repo downloads share the weights dir and cache layout with
        the Whisper backend (weights/models--<org>--<name>), so a cached model
        loads offline.
        """
        name = self.parakeet_model_name
        if name.endswith(".nemo"):
            if not os.path.isfile(name):
                raise FileNotFoundError(f"Parakeet checkpoint not found: {name}")
            return name

        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        filename = name.split("/")[-1] + ".nemo"
        os.makedirs(weights_dir, exist_ok=True)

        cached = try_to_load_from_cache(repo_id=name, filename=filename, cache_dir=weights_dir)
        if isinstance(cached, str) and os.path.isfile(cached):
            self._logger.info(f"Using local snapshot: {cached}")
            return cached

        self._logger.info(f"No valid local snapshot found — downloading '{name}' to {weights_dir}")
        return hf_hub_download(repo_id=name, filename=filename, cache_dir=weights_dir)

    @staticmethod
    def _configure_decoding(model) -> None:
        """Enable timestamps once instead of on every transcribe() call.

        transcribe(timestamps=True) rebuilds the decoding strategy per call.
        Word timings are required to split a chunk across speakers.
        """
        from omegaconf import open_dict

        with open_dict(model.cfg.decoding):
            model.cfg.decoding.compute_timestamps = True
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)

    def _warm_up(self) -> None:
        """Run one silent chunk so the first real utterance is not delayed.

        The first transcribe() call initialises CUDA kernels and takes ~2 s;
        later chunks of any length take ~100 ms on Jetson Thor.
        """
        silence = np.zeros(int(PARAKEET_SAMPLE_RATE * self._WARMUP_DURATION), dtype=np.float32)
        try:
            self._transcribe_hypotheses(silence)
        except Exception as e:
            self._logger.warn(f"Parakeet warm-up failed (first utterance may be slow): {e}")

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------

    def _resolve_language(self, audio_data: np.ndarray) -> str:
        """Parakeet has no language-forcing argument, so nothing is resolved up
        front; _identify_language() picks the published code per chunk."""
        return self.language

    def _configured_languages(self) -> List[str]:
        if not self.language or self.language == "auto":
            return []
        return [code.strip() for code in self.language.split(",") if code.strip()]

    def _candidate_languages(self) -> List[str]:
        """Languages to choose from, with the same meaning as for Whisper:
        "auto" -> DEFAULT_DETECTION_LANGUAGES, "es" -> only es, "en, es" -> that list."""
        return self._configured_languages() or list(DEFAULT_DETECTION_LANGUAGES)

    def _warn_unsupported_languages(self) -> None:
        unsupported = [c for c in self._configured_languages() if c not in PARAKEET_LANGUAGES]
        if unsupported:
            self._logger.warn(
                f"asr_backend=parakeet does not support {unsupported} "
                f"(model: {self.parakeet_model_name}). Speech in those languages is detected "
                "(language code published) but transcribed as one of the 25 supported "
                "languages. Use asr_backend: whisper if that matters."
            )

    def _identify_language(self, audio: np.ndarray) -> str:
        """Language code for the published message of one 16 kHz chunk.

        LangID picks the most likely candidate language. When its confidence is
        below language_id_min_confidence, or the chunk is too short, the last
        confidently detected language is kept: short replies ("Yeah.") and noisy
        chunks otherwise flip language (offline test: 78% -> 98% correct on the
        English mp3). In the ROS pipeline: 46/46 correct on a Spanish/Catalan/
        English sequence, ~97% on the English mp3.
        Called with _transcribe_lock held, so _last_language updates in order.
        """
        candidates = self._candidate_languages()
        fallback = self._last_language or candidates[0]
        if self._language_id is None or len(candidates) < 2:
            return fallback
        if len(audio) < int(LANGUAGE_ID_SAMPLE_RATE * self._MIN_LANGUAGE_ID_DURATION):
            return fallback

        try:
            probabilities = self._language_id.probabilities(audio, candidates)
        except Exception as e:
            self._logger.warn(f"Language identification failed: {e}. Keeping '{fallback}'.")
            return fallback
        if not probabilities:
            return fallback

        best = max(probabilities, key=probabilities.get)
        self._logger.info(
            "Detected languages (filtered): "
            f"{[(code, f'{p:.4f}') for code, p in probabilities.items()]}"
        )
        if probabilities[best] < self.language_id_min_confidence:
            self._logger.info(
                f"Language '{best}' below min confidence "
                f"({probabilities[best]:.2f} < {self.language_id_min_confidence}); "
                f"keeping '{fallback}'."
            )
            return fallback
        self._last_language = best
        return best

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _run_transcription(
        self, audio_data: np.ndarray, language: str
    ) -> tuple[List[dict], str]:
        audio = self._to_model_rate(audio_data)
        hypotheses = self._transcribe_hypotheses(audio)
        segments = self._to_segments(hypotheses[0]) if hypotheses else []
        if not segments:
            # Nothing is published, so a noise chunk must not change the language.
            return [], self._last_language or self._candidate_languages()[0]
        return segments, self._identify_language(audio)

    def _to_model_rate(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        rate = self.sample_rate or PARAKEET_SAMPLE_RATE
        if rate == PARAKEET_SAMPLE_RATE:
            return audio
        from scipy.signal import resample_poly

        divisor = gcd(PARAKEET_SAMPLE_RATE, rate)
        return resample_poly(audio, PARAKEET_SAMPLE_RATE // divisor, rate // divisor).astype(
            np.float32
        )

    def _transcribe_hypotheses(self, audio: np.ndarray):
        """Transcribe one 16 kHz chunk; return NeMo hypotheses with timestamps in seconds."""
        import torch
        from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs

        with torch.inference_mode():
            hypotheses = self.model.transcribe(
                [audio], batch_size=1, return_hypotheses=True, verbose=False
            )
        if not hypotheses:
            # A chunk of silence can decode to nothing. process_timestamp_outputs()
            # indexes outputs[0] without a length check.
            return []
        return process_timestamp_outputs(
            hypotheses,
            self.model.encoder.subsampling_factor,
            self.model.cfg.preprocessor.window_stride,
        )

    def _to_segments(self, hypothesis) -> List[dict]:
        """Normalise a NeMo hypothesis into _collect_segments()'s structure."""
        text = (getattr(hypothesis, "text", None) or "").strip()
        if not text:
            return []

        stamps = getattr(hypothesis, "timestamp", None)
        if not isinstance(stamps, dict):
            stamps = {}
        words = self._words_from(stamps)
        segments = self._segments_from(stamps, words)
        if segments:
            return segments
        if words:
            return [{"start": words[0][0], "end": words[-1][1], "text": text, "words": words}]

        # No usable timestamps: one segment for the whole chunk. The speaker
        # grouping then attributes all of it to a single speaker, as it does for
        # a Whisper segment without word timings.
        self._logger.warn(
            "Parakeet returned no timestamps; publishing the chunk as one segment."
        )
        return [{"start": 0.0, "end": 0.0, "text": text, "words": []}]

    @staticmethod
    def _words_from(stamps: dict) -> List[tuple]:
        words = []
        for entry in stamps.get("word") or []:
            token = (entry.get("word") or "").strip()
            if not token:
                continue
            # faster-whisper words carry their leading space and the speaker
            # grouper concatenates tokens verbatim. NeMo words do not.
            words.append((float(entry.get("start", 0.0)), float(entry.get("end", 0.0)), " " + token))
        return words

    @staticmethod
    def _segments_from(stamps: dict, words: List[tuple]) -> List[dict]:
        """Use the model's own sentence segments, each with the words inside it."""
        segments = []
        for entry in stamps.get("segment") or []:
            seg_text = (entry.get("segment") or "").strip()
            if not seg_text:
                continue
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", 0.0))
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    "words": [w for w in words if start <= w[0] <= end],
                }
            )
        return segments
