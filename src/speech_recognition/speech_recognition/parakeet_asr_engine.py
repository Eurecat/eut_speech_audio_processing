"""NVIDIA Parakeet TDT backend for the ASR node.

Selected with `asr_backend: "parakeet"` in asr_params.yaml. Everything except
the model is inherited from ASREngine (the Whisper backend): audio buffering,
VAD state machine, speaker-change flushes, per-speaker grouping and the
on_transcript_ready callback that publishes /speech_result. Only model loading
and the per-chunk transcription call are replaced here.

Language note: parakeet-tdt-0.6b-v3 covers 25 European languages. Spanish is
included, Catalan is not. Keep asr_backend: "whisper" where 'ca' matters.
"""

from __future__ import annotations

import logging
import os
from math import gcd
from typing import List

import numpy as np

from speech_recognition.asr_engine import ASREngine

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

    def __init__(self, *, parakeet_model_name: str, **kwargs) -> None:
        self.parakeet_model_name = parakeet_model_name or DEFAULT_PARAKEET_MODEL
        self._device = "cpu"
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
        return model, None

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
        """Parakeet identifies the language itself and has no language-forcing
        argument, so there is nothing to detect up front."""
        return self.language

    def _configured_languages(self) -> List[str]:
        if not self.language or self.language == "auto":
            return []
        return [code.strip() for code in self.language.split(",") if code.strip()]

    def _warn_unsupported_languages(self) -> None:
        unsupported = [c for c in self._configured_languages() if c not in PARAKEET_LANGUAGES]
        if unsupported:
            self._logger.warn(
                f"asr_backend=parakeet does not support {unsupported} "
                f"(model: {self.parakeet_model_name}). Speech in those languages will be "
                "transcribed as one of the 25 supported languages. Use asr_backend: whisper "
                "if that matters."
            )

    def _reported_language(self) -> str:
        """Language code for the published message.

        The model does not expose the language it identified, so report the
        first configured language ("en, es, ca" -> "en").
        """
        languages = self._configured_languages()
        return languages[0] if languages else "unknown"

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _run_transcription(
        self, audio_data: np.ndarray, language: str
    ) -> tuple[List[dict], str]:
        hypotheses = self._transcribe_hypotheses(self._to_model_rate(audio_data))
        if not hypotheses:
            return [], self._reported_language()
        return self._to_segments(hypotheses[0]), self._reported_language()

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
