"""NVIDIA Parakeet TDT backend for the ASR node.

Selected with `asr_backend: "parakeet"` in asr_params.yaml. .
Language note: parakeet-tdt-0.6b-v3 covers 25 European languages.
Spanish included but no Catalan.
"""

from __future__ import annotations

import os
import platform
import tempfile
from typing import List, Optional

import numpy as np

from speech_recognition.asr_engine import ASREngine

# 25 languages per the v3 model card. Used to warn about a misconfigured stack,
# not to reject audio — the model does its own language identification.
PARAKEET_LANGUAGES = frozenset(
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()
)


class ParakeetASREngine(ASREngine):
    """ASREngine that transcribes with a NeMo TDT model instead of Whisper."""

    def __init__(
        self,
        *,
        parakeet_model_name: str,
        **kwargs,
    ) -> None:
        self.parakeet_model_name = parakeet_model_name or "nvidia/parakeet-tdt-0.6b-v3"
        self._parakeet_device = "cpu"
        super().__init__(**kwargs)
        self._warn_unsupported_languages()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def validate_model_size(model_size: str) -> str:
        """model_size names a Whisper checkpoint, so it is unused here.

        Overridden so the node can validate uniformly whichever engine class it
        selected, without special-casing the backend.
        """
        return model_size

    @staticmethod
    def _validate_platform(cuda_available: bool) -> None:
        if (platform.machine().lower() not in ("x86_64", "amd64")
                or os.environ.get("ROS_DISTRO") != "jazzy"
                or not cuda_available):
            raise RuntimeError(
                "Parakeet is currently supported only on x86_64 Jazzy with CUDA. "
                "CPU, ARM and ARM Thor are not supported by this deployment. "
                "Select asr_backend: whisper on this platform."
            )

    def _load_model(self, model_size: str, compute_type: str, weights_dir: str):
        """Load the NeMo model. Returns (model, batched_model) for interface
        parity with the Whisper path; Parakeet has no batched variant."""
        import torch

        self._validate_platform(torch.cuda.is_available())

        # Imported lazily so a stack running the Whisper backend neither pays
        # the import cost nor needs nemo_toolkit installed at all.
        import nemo.collections.asr as nemo_asr

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._parakeet_device = device
        self._logger.info(f"Using device on ASR: {device}")

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            # NeMo resolves its Hugging Face downloads through this cache.
            os.environ.setdefault("HF_HOME", weights_dir)

        self._logger.info(f"Loading Parakeet model: {self.parakeet_model_name}")
        model = self._from_pretrained(nemo_asr, device)
        model.eval()
        self._configure_timestamps(model)

        dtype = self._resolve_dtype(compute_type, device, torch)
        if dtype is not None:
            model = model.to(dtype=dtype)

        self._logger.info(
            f"Parakeet model '{self.parakeet_model_name}' loaded on "
            f"{'GPU' if device == 'cuda' else 'CPU'} "
            f"with compute_type '{compute_type if dtype is not None else 'float32'}'."
        )
        return model, None

    @staticmethod
    def _configure_timestamps(model) -> None:
        """Configure the decoder once instead of rebuilding it for every chunk."""
        from omegaconf import open_dict

        with open_dict(model.cfg.decoding):
            model.cfg.decoding.compute_timestamps = True
            model.cfg.decoding.preserve_alignments = True
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)

    def _transcribe_with_timestamps(self, path):
        from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs

        # NeMo 2.4 transcribe(timestamps=True) rebuilds the decoding strategy.
        # Request hypotheses from the configured decoder and apply the same
        # frame-to-seconds conversion that NeMo normally runs afterward.
        output = self.model.transcribe(
            [path], return_hypotheses=True, verbose=False
        )
        if not output:
            # A chunk of silence or noise decodes to nothing. NeMo's
            # process_timestamp_outputs() indexes outputs[0] without a length
            # check, so an empty list raises IndexError there rather than
            # returning no hypotheses.
            self._logger.debug("Parakeet decoded no hypotheses for this chunk.")
            return []
        return process_timestamp_outputs(
            output, self.model.encoder.subsampling_factor,
            self.model.cfg.preprocessor.window_stride,
        )

    def _from_pretrained(self, nemo_asr, device: str):
        """Load the checkpoint, tolerating NeMo's two loading APIs.

        The model card documents ASRModel.from_pretrained(), but on NeMo 3.x
        ASRModel is abstract and that call dies with

            TypeError: Can't instantiate abstract class ASRModel without an
            implementation for abstract methods 'setup_training_data', ...

        because it never resolves the concrete class named in the checkpoint's
        own config (for parakeet-tdt-0.6b-v3: EncDecRNNTBPEModel). Try the
        documented path first so newer/older NeMo keeps working, then fall back
        to the concrete RNNT class.
        """
        try:
            return nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.parakeet_model_name, map_location=device
            )
        except TypeError as e:
            if "abstract" not in str(e):
                raise
            self._logger.warn(
                f"ASRModel.from_pretrained() is abstract on this NeMo "
                f"({type(e).__name__}); retrying with EncDecRNNTBPEModel."
            )

        from nemo.collections.asr.models import EncDecRNNTBPEModel

        return EncDecRNNTBPEModel.from_pretrained(
            model_name=self.parakeet_model_name, map_location=device
        )

    def _resolve_dtype(self, compute_type: str, device: str, torch):
        """Map the shared `compute_type` parameter onto a torch dtype.

        The Whisper path accepts CTranslate2 names (int8, int8_float16, ...)
        with no torch equivalent, so anything unrecognised falls back to
        float32 rather than failing a stack that reuses its existing config.
        """
        if device == "cpu":
            # Half precision on CPU is slower, not faster, for this model.
            return None

        if compute_type in ("float16", "fp16", "int8_float16"):
            # TDT decoding is numerically fragile in pure float16; bfloat16 is
            # the safe half precision on Ampere and newer.
            if torch.cuda.is_bf16_supported():
                self._logger.info(
                    f"compute_type '{compute_type}' mapped to bfloat16 "
                    "(TDT decoding is unstable in float16)."
                )
                return torch.bfloat16
            self._logger.warn(
                f"compute_type '{compute_type}' requested but bfloat16 is "
                "unsupported here; using float32."
            )
            return None

        if compute_type in ("bfloat16", "bf16"):
            return torch.bfloat16

        if compute_type not in ("float32", "fp32"):
            self._logger.warn(
                f"compute_type '{compute_type}' has no Parakeet equivalent; using float32."
            )
        return None

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------

    def _resolve_language(self, audio_data: np.ndarray) -> str:
        """Parakeet identifies the language itself and exposes no
        language-forcing argument, so there is nothing to resolve up front; the
        code it reports comes back from _run_transcription()."""
        return self.language

    def _warn_unsupported_languages(self) -> None:
        if not self.language or self.language == "auto":
            return
        unsupported = [
            code.strip()
            for code in self.language.split(",")
            if code.strip() and code.strip() not in PARAKEET_LANGUAGES
        ]
        if unsupported:
            self._logger.warn(
                f"asr_backend=parakeet does not support {unsupported} "
                f"(model: {self.parakeet_model_name}). Speech in those languages "
                "will be transcribed as one of the 25 supported languages. Use "
                "asr_backend=whisper if that matters."
            )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _run_transcription(
        self, audio_data: np.ndarray, language: str
    ) -> tuple[List[dict], str]:
        import torch

        # NeMo's transcribe() takes file paths, so the in-memory chunk from the
        # VAD buffer has to land on disk for the duration of the call.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            self._write_wav(tmp.name, audio_data)

            with torch.inference_mode():
                output = self._transcribe_with_timestamps(tmp.name)

            return self._to_segments(output), self._detected_language(output, language)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _write_wav(self, path: str, audio: np.ndarray) -> None:
        import soundfile as sf

        sf.write(path, audio, self.sample_rate or 16000)

    def _to_segments(self, output) -> List[dict]:
        """Normalise NeMo output into _collect_segments()'s structure."""
        if not output:
            return []

        hyp = output[0]
        text = self._hyp_text(hyp).strip()
        if not text:
            return []

        stamps = getattr(hyp, "timestamp", None) or {}
        words = self._words_from(stamps)
        segments = self._segments_from(stamps, words, text)

        if segments:
            return segments

        # No usable timestamps: emit one segment covering the whole chunk. The
        # speaker grouping then attributes all of it to a single speaker, which
        # is how a Whisper segment without word timings behaves too.
        self._logger.warn(
            "Parakeet returned no timestamps; publishing the chunk as one segment "
            "(per-speaker splitting is disabled for it)."
        )
        return [{"start": 0.0, "end": 0.0, "text": text, "words": words}]

    @staticmethod
    def _hyp_text(hyp) -> str:
        if isinstance(hyp, str):
            return hyp
        return getattr(hyp, "text", None) or getattr(hyp, "pred_text", "") or ""

    @staticmethod
    def _words_from(stamps) -> List[tuple]:
        if not isinstance(stamps, dict):
            return []
        out = []
        for w in stamps.get("word") or []:
            token = (w.get("word") or w.get("char") or "").strip()
            if not token:
                continue
            # The shared speaker grouper concatenates tokens verbatim, as
            # faster-whisper words already contain their leading whitespace.
            # NeMo words do not, so preserve word boundaries in this adapter.
            out.append((float(w.get("start", 0.0)), float(w.get("end", 0.0)), " " + token))
        return out

    @staticmethod
    def _segments_from(stamps, words: List[tuple], text: str) -> List[dict]:
        """Prefer the model's own segment split; fall back to one segment."""
        if not isinstance(stamps, dict):
            return []

        collected = []
        for seg in stamps.get("segment") or []:
            seg_text = (seg.get("segment") or seg.get("text") or "").strip()
            if not seg_text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            collected.append(
                {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    # Keep only the words inside this segment, so the speaker
                    # grouping sees the same word->segment nesting as Whisper.
                    "words": [w for w in words if start <= w[0] <= end] if words else [],
                }
            )

        if collected:
            return collected

        if words:
            return [{
                "start": words[0][0],
                "end": words[-1][1],
                "text": text,
                "words": words,
            }]
        return []

    def _detected_language(self, output, requested: Optional[str]) -> str:
        """Report the language NeMo identified, if it exposes one."""
        if output:
            hyp = output[0]
            for attr in ("langs", "lang", "language"):
                value = getattr(hyp, attr, None)
                if isinstance(value, str) and value:
                    return value

        # Fall back to the configured language, taking the first of a list like
        # "es,ca" — the published message needs some language code.
        if requested and requested != "auto":
            first = requested.split(",")[0].strip()
            if first:
                return first
        return "unknown"
