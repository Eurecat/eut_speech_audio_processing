"""Spoken language identification for ASR backends that do not report a language.

parakeet-tdt-0.6b-v3 neither outputs nor accepts a language id; NVIDIA
recommends running a separate LangID model next to it (NeMo issues #14799 and
#15097). AmberNet (`langid_ambernet`, VoxLingua107: 107 languages including
Catalan, 29M parameters) runs in ~6 ms per chunk on Jetson Thor.

Measured on FLEURS dev, restricted to en/es/ca: 88% at 1 s, 96% at 2 s, 99% at
3 s of speech. Whisper turbo's own detection (the Whisper backend) scored 82%,
91% and 98%, and only 48% on 1 s of Catalan.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

DEFAULT_LANGUAGE_ID_MODEL = "langid_ambernet"

# Language models expect 16 kHz mono audio.
LANGUAGE_ID_SAMPLE_RATE = 16000


class SpokenLanguageIdentifier:
    """NeMo speaker-label LangID model restricted to an allow-list of languages."""

    def __init__(self, *, model_name: str, weights_dir: str, device: str, logger) -> None:
        import torch
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self._logger = logger
        self._torch = torch
        self._device = device

        checkpoint = self._resolve_checkpoint(model_name, weights_dir, EncDecSpeakerLabelModel)
        self._logger.info(f"Loading language identification model from: {checkpoint}")
        self._model = EncDecSpeakerLabelModel.restore_from(checkpoint, map_location=device)
        self._model.eval()
        self.labels = list(self._model.cfg.train_ds.labels)

    def _resolve_checkpoint(self, model_name: str, weights_dir: str, model_class) -> str:
        """Return a local .nemo path, downloading a pretrained model into weights_dir.

        model_name is a path to a .nemo file or a NeMo pretrained name
        (EncDecSpeakerLabelModel.list_available_models()). Downloads go to
        weights/nemo/<name>/, so a cached model loads offline.
        """
        if model_name.endswith(".nemo"):
            if not os.path.isfile(model_name):
                raise FileNotFoundError(f"Language identification checkpoint not found: {model_name}")
            return model_name

        locations = {
            info.pretrained_model_name: info.location
            for info in model_class.list_available_models() or []
        }
        if model_name not in locations:
            raise ValueError(
                f"Unknown language identification model '{model_name}'. "
                f"Use a .nemo path or one of: {sorted(n for n in locations if 'lang' in n)}"
            )

        from nemo.utils.cloud import maybe_download_from_cloud

        url = locations[model_name]
        filename = url.split("/")[-1]
        # Returns the cached file without network access when it already exists.
        return str(
            maybe_download_from_cloud(
                url=url[: -len(filename)],
                filename=filename,
                cache_dir=Path(weights_dir) / "nemo",  # NeMo joins it with pathlib
                subfolder=model_name,
            )
        )

    def supported(self, languages: Sequence[str]) -> list:
        return [code for code in languages if code in self.labels]

    def probabilities(self, audio: np.ndarray, languages: Sequence[str]) -> Dict[str, float]:
        """Softmax over `languages` only, like Whisper's language allow-list.

        audio is 16 kHz mono float32. Codes the model does not know are ignored.
        """
        torch = self._torch
        codes = self.supported(languages)
        if not codes:
            return {}
        with torch.inference_mode():
            logits, _ = self._model.forward(
                input_signal=torch.as_tensor(audio, dtype=torch.float32, device=self._device)[None],
                input_signal_length=torch.tensor([len(audio)], device=self._device),
            )
        selected = logits[0, [self.labels.index(code) for code in codes]].float()
        probs = torch.softmax(selected, dim=-1).cpu().numpy()
        return {code: float(p) for code, p in zip(codes, probs)}
