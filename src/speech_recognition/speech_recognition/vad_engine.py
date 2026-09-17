import os

import numpy as np
import torch


class VADEngine:
    """Loads Silero VAD and runs inference. Zero ROS2 dependencies."""

    EXPECTED_CHUNK_SIZE = 512

    def __init__(self, *, repo_model: str, model_name: str, weights_dir: str, logger):
        self._logger = logger

        os.makedirs(weights_dir, exist_ok=True)
        torch.hub.set_dir(weights_dir)

        self._logger.info(f"Loading VAD model from: {weights_dir}")
        self.model, _ = torch.hub.load(repo_or_dir=repo_model, model=model_name, trust_repo=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger.info(f"Using device on VAD: {self.device}")
        self.model.to(self.device)
        self._logger.info("VAD engine ready.")
        self._pending = np.empty(0, dtype=np.float32)
        self._last_probability = 0.0

    def predict(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Run VAD inference on streaming audio.

        Input can be any chunk size. The engine buffers samples and evaluates
        contiguous 512-sample windows, keeping remainder for the next call.
        """
        if audio_data.size == 0:
            return self._last_probability

        if self._pending.size == 0:
            merged = audio_data
        else:
            merged = np.concatenate((self._pending, audio_data))

        probabilities = []
        offset = 0
        while offset + self.EXPECTED_CHUNK_SIZE <= merged.size:
            window = merged[offset : offset + self.EXPECTED_CHUNK_SIZE]
            audio_tensor = torch.from_numpy(window).to(self.device)
            with torch.no_grad():
                probabilities.append(self.model(audio_tensor, sr=sample_rate).item())
            offset += self.EXPECTED_CHUNK_SIZE

        self._pending = merged[offset:].astype(np.float32, copy=False)

        if not probabilities:
            return self._last_probability

        self._last_probability = max(probabilities)
        return self._last_probability
