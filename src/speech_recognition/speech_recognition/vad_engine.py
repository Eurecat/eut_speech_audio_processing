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

    def predict(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Run VAD inference on a chunk of audio.

        Args:
            audio_data: Float32 numpy array, expected length 512.
            sample_rate: Sample rate in Hz.

        Returns:
            Speech probability in [0.0, 1.0]. Returns 0.0 for unexpected chunk sizes.
        """
        if audio_data.size != self.EXPECTED_CHUNK_SIZE:
            self._logger.warn(
                f"Unexpected audio chunk size: {audio_data.size}. "
                f"Expected {self.EXPECTED_CHUNK_SIZE} samples."
            )
            return 0.0

        audio_tensor = torch.from_numpy(audio_data).to(self.device)
        with torch.no_grad():
            return self.model(audio_tensor, sr=sample_rate).item()
