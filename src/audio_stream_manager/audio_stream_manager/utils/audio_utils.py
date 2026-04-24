import os

import numpy as np


class SuppressStderr:
    """Context manager to silence ALSA/PortAudio C-level error prints."""

    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.saved_stderr = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *args):
        os.dup2(self.saved_stderr, 2)
        os.close(self.null_fd)
        os.close(self.saved_stderr)


def compute_rms(audio_data: np.ndarray) -> float:
    """Compute the Root Mean Square of an audio array.

    Args:
        audio_data: Numpy array of audio samples.

    Returns:
        RMS value as a float.
    """
    return float(np.sqrt(np.mean(audio_data**2)))


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int, logger=None) -> np.ndarray:
    """Resample *audio_data* from *orig_sr* to *target_sr* using librosa.

    Args:
        audio_data: Float32 numpy array of audio samples.
        orig_sr: Original sample rate in Hz.
        target_sr: Target sample rate in Hz.
        logger: Optional ROS2 logger for error messages.

    Returns:
        Resampled float32 numpy array. If librosa is not installed, the original
        array is returned unchanged.
    """
    try:
        import librosa

        resampled = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        return resampled.astype(np.float32)
    except ImportError:
        if logger:
            logger.error("librosa is not installed. Cannot resample audio. Please install librosa.")
        return audio_data
