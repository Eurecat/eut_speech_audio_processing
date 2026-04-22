import subprocess
import wave

import numpy as np


def save_to_wav(audio_buffer: list, wav_path: str, sample_rate: int, logger=None) -> bool:
    """Convert a list of float32 numpy arrays to a 16-bit PCM WAV file.

    Args:
        audio_buffer: List of numpy float32 arrays with samples in [-1.0, 1.0].
        wav_path: Destination path for the WAV file.
        sample_rate: Sample rate in Hz.
        logger: Optional ROS2 logger for status messages.

    Returns:
        True if the file was written successfully, False if the buffer was empty.
    """
    if not audio_buffer:
        if logger:
            logger.warn("No audio received, not writing WAV")
        return False

    samples = np.concatenate(audio_buffer)

    # Convert [-1.0, 1.0] float32 → int16 PCM
    samples = np.clip(samples, -1.0, 1.0)
    samples_int16 = (samples * 32767.0).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())

    if logger:
        logger.info(f"WAV written to {wav_path}, {len(samples_int16)} samples")
    return True


def convert_wav_to_mp3(wav_path: str, mp3_path: str, logger=None) -> None:
    """Convert a WAV file to MP3 using ffmpeg.

    Args:
        wav_path: Path to the source WAV file.
        mp3_path: Destination path for the MP3 file.
        logger: Optional ROS2 logger for status messages.

    Raises:
        subprocess.CalledProcessError: If ffmpeg exits with a non-zero return code.
    """
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output file if it exists
        "-i",
        wav_path,
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",  # VBR quality (2 = high quality)
        mp3_path,
    ]
    if logger:
        logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if logger:
        logger.info(f"MP3 written to {mp3_path}")
