import os
import subprocess
import wave

import numpy as np


def save_to_wav(
    audio_buffer: list,
    wav_path: str,
    sample_rate: int,
    logger=None,
    append: bool = False,
) -> bool:
    """Convert a list of float32 numpy arrays to 16-bit PCM WAV data.

    The WAV module does not support append mode, so when appending we must read the
    existing WAV header and frame payload, then rewrite a combined file with the new
    samples appended. This preserves the periodic flush strategy without triggering
    the invalid wave.open() mode error.
    """
    if not audio_buffer:
        if logger:
            logger.warn("No audio received, not writing WAV")
        return False

    samples = np.concatenate(audio_buffer)
    samples = np.clip(samples, -1.0, 1.0)
    new_samples_int16 = (samples * 32767.0).astype(np.int16)
    new_frames = new_samples_int16.tobytes()

    file_exists = os.path.exists(wav_path)
    if append and file_exists:
        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != sample_rate:
                raise ValueError(
                    f"Existing WAV metadata mismatch: {wf.getnchannels()}ch, "
                    f"{wf.getsampwidth()*8}bit, {wf.getframerate()}Hz"
                )
            previous_frames = wf.readframes(wf.getnframes())

        combined = previous_frames + new_frames
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(combined)

        if logger:
            logger.info(f"WAV updated to {wav_path}, +{len(new_samples_int16)} samples")
        return True

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(new_frames)

    if logger:
        logger.info(f"WAV written to {wav_path}, {len(new_samples_int16)} samples")
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
