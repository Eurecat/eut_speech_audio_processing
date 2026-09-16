"""Location of the shared model weights folder.

Models (Whisper, Parakeet, Silero VAD, ReDimNet2, pyannote) are never baked into
the Docker image. Compose bind-mounts the host folder
src/speech_recognition/speech_recognition/weights at /workspace/weights and sets
WEIGHTS_DIR, the same pattern as the other EutPerceptionStack repos. Outside
Docker the package weights/ folder is used.
"""

import os


def weights_dir() -> str:
    """Return WEIGHTS_DIR if set, else the package weights/ folder."""
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    return os.path.abspath(os.environ.get("WEIGHTS_DIR") or default)
