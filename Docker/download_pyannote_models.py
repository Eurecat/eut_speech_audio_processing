"""
Pre-download pyannote/segmentation and pyannote/embedding into the HF hub cache.

Called as a Docker BuildKit --secret RUN step in Dockerfile.arm so the models
are baked into the image and available offline at runtime.

ARM / NVIDIA pre-release workarounds applied here (same as diarization_engine.py):
  • torch.load weights_only=False  — NVIDIA PyTorch pre-release (2.6+) defaults
    weights_only=True, but pyannote/lightning checkpoints pickle omegaconf objects
    that are not in the allowlist.
  • pyannote.audio check_version stub — NVIDIA version strings like
    "2.8.0a0+34c6371d24" are not valid SemVer and raise ValueError inside
    pyannote's on_load_checkpoint().

HF_TOKEN is read from /run/secrets/hf_token (injected by BuildKit --secret).
If the secret file is absent or empty the script exits 0 so the build still
succeeds — models will then download on first container run (requires internet).
"""
import os
import sys

# --- Read HF_TOKEN from BuildKit secret file -----------------------------------
_SECRET_PATH = "/run/secrets/hf_token"
if not os.path.exists(_SECRET_PATH):
    print(
        "[download_pyannote_models] No HF_TOKEN secret provided — "
        "skipping model pre-download. Models will be fetched on first container run.",
        file=sys.stderr,
    )
    sys.exit(0)

with open(_SECRET_PATH) as _f:
    hf_token = _f.read().strip()

if not hf_token: 
    print(
        "[download_pyannote_models] HF_TOKEN secret is empty — "
        "skipping model pre-download.",
        file=sys.stderr,
    )
    sys.exit(0)

# Expose to huggingface_hub so hf_hub_download authenticates correctly.
os.environ["HF_TOKEN"] = hf_token

# --- ARM / NVIDIA pre-release patches -----------------------------------------
import torch  # noqa: E402

_orig_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    """Force weights_only=False for all torch.load calls during model loading."""
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

# pyannote.audio imports check_version to validate the torch version string.
# NVIDIA pre-release strings (e.g. "2.8.0a0+34c6371d24") are not valid SemVer
# and raise ValueError. Stub the check out entirely for this script.
import pyannote.audio.core.model as _pyannote_model_mod  # noqa: E402

_pyannote_model_mod.check_version = lambda *a, **kw: None

# --- Download models -----------------------------------------------------------
import diart.models as m  # noqa: E402

SEGMENTATION_MODEL = "pyannote/segmentation"
EMBEDDING_MODEL = "pyannote/embedding"

print(f"[download_pyannote_models] Downloading {SEGMENTATION_MODEL} …")
seg = m.SegmentationModel.from_pretrained(SEGMENTATION_MODEL, use_hf_token=hf_token)
seg.load()
print(f"[download_pyannote_models]   ✓ {type(seg.model).__name__}")

print(f"[download_pyannote_models] Downloading {EMBEDDING_MODEL} …")
emb = m.EmbeddingModel.from_pretrained(EMBEDDING_MODEL, use_hf_token=hf_token)
emb.load()
print(f"[download_pyannote_models]   ✓ {type(emb.model).__name__}")

# Restore original torch.load before exiting.
torch.load = _orig_torch_load
_pyannote_model_mod.check_version = _pyannote_model_mod.check_version  # already stubbed, leave it

print(
    f"[download_pyannote_models] Models cached to HF_HOME="
    f"{os.environ.get('HF_HOME', '~/.cache/huggingface')}"
)
