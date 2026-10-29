"""Hugging Face access for the gated pyannote models.

Single place for the token lookup and for loading pyannote pipelines/models, so
the PyTorch compatibility shim is always applied before a checkpoint is read.
"""
from __future__ import annotations

import os
from pathlib import Path

DIARIZATION_PIPELINE = "pyannote/speaker-diarization-3.1"
EMBEDDING_MODEL = "pyannote/embedding"

_TORCH_PATCHED = False


def token() -> str | None:
    """HF token from the environment, falling back to a local `.env` file."""
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN" and value.strip():
                return value.strip()
    return None


def _patch_torch_load() -> None:
    """PyTorch >= 2.6 defaults `torch.load(weights_only=True)`, which rejects the
    official pyannote checkpoints (they pickle `torch.torch_version.TorchVersion`).
    Force `weights_only=False` — callers like lightning_fabric pass True explicitly,
    so `setdefault` is not enough."""
    global _TORCH_PATCHED
    if _TORCH_PATCHED:
        return
    import torch

    original = torch.load

    def load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = load
    _TORCH_PATCHED = True


def load_pipeline(name: str = DIARIZATION_PIPELINE):
    from pyannote.audio import Pipeline

    _patch_torch_load()
    pipe = Pipeline.from_pretrained(name, use_auth_token=token())
    if pipe is None:  # pyannote returns None instead of raising on a 401/403
        raise RuntimeError(
            f"Could not load {name}. Check HF_TOKEN and accept the model terms at "
            f"https://huggingface.co/{name} (and pyannote/segmentation-3.0)."
        )
    return pipe


def load_embedding(name: str = EMBEDDING_MODEL):
    from pyannote.audio import Inference, Model

    _patch_torch_load()
    return Inference(Model.from_pretrained(name, use_auth_token=token()), window="whole")
