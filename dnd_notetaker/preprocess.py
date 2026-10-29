"""Step 1 — normalize any input audio to 16 kHz mono WAV (required by pyannote/ECAPA)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def to_16k_mono(src: str | Path, dst: str | Path | None = None) -> Path:
    """Convert `src` (any ffmpeg-readable audio) to 16 kHz mono WAV at `dst`.

    Returns the output path. If the input is already a 16k mono wav this still
    re-encodes it (cheap, keeps the rest of the pipeline simple).
    """
    src = Path(src)
    if dst is None:
        dst = src.with_name(src.stem + "_16k.wav")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        "-vn",               # no video
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst
