"""Step 3 — ASR with Whisper. Returns segments with start/end/text."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str


def transcribe(wav_16k: str | Path, model_size: str = "small", language: str | None = None) -> list[AsrSegment]:
    """Transcribe with openai-whisper.

    model_size: "small"/"base" run on CPU; "large-v3" for GPU accuracy.
    """
    import whisper  # lazy import (heavy)

    model = whisper.load_model(model_size)
    result = model.transcribe(str(wav_16k), language=language, verbose=False)
    return [
        AsrSegment(float(s["start"]), float(s["end"]), s["text"].strip())
        for s in result["segments"]
    ]
