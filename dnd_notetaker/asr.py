"""Step 3 — ASR with Whisper. Returns segments with start/end/text and per-word timing."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    text: str  # includes Whisper's leading space, so words can be joined directly


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def transcribe(wav_16k: str | Path, model_size: str = "small", language: str | None = None) -> list[AsrSegment]:
    """Transcribe with openai-whisper.

    model_size: "small"/"base" run on CPU; "large-v3" for GPU accuracy.

    word_timestamps lets merge.py split a line where the speaker changes.
    condition_on_previous_text=False stops Whisper feeding its own output back in,
    the usual cause of repeat loops (one AMI run produced "I'm Sarah." x8).

    Call this BEFORE loading any pyannote model in the same process — see the
    note in pipeline.run (numba vs speechbrain import crash).
    """
    import whisper  # lazy import (heavy)

    model = whisper.load_model(model_size)
    result = model.transcribe(
        str(wav_16k),
        language=language,
        verbose=False,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    return [
        AsrSegment(
            float(s["start"]), float(s["end"]), s["text"].strip(),
            [Word(float(w["start"]), float(w["end"]), w["word"]) for w in s.get("words", [])],
        )
        for s in result["segments"]
    ]
