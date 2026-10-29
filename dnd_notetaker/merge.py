"""Step 4 — merge ASR text with diarization labels.

Each ASR segment is assigned the diarization cluster it overlaps most in time.
Produces the output transcript records (the I/O contract from the plan).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .asr import AsrSegment
from .diarize import DiarSegment


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    speaker: str                      # raw diarization cluster, always present
    player_name: str | None = None    # filled only if enrollment ran
    spk_confidence: float | None = None
    ic_ooc: str | None = None         # downstream hook — left null by this component
    character_name: str | None = None # downstream hook — left null by this component

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "speaker": self.speaker,
            "player_name": self.player_name,
            "spk_confidence": self.spk_confidence,
            "ic_ooc": self.ic_ooc,
            "character_name": self.character_name,
        }


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge(asr_segs: list[AsrSegment], diar_segs: list[DiarSegment]) -> list[Utterance]:
    out: list[Utterance] = []
    for a in asr_segs:
        best_label, best_ov = "SPEAKER_00", -1.0
        for d in diar_segs:
            ov = _overlap(a.start, a.end, d.start, d.end)
            if ov > best_ov:
                best_ov, best_label = ov, d.speaker
        out.append(Utterance(start=a.start, end=a.end, text=a.text, speaker=best_label))
    return out
