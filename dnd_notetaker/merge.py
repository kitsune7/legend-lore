"""Step 4 — merge ASR text with diarization labels.

Each WORD is assigned the diarization cluster it overlaps most in time, and a
Whisper line is split wherever the speaker changes — otherwise a line covering
two people gets one label and the quieter speaker disappears from the transcript.
Words that fall in a gap between diarization turns inherit a neighbouring word's
speaker; a line with no speech overlap at all gets speaker=None.

Optionally (`min_speech_fraction` > 0) lines that barely overlap detected speech
are dropped, since Whisper can invent text over silence. Off by default: on AMI
it also dropped real words wherever the diarizer missed a quiet speaker.
Produces the output transcript records (the I/O contract from the plan).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from .asr import AsrSegment
from .diarize import DiarSegment

# Suggested value for --drop-silence: share of a line that must overlap detected speech.
DROP_SILENCE_FRACTION = 0.2


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    speaker: str | None               # raw diarization cluster; None = no speech overlap
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


def _best_speaker(start: float, end: float, diar_segs: list[DiarSegment]) -> tuple[str | None, float]:
    """(speaker with most overlap, total speech overlap) for a time span."""
    best_label, best_ov, speech = None, 0.0, 0.0
    for d in diar_segs:
        ov = _overlap(start, end, d.start, d.end)
        speech += ov
        if ov > best_ov:
            best_ov, best_label = ov, d.speaker
    return best_label, speech


def _split_by_speaker(seg: AsrSegment, diar_segs: list[DiarSegment]) -> list[Utterance]:
    labels = [_best_speaker(w.start, w.end, diar_segs)[0] for w in seg.words]
    if not any(labels):
        return [Utterance(seg.start, seg.end, seg.text, None)]

    # Words in a gap between turns take the previous word's speaker (or the next
    # one's at the start of a line), so short gaps don't fragment a sentence.
    for i in range(len(labels)):
        if labels[i] is None and i > 0:
            labels[i] = labels[i - 1]
    for i in reversed(range(len(labels))):
        if labels[i] is None:
            labels[i] = labels[i + 1]

    out: list[Utterance] = []
    for label, group in groupby(zip(seg.words, labels), key=lambda wl: wl[1]):
        run = [w for w, _ in group]
        out.append(Utterance(run[0].start, run[-1].end,
                             "".join(w.text for w in run).strip(), label))
    return out


def merge(
    asr_segs: list[AsrSegment],
    diar_segs: list[DiarSegment],
    min_speech_fraction: float = 0.0,
) -> list[Utterance]:
    """Attach speakers to ASR text, splitting lines at speaker changes.
    `min_speech_fraction` > 0 drops lines mostly over silence (see module doc)."""
    out: list[Utterance] = []
    for seg in asr_segs:
        label, speech = _best_speaker(seg.start, seg.end, diar_segs)
        if speech < min_speech_fraction * max(seg.end - seg.start, 1e-6):
            continue  # mostly silence -> likely a Whisper hallucination
        if seg.words:
            out.extend(_split_by_speaker(seg, diar_segs))
        else:
            out.append(Utterance(seg.start, seg.end, seg.text, label))
    return out
