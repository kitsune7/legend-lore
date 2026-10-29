"""Step 2 — diarization: "how many voices, and who spoke when?"

Uses the pretrained pyannote `speaker-diarization-3.1` pipeline (VAD + embeddings +
clustering + overlap handling bundled). Output is ANONYMOUS clusters (SPEAKER_00, ...);
binding those to real players is a separate step (enroll.py).

Without an HF token this degrades gracefully to a single SPEAKER_00 spanning the
whole file, so the rest of the pipeline still produces a transcript.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import hf


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str  # anonymous cluster label, e.g. "SPEAKER_00"


def diarize(
    wav_16k: str | Path,
    num_speakers: int | None = None,
    rttm_out: str | Path | None = None,
) -> list[DiarSegment]:
    """Run pyannote diarization and optionally write an RTTM.

    `num_speakers` forces the cluster count. On AMI this made DER worse
    (19.4% -> 39.9%), so leave it unset unless you have measured it helps.
    """
    wav_16k = Path(wav_16k)

    if not hf.token():
        print(
            "[diarize] WARNING: no HF_TOKEN — cannot load the gated pyannote model.\n"
            "          Falling back to a single SPEAKER_00 for the whole file.\n"
            "          See README.md > Setup to enable real diarization."
        )
        import soundfile as sf

        info = sf.info(str(wav_16k))
        segs = [DiarSegment(0.0, info.frames / info.samplerate, "SPEAKER_00")]
        if rttm_out:
            write_rttm(segs, wav_16k.stem, rttm_out)
        return segs

    pipe = hf.load_pipeline()
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    annotation = pipe(str(wav_16k), **kwargs)

    segs = sorted(
        (DiarSegment(float(turn.start), float(turn.end), str(label))
         for turn, _, label in annotation.itertracks(yield_label=True)),
        key=lambda s: s.start,
    )
    if rttm_out:
        with open(rttm_out, "w") as fh:
            annotation.write_rttm(fh)
    return segs


def write_rttm(segs: list[DiarSegment], uri: str, path: str | Path) -> None:
    with open(path, "w") as fh:
        for s in segs:
            fh.write(
                f"SPEAKER {uri} 1 {s.start:.3f} {s.end - s.start:.3f} "
                f"<NA> <NA> {s.speaker} <NA> <NA>\n"
            )
