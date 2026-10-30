"""Orchestrates the MVP pipeline and writes JSON / SRT / RTTM outputs."""
from __future__ import annotations

import json
from pathlib import Path

from . import asr, diarize, enroll, merge, preprocess


def _fmt_ts(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def write_srt(utterances, path: str | Path) -> None:
    with open(path, "w") as fh:
        for i, u in enumerate(utterances, 1):
            who = u.player_name or u.speaker or "unknown"
            fh.write(f"{i}\n{_fmt_ts(u.start)} --> {_fmt_ts(u.end)}\n{who}: {u.text}\n\n")


def run(
    audio: str | Path,
    out_dir: str | Path,
    model_size: str = "small",
    num_speakers: int | None = None,
    roster: dict | None = None,
    language: str | None = None,
    drop_silence: bool = False,
    min_speakers: int | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio).stem

    print("[1/5] preprocess -> 16kHz mono")
    wav = preprocess.to_16k_mono(audio, out_dir / f"{stem}_16k.wav")

    # ASR runs before diarization on purpose: Whisper's word timestamps compile a
    # numba kernel on first use, and that compile crashes if pyannote has already
    # imported speechbrain (its lazy `k2` placeholder module raises when touched).
    print(f"[2/5] ASR (whisper {model_size})")
    asr_segs = asr.transcribe(wav, model_size=model_size, language=language)
    print(f"      {len(asr_segs)} transcript segments")

    print("[3/5] diarize (who spoke when)")
    rttm = out_dir / f"{stem}.rttm"
    diar_segs = diarize.diarize(wav, num_speakers=num_speakers,
                                min_speakers=min_speakers, rttm_out=rttm)
    print(f"      {len(diar_segs)} segments, "
          f"{len({d.speaker for d in diar_segs})} speaker cluster(s)")

    print("[4/5] merge text + speaker")
    utterances = merge.merge(
        asr_segs, diar_segs,
        min_speech_fraction=merge.DROP_SILENCE_FRACTION if drop_silence else 0.0,
    )
    print(f"      {len(utterances)} utterances "
          f"({sum(u.speaker is None for u in utterances)} with no detected speaker)")

    if roster and roster.get("players"):
        print("[5/5] enrollment: match clusters -> players")
        utterances = enroll.assign_players(utterances, diar_segs, wav, roster)
    else:
        print("[5/5] no roster -> keeping anonymous SPEAKER_xx labels")

    transcript = {"audio": str(audio), "segments": [u.to_dict() for u in utterances]}
    json_path = out_dir / f"{stem}.transcript.json"
    srt_path = out_dir / f"{stem}.srt"
    json_path.write_text(json.dumps(transcript, indent=2))
    write_srt(utterances, srt_path)

    print(f"\nWrote:\n  {json_path}\n  {srt_path}\n  {rttm}")
    return {"json": str(json_path), "srt": str(srt_path), "rttm": str(rttm),
            "utterances": utterances, "diar_segs": diar_segs}
