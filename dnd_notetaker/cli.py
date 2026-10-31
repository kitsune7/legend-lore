"""CLI: audio -> player-attributed transcript.

Examples:
  uv run python -m dnd_notetaker.cli --audio data/session.m4a --out outputs --model small
  uv run python -m dnd_notetaker.cli --audio data/session.wav --roster config/roster.sample.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dnd_notetaker import pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="D&D audio -> player-attributed transcript (MVP)")
    ap.add_argument("--audio", required=True, help="input audio (wav/mp3/m4a/opus/...)")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--model", default="small",
                    help="whisper model: tiny/base/small/medium/large-v3 (default: small)")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="force the cluster count (made DER worse on AMI; measure first)")
    ap.add_argument("--roster", default=None,
                    help="roster JSON with player enrollment wavs (enables name matching)")
    ap.add_argument("--language", default=None, help="force a language code, e.g. en")
    ap.add_argument("--min-speakers", type=int, default=None,
                    help="lower bound on speaker count (unlike --num-speakers, doesn't force it)")
    ap.add_argument("--drop-silence", action="store_true",
                    help="drop lines mostly over silence (cuts Whisper hallucinations, "
                         "but also drops speech the diarizer missed)")
    args = ap.parse_args()

    roster = None
    if args.roster:
        roster = json.loads(Path(args.roster).read_text())

    pipeline.run(
        audio=args.audio,
        out_dir=args.out,
        model_size=args.model,
        num_speakers=args.num_speakers,
        roster=roster,
        language=args.language,
        drop_silence=args.drop_silence,
        min_speakers=args.min_speakers,
    )


if __name__ == "__main__":
    main()
