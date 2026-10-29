"""Experiment 1 — full AMI ES2004a file, with and without a forced speaker count.

Prereq: bash scripts/fetch_ami.sh
Run:    .venv/bin/python -m experiments.ami_compare      (~12 min each pass on CPU)

Result (2026-10-22): auto 19.4% DER (5 clusters) vs num_speakers=4 39.9% DER —
forcing the count blew confusion up from 3.7% to 24.3%.
"""
from __future__ import annotations

import time

from dnd_notetaker import diarize
from dnd_notetaker.evaluate import format_header, format_row, score_der

AUDIO = "data/ami/ES2004a.Mix-Headset.wav"
GOLD = "data/ami/ES2004a.rttm"

n_spk = len({line.split()[7] for line in open(GOLD) if line.strip()})
print(f"AMI ES2004a — full file. Ground-truth speakers: {n_spk}\n")

rows = []
for name, ns in [("auto (no num_speakers)", None), (f"num_speakers={n_spk}", n_spk)]:
    t0 = time.time()
    rttm = f"outputs/ES2004a_full_{ns or 'auto'}.rttm"
    diarize.diarize(AUDIO, num_speakers=ns, rttm_out=rttm)
    rows.append((name, score_der(GOLD, rttm)))
    print(f"[{name}] done in {time.time() - t0:.0f}s")

print("\n" + format_header())
for name, r in rows:
    print(format_row(name, r))
