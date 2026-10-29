"""Experiment 2 — how much of the AMI error is overlapping speech?

Re-scores the saved full-file diarization (no re-diarization) with and without
overlapped regions and a boundary collar.

Prereq: experiments/ami_compare.py (writes outputs/ES2004a_full_auto.rttm)
Run:    .venv/bin/python -m experiments.overlap_eval      (seconds)

Result (2026-10-22): 15.8% of speech is overlapped. DER 19.4% strict ->
11.3% skipping overlap -> 8.4% with collar=0.25 + skip_overlap.
"""
from __future__ import annotations

from dnd_notetaker.evaluate import format_header, format_row, load_annotation, score_der

GOLD = "data/ami/ES2004a.rttm"
PRED = "outputs/ES2004a_full_auto.rttm"

ref = load_annotation(GOLD)
hyp = load_annotation(PRED)

speech = ref.get_timeline().support().duration()
overlap = ref.get_overlap().duration()
print(f"Reference: {speech:.0f}s speech, {overlap:.0f}s overlapped "
      f"({overlap / speech * 100:.1f}% of speech)\n")

configs = [
    ("strict (collar=0, overlap scored)", 0.0, False),
    ("collar=0.25s", 0.25, False),
    ("skip_overlap", 0.0, True),
    ("collar=0.25 + skip_overlap", 0.25, True),
]
print(format_header(36))
for name, collar, skip in configs:
    print(format_row(name, score_der(ref, hyp, collar, skip), 36))
