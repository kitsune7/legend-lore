"""Experiment 3 — can VAD/clustering hyperparameters cut the missed detection?

Sweeps pyannote 3.1 knobs on the 5-min AMI slice, loading the model once.
  segmentation.min_duration_off  — bridge short silences
  clustering.threshold           — lower = more clusters, higher = more merging
  clustering.min_cluster_size    — lower = keep small clusters

Prereq: bash scripts/fetch_ami.sh
Run:    uv run python -m experiments.vad_tuning      (~3 min per config on CPU)

Result (2026-10-22): missed detection stuck at ~11% in every config (3.1's
segmentation model is fixed; only clustering is tunable). No config beat the
defaults; lowering threshold / min_cluster_size only raised confusion.
"""
from __future__ import annotations

import copy

from pyannote.core import Annotation

from dnd_notetaker import hf
from dnd_notetaker.evaluate import format_header, format_row, load_annotation, score_der

AUDIO = "data/ami/ES2004a_5min.wav"
GOLD = "data/ami/ES2004a_5min.gold.rttm"

pipe = hf.load_pipeline()
base = dict(pipe.parameters(instantiated=True))
print("default params:", base)
ref = load_annotation(GOLD)


def tweak(**changes) -> dict:
    params = copy.deepcopy(base)
    for dotted, value in changes.items():
        section, key = dotted.split(".")
        params[section][key] = value
    return params


configs = [
    ("baseline (defaults)", base),
    ("min_duration_off=0.5", tweak(**{"segmentation.min_duration_off": 0.5})),
    ("threshold=0.5", tweak(**{"clustering.threshold": 0.5})),
    ("threshold=0.85", tweak(**{"clustering.threshold": 0.85})),
    ("min_cluster_size=6", tweak(**{"clustering.min_cluster_size": 6})),
    ("off=0.5 + min_cluster=6",
     tweak(**{"segmentation.min_duration_off": 0.5, "clustering.min_cluster_size": 6})),
]

print("\n" + format_header())
for name, params in configs:
    pipe.instantiate(params)
    hyp: Annotation = pipe(AUDIO)
    print(format_row(name, score_der(ref, hyp)), flush=True)
