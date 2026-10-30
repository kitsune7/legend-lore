"""Experiment 4 — identify roster players directly vs cluster-then-name.

Voice samples (~20s per person, solo speech only) come from meeting ES2004b; the
test is ES2004a — same four people, different meeting, so no leakage.

Compares against the clustering run saved by ami_compare.py on:
  - DER (label-agnostic; clusters are optimally mapped to people)
  - per-person recall: share of each person's speech credited to them
    (for clustering: via the cluster that overlaps them most)

Prereq: bash scripts/fetch_ami.sh ; experiments/ami_compare.py (baseline RTTM)
Run:    .venv/bin/python -m experiments.ami_identify      (~2 min on CPU)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from dnd_notetaker.diarize import write_rttm
from dnd_notetaker.evaluate import load_annotation, score_der
from dnd_notetaker.identify import label_windows, score_windows

ENROLL_AUDIO, ENROLL_GOLD = "data/ami/ES2004b.Mix-Headset.wav", "data/ami/ES2004b.rttm"
TEST_AUDIO, TEST_GOLD = "data/ami/ES2004a.Mix-Headset.wav", "data/ami/ES2004a.rttm"
BASELINE = "outputs/ES2004a_full_auto.rttm"
ENROLL_SECONDS, MIN_SOLO = 20.0, 2.0


def build_enrollment(out_dir: Path) -> dict:
    """Concatenate each speaker's longest solo stretches (no one else talking)."""
    gold = [(float(l.split()[3]), float(l.split()[4]), l.split()[7]) for l in open(ENROLL_GOLD)]
    audio, sr = sf.read(ENROLL_AUDIO)
    out_dir.mkdir(parents=True, exist_ok=True)

    def solo(s, d, spk):
        return all(o_spk == spk or o_s + o_d <= s or o_s >= s + d for o_s, o_d, o_spk in gold)

    players = []
    for spk in sorted({g[2] for g in gold}):
        segs = sorted((g for g in gold if g[2] == spk and g[1] >= MIN_SOLO and solo(*g)),
                      key=lambda g: -g[1])
        chunks, total = [], 0.0
        for s, d, _ in segs:
            chunks.append(audio[int(s * sr):int((s + d) * sr)])
            total += d
            if total >= ENROLL_SECONDS:
                break
        path = out_dir / f"{spk}.wav"
        sf.write(path, np.concatenate(chunks), sr)
        players.append({"player_id": spk, "player_name": spk, "enroll_wav": str(path)})
        print(f"  enrollment {spk}: {total:.1f}s from {len(chunks)} solo segment(s)")
    return {"players": players}


def per_person_recall(ref, hyp, mapping=None) -> dict[str, float]:
    """Share of each reference speaker's time covered by the hypothesis label
    credited to them (identity mapping for identification)."""
    if mapping is None:  # clustering: credit each person with their best cluster
        ov = defaultdict(float)
        for spk in ref.labels():
            for lab in hyp.labels():
                ov[(spk, lab)] = ref.label_timeline(spk).crop(hyp.label_timeline(lab)).duration()
        mapping, used = {}, set()
        for (spk, lab), t in sorted(ov.items(), key=lambda kv: -kv[1]):
            if spk not in mapping and lab not in used and t > 0:
                mapping[spk], used = lab, used | {lab}
    out = {}
    for spk in ref.labels():
        lab = mapping.get(spk, spk)
        own = ref.label_timeline(spk)
        got = own.crop(hyp.label_timeline(lab)).duration() if lab in hyp.labels() else 0.0
        out[spk] = got / own.duration()
    return out


print("Building enrollment from ES2004b ...")
roster = build_enrollment(Path("data/ami/enroll"))
ref = load_annotation(TEST_GOLD)
speakers = sorted(ref.labels())

print("\nScoring ES2004a windows (once) ...")
ws = score_windows(TEST_AUDIO, roster)

print(f"\n{'method':26} {'DER':>6} {'miss':>6} {'FA':>6} {'conf':>6}   "
      + "  ".join(f"{s:>6}" for s in speakers))
base = load_annotation(BASELINE)
r, rec = score_der(ref, base), per_person_recall(ref, base)
print(f"{'clustering (baseline)':26} {r['der']*100:5.1f}% {r['missed_detection']*100:5.1f}% "
      f"{r['false_alarm']*100:5.1f}% {r['confusion']*100:5.1f}%   "
      + "  ".join(f"{rec[s]*100:5.0f}%" for s in speakers))

for th in (0.1, 0.2, 0.3, 0.4, 0.5):
    rttm = f"outputs/ES2004a_identify_{th:.1f}.rttm"
    write_rttm(label_windows(ws, th), "ES2004a", rttm)
    hyp = load_annotation(rttm)
    r = score_der(ref, hyp)
    rec = per_person_recall(ref, hyp, mapping={s: s for s in speakers})
    print(f"{f'identify threshold={th:.1f}':26} {r['der']*100:5.1f}% {r['missed_detection']*100:5.1f}% "
          f"{r['false_alarm']*100:5.1f}% {r['confusion']*100:5.1f}%   "
          + "  ".join(f"{rec[s]*100:5.0f}%" for s in speakers))
