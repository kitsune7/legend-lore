"""Step 6 — score the pipeline against a hand-labeled gold set.

- DER (Diarization Error Rate) with its miss / false-alarm / confusion split, all
  reported as fractions of total reference speech.
- Speaker-attribution accuracy: time-weighted share of utterances whose predicted
  player_name matches gold.

DER is label-agnostic (it optimally maps clusters to reference speakers), so it
measures diarization. Attribution accuracy measures the enrollment step.

CLI:
  dnd-eval --pred-rttm outputs/x.rttm --gold-rttm data/gold/x.gold.rttm \
           [--pred-json outputs/x.transcript.json --gold-labels data/gold/x.gold_labels.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_annotation(rttm: str | Path):
    from pyannote.database.util import load_rttm

    return next(iter(load_rttm(str(rttm)).values()))


def score_der(gold_rttm, pred_rttm, collar: float = 0.0, skip_overlap: bool = False) -> dict:
    """DER and its components as fractions of total reference speech.

    `collar` forgives boundary jitter (0.25s is common in the literature);
    `skip_overlap` excludes regions where 2+ reference speakers overlap.
    Accepts RTTM paths or pyannote Annotations.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = gold_rttm if hasattr(gold_rttm, "itertracks") else load_annotation(gold_rttm)
    hyp = pred_rttm if hasattr(pred_rttm, "itertracks") else load_annotation(pred_rttm)
    comp = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)(ref, hyp, detailed=True)
    total = comp["total"] or 1.0
    return {
        "der": comp["diarization error rate"],
        "missed_detection": comp["missed detection"] / total,
        "false_alarm": comp["false alarm"] / total,
        "confusion": comp["confusion"] / total,
        "clusters": len(hyp.labels()),
    }


def attribution_accuracy(segments: list[dict], gold: list[dict]) -> float | None:
    """Time-weighted accuracy of predicted player_name (falls back to the raw
    `speaker` label when enrollment didn't run). Segments whose midpoint has no
    gold label are not scored."""
    correct = total = 0.0
    for s in segments:
        mid = (s["start"] + s["end"]) / 2
        g = next((g["player_name"] for g in gold if g["start"] <= mid < g["end"]), None)
        if g is None:
            continue
        dur = s["end"] - s["start"]
        total += dur
        if (s.get("player_name") or s.get("speaker")) == g:
            correct += dur
    return correct / total if total else None


def format_row(name: str, r: dict, width: int = 32) -> str:
    return (f"{name:{width}} {r['clusters']:>4} {r['der']*100:>6.1f}% "
            f"{r['missed_detection']*100:>6.1f}% {r['false_alarm']*100:>6.1f}% "
            f"{r['confusion']*100:>6.1f}%")


def format_header(width: int = 32) -> str:
    return f"{'setting':{width}} {'clus':>4} {'DER':>7} {'miss':>7} {'FA':>7} {'conf':>7}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DER + speaker attribution")
    ap.add_argument("--pred-rttm", required=True)
    ap.add_argument("--gold-rttm", required=True)
    ap.add_argument("--pred-json", help="transcript JSON, for attribution accuracy")
    ap.add_argument("--gold-labels", help='gold JSON: [{"start","end","player_name"}, ...]')
    ap.add_argument("--collar", type=float, default=0.0)
    ap.add_argument("--skip-overlap", action="store_true")
    args = ap.parse_args()

    r = score_der(args.gold_rttm, args.pred_rttm, args.collar, args.skip_overlap)
    print(format_header())
    print(format_row(Path(args.pred_rttm).stem, r))

    if args.pred_json and args.gold_labels:
        segs = json.loads(Path(args.pred_json).read_text())["segments"]
        gold = json.loads(Path(args.gold_labels).read_text())
        acc = attribution_accuracy(segs, gold)
        print(f"\nattribution accuracy (time-weighted): "
              f"{'n/a' if acc is None else f'{acc*100:.1f}%'}")


if __name__ == "__main__":
    main()
