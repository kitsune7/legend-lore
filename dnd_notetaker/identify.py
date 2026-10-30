"""Step 3 alternative — identify known players directly instead of clustering.

Clustering (diarize.py) groups voices first and names the groups afterwards, so a
player who speaks rarely never gets a group and can't be named. When we have a
roster with enrollment samples, we can skip clustering: slide a 1.5s window over
the speech, embed each window, and give it the closest player's name.

  speech / overlap regions  <- pyannote segmentation-3.0 (VAD + overlap detection)
  voiceprints               <- pyannote wespeaker-voxceleb-resnet34-LM
  per window                <- cosine vs each player's enrollment embedding(s);
                               best player if >= threshold, else "unknown";
                               in detected overlap, also the runner-up if >= threshold

A roster entry's `enroll_wav` may be a single path or a list (e.g. a player's
natural voice plus their character voice); a window scores against the closest one.
Output segments use player names as speaker labels, so merge.py works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import hf
from .diarize import DiarSegment, write_rttm

SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
WINDOW, STEP = 1.5, 0.75  # seconds
# Cosine similarity. On AMI ES2004a, 0.1 was best (DER 20.4%); 0.3 -> 26.5%, 0.5 -> 45.6%.
DEFAULT_THRESHOLD = 0.1
UNKNOWN = "unknown"


@dataclass
class WindowScores:
    """Per-window similarity to each player, plus the speech/overlap regions."""
    centers: np.ndarray        # (n_windows,) window centre times, seconds
    scores: np.ndarray         # (n_windows, n_players) cosine similarity, smoothed
    players: list[str]
    speech: object             # pyannote Timeline
    overlap: object            # pyannote Timeline
    duration: float


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def score_windows(wav_16k: str | Path, roster: dict) -> WindowScores:
    from pyannote.audio import Inference, Model
    from pyannote.audio.pipelines import OverlappedSpeechDetection, VoiceActivityDetection
    import soundfile as sf

    hf._patch_torch_load()
    wav = str(wav_16k)

    seg_model = Model.from_pretrained(SEGMENTATION_MODEL, use_auth_token=hf.token())
    vad = VoiceActivityDetection(segmentation=seg_model)
    vad.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
    osd = OverlappedSpeechDetection(segmentation=seg_model)
    osd.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
    speech = vad(wav).get_timeline().support()
    overlap = osd(wav).get_timeline().support()

    emb_model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=hf.token())
    whole = Inference(emb_model, window="whole")
    players, modes = [], []
    for p in roster["players"]:
        wavs = p["enroll_wav"] if isinstance(p["enroll_wav"], list) else [p["enroll_wav"]]
        players.append(p["player_name"])
        modes.append(_unit(np.stack([np.asarray(whole(w)).reshape(-1) for w in wavs])))

    feats = Inference(emb_model, window="sliding", duration=WINDOW, step=STEP)(wav)
    win = _unit(np.nan_to_num(feats.data))
    # Each player scores as their closest enrollment "voice mode".
    scores = np.stack([(win @ m.T).max(axis=1) for m in modes], axis=1)
    # Light temporal smoothing: average each window with its neighbours.
    kernel = np.ones(3) / 3
    scores = np.stack([np.convolve(scores[:, j], kernel, mode="same")
                       for j in range(scores.shape[1])], axis=1)

    sw = feats.sliding_window
    centers = np.array([sw[i].middle for i in range(len(win))])
    info = sf.info(wav)
    return WindowScores(centers, scores, players, speech, overlap,
                        info.frames / info.samplerate)


def label_windows(ws: WindowScores, threshold: float = DEFAULT_THRESHOLD) -> list[DiarSegment]:
    """Turn window scores into named speech segments."""
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    order = np.argsort(-ws.scores, axis=1)
    for i, c in enumerate(ws.centers):
        lo = 0.0 if i == 0 else c - STEP / 2
        hi = ws.duration if i == len(ws.centers) - 1 else c + STEP / 2
        region = Segment(lo, hi)
        best, second = order[i, 0], order[i, 1] if ws.scores.shape[1] > 1 else None
        names = [ws.players[best] if ws.scores[i, best] >= threshold else UNKNOWN]
        for part in ws.speech.crop(region):
            ann[part, f"{i}a"] = names[0]
        if second is not None and ws.scores[i, second] >= threshold and names[0] != UNKNOWN:
            for part in ws.overlap.crop(region):
                ann[part, f"{i}b"] = ws.players[second]

    segs: list[DiarSegment] = []
    for label in ann.labels():
        for seg in ann.label_timeline(label).support():
            segs.append(DiarSegment(seg.start, seg.end, label))
    return sorted(segs, key=lambda s: s.start)


def identify(
    wav_16k: str | Path,
    roster: dict,
    threshold: float | None = None,
    rttm_out: str | Path | None = None,
) -> list[DiarSegment]:
    if threshold is None:
        threshold = float(roster.get("identify_threshold", DEFAULT_THRESHOLD))
    segs = label_windows(score_windows(wav_16k, roster), threshold)
    if rttm_out:
        write_rttm(segs, Path(wav_16k).stem, rttm_out)
    return segs
