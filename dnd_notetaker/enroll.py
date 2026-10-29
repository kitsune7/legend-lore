"""Step 5 (optional) — bind anonymous clusters to real players by nearest-centroid
cosine matching against per-player enrollment samples.

Each player's enroll_wav -> one embedding (their centroid). Each diarization cluster
-> mean embedding of (a sample of) its segments. Assign each cluster to its best
player; if the best cosine similarity < unknown_threshold, label it "unknown"
(covers guests and the DM's extra NPC voices).

Skipped (anonymous labels kept) if there is no HF token or no roster.
"""
from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from . import hf
from .diarize import DiarSegment
from .merge import Utterance

MIN_SEGMENT_SECONDS = 0.4  # shorter snippets give unreliable embeddings


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def match_clusters(
    cluster_vecs: dict[str, np.ndarray],
    player_vecs: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, tuple[str | None, float]]:
    """Greedy one-to-one assignment of clusters to players, best similarity first.
    Clusters with no match above `threshold` map to (None, 0.0)."""
    pairs = sorted(
        ((cosine(cv, pv), cl, name)
         for cl, cv in cluster_vecs.items()
         for name, pv in player_vecs.items()),
        reverse=True,
    )
    mapping: dict[str, tuple[str | None, float]] = {}
    taken: set[str] = set()
    for sim, cl, name in pairs:
        if sim < threshold or cl in mapping or name in taken:
            continue
        mapping[cl] = (name, sim)
        taken.add(name)
    for cl in cluster_vecs:
        mapping.setdefault(cl, (None, 0.0))
    return mapping


def _embed(embedder, wav: str | Path) -> np.ndarray:
    return np.asarray(embedder(str(wav))).reshape(-1)


def _cluster_centroids(
    embedder, wav_16k: str | Path, diar_segs: list[DiarSegment], max_segs: int = 8
) -> dict[str, np.ndarray]:
    """Mean embedding per cluster, using up to `max_segs` of its longest segments."""
    audio, sr = sf.read(str(wav_16k))
    by_label: dict[str, list[DiarSegment]] = defaultdict(list)
    for s in diar_segs:
        by_label[s.speaker].append(s)

    centroids: dict[str, np.ndarray] = {}
    for label, segs in by_label.items():
        vecs = []
        for s in sorted(segs, key=lambda s: s.end - s.start, reverse=True)[:max_segs]:
            lo, hi = int(s.start * sr), int(s.end * sr)
            if hi - lo < sr * MIN_SEGMENT_SECONDS:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
                sf.write(tf.name, audio[lo:hi], sr)
                vecs.append(_embed(embedder, tf.name))
        if vecs:
            centroids[label] = np.mean(vecs, axis=0)
    return centroids


def assign_players(
    utterances: list[Utterance],
    diar_segs: list[DiarSegment],
    wav_16k: str | Path,
    roster: dict,
) -> list[Utterance]:
    """Rename clusters -> player names in place. `roster` is the parsed roster JSON."""
    players = roster.get("players", [])
    if not players:
        return utterances
    if not hf.token():
        print("[enroll] No HF_TOKEN — skipping enrollment, keeping anonymous labels.")
        return utterances

    embedder = hf.load_embedding()
    player_vecs: dict[str, np.ndarray] = {}
    for p in players:
        wav = p.get("enroll_wav")
        if wav and Path(wav).exists():
            player_vecs[p["player_name"]] = _embed(embedder, wav)
        else:
            print(f"[enroll] enroll_wav missing for {p.get('player_name')}: {wav}")

    mapping = match_clusters(
        _cluster_centroids(embedder, wav_16k, diar_segs),
        player_vecs,
        float(roster.get("unknown_threshold", 0.5)),
    )
    for u in utterances:
        name, sim = mapping.get(u.speaker, (None, 0.0))
        u.player_name = name or "unknown"
        u.spk_confidence = round(sim, 3) if name else None
    return utterances
