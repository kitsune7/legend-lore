# MVP Plan — Player-Attributed Transcript for D&D Actual-Play Audio

**Scope (tight):** a minimal working MVP. **INPUT:** session audio (+ optional per-player
enrollment voice samples + a roster). **OUTPUT:** a timestamped transcript where each utterance is
labeled with *which human (player)* spoke it.

This is **diarization + speaker-ID + ASR using off-the-shelf pretrained models only — NO training or
fine-tuning.** IC/OOC tagging and per-character-voice labeling are out of scope; the output JSON just
leaves null hooks for them.

A student should be able to run this in an **afternoon on a single GPU** (or CPU with Whisper small).

---

## 1. Recommended stack (one primary path)

**Primary MVP stack: OpenAI Whisper (ASR) + pyannote.audio `speaker-diarization-3.1` (pretrained
diarization) + pyannote/ECAPA embeddings for optional enrollment matching.**

Why this one: it is the fastest to stand up, runs locally, and uses **pretrained models with no
training** — the diarization pipeline is a few lines of Python. Whisper is MIT; pyannote code is MIT
(its pretrained models are gated — you accept terms and use a free Hugging Face token).

**NeMo vs Whisper+pyannote (short):** NVIDIA's NeMo stack (Parakeet/Canary ASR, Sortformer
diarization, TitaNet embeddings) is excellent and often more accurate on overlap, but it is heavier
to install, Sortformer v1 is 4-speaker-capped and non-commercially licensed, and it adds no MVP
value over pyannote. Skip it for the MVP; revisit if pyannote accuracy proves inadequate.

**Nemotron ≠ a speech model.** Nemotron is NVIDIA's *LLM* family (text). It is **not needed** for
this transcript+speaker component. It would only matter for the downstream IC/OOC / recap stretch
goals, where an LLM reads the finished transcript. Don't put any LLM in this pipeline.

---

## 2. I/O contract

**Input audio:** anything `ffmpeg` reads (WAV/MP3/M4A/Opus); the code normalizes to **16 kHz mono**
(required by pyannote/ECAPA; Whisper is fine with it).

**Roster + enrollment (JSON), optional:**
```json
{
  "players": [
    {"player_id": "p1", "player_name": "Matt",  "enroll_wav": "enroll/matt.wav"},
    {"player_id": "p2", "player_name": "Laura", "enroll_wav": "enroll/laura.wav"}
  ],
  "unknown_threshold": 0.5
}
```
Each `enroll_wav` = ~10–20 s of that player talking alone.

**Output transcript (JSON):** one object per utterance.
```json
{
  "segments": [
    {
      "start": 12.44, "end": 17.90,
      "text": "You come to the edge of the ravine.",
      "speaker": "SPEAKER_00",     // raw diarization label (always present)
      "player_name": "Matt",        // filled only if enrollment matching ran; else null
      "spk_confidence": 0.78,       // cosine sim to matched centroid; null if no enrollment
      "ic_ooc": null, "character_name": null   // downstream hooks, left null by this component
    }
  ]
}
```
Also emit **SRT/VTT** for human reading and **RTTM** for DER scoring (pyannote writes RTTM directly).

MVP behavior: **always** produce anonymous labels (`SPEAKER_00`, `SPEAKER_01`, …). If a roster with
enrollment samples is supplied, a second optional pass renames clusters to real players.

---

## 3. Diarization (central, explained simply)

Diarization answers **"how many voices, and who spoke when?"** — but it only gives back **anonymous
clusters** (`SPEAKER_00`, `SPEAKER_01`…), not names. For the MVP we use the **pretrained pyannote
`speaker-diarization-3.1` pipeline** as a black box (it bundles VAD + embeddings + clustering +
overlap handling). No building or comparing architectures from scratch.

```python
from pyannote.audio import Pipeline
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
diar = pipe("session_16k.wav")                 # or num_speakers=5 if you know the cast size
diar.write_rttm(open("session.rttm", "w"))     # -> SPEAKER_00 [t0..t1], ...
```

Two useful knobs (optional): pass `num_speakers=N` (we often know N from the roster — cheap accuracy
win), or `min_speakers`/`max_speakers`. That's it for the MVP.

**Why labels are anonymous and what fixes it:** diarization has no idea `SPEAKER_00` is "Matt."
Binding clusters to real humans is a **separate step** (§4). Attribution can only be as good as the
diarization — if two players get merged into one cluster, no enrollment step can un-merge them. Good
enough for an MVP; note the limitation.

**Later / if needed (not MVP):** NeMo Sortformer / Nemotron-3-Diarization (end-to-end, stronger on
overlap), per-player multitrack recordings (makes diarization nearly free), overlap-aware
resegmentation, VAD/clustering-threshold tuning.

---

## 4. Enrollment-based speaker ID (simple, optional)

Rename anonymous clusters to roster players by **nearest-centroid cosine matching**:

1. **Enroll:** embed each player's `enroll_wav` with a pretrained speaker-embedding model (ECAPA /
   pyannote embedding) → one vector per player (their centroid).
2. **Embed clusters:** for each diarization cluster, average the embeddings of its segments → one
   cluster vector.
3. **Match:** cosine-similarity each cluster vector to each player centroid; assign the best match
   (greedy, one player per cluster). If the best similarity `< unknown_threshold` (default ~0.5),
   label it `unknown` instead of forcing a name — covers guests and the DM's extra NPC voices.
4. **No enrollment given:** skip this step entirely and ship the anonymous `SPEAKER_xx` labels.

```python
from pyannote.audio import Model, Inference
import numpy as np
emb = Inference(Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN), window="whole")

def centroid(wav): return emb(wav)   # 1 vector
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
# roster[p] -> centroid(enroll_wav);  cluster[s] -> mean of segment embeddings
# assign speaker s -> argmax_p cos(cluster[s], roster[p]); if max < 0.5 -> "unknown"
```

Keep it this simple for the MVP — nearest centroid + one threshold.

---

## 5. ASR + merge

Transcribe with Whisper, then attach each transcript segment to the diarization label it overlaps
most in time.

```python
import whisper
model = whisper.load_model("small")        # "small" runs on CPU; "large-v3" on GPU for accuracy
asr = model.transcribe("session_16k.wav")  # -> segments with start/end/text
# for each asr segment, speaker = the diarization label with max temporal overlap
```

(For better word-level timing, `whisperx` adds forced alignment — optional, not required for MVP.)

---

## 6. Datasets — tiny gold set only

**No corpus build.** For the MVP you just need **one or two short labeled clips** to sanity-check.
Grab either:

- **A couple of Critical Role clips** (a few minutes each) from YouTube + their published
  speaker-labeled transcript. Text source: **CRD3** (https://github.com/RevanthRameshkumar/CRD3 —
  Critical Role transcripts, speaker = player name, text only/no timestamps) or **CritRoleStats**
  (https://www.critrolestats.com).
- **Or the Playing-with-Voices YouTube sources** — repo https://github.com/LiRem101/playing-with-voices
  (MIT) ships `links.txt` (YouTube IDs) + scripts that produce RTTM via forced alignment. It is the
  closest existing TTRPG diarization benchmark; reuse its clips + scripts.

Either way: download ~2 short clips, hand-mark "who speaks when" into a small **gold RTTM** (and a
plain speaker-per-segment list), keep audio local (don't redistribute). That's your entire test set.

For reference only (NOT needed for the MVP): generic diarization benchmarks with audio+labels are
**VoxConverse**, **AMI**, and **VoxCeleb** — mention if you later want harder evaluation.
**FIREBALL** (https://github.com/zhudotexe/FIREBALL) is D&D but **text/game-state, no audio** — not
usable here.

**Reality to state plainly:** there is **no off-the-shelf D&D actual-play audio corpus with speaker
labels.** The MVP sidesteps this by using a tiny hand-labeled clip set; a real system would build one
(YouTube audio + published subtitles via forced alignment).

---

## 7. Evaluation — essentials only

Against the tiny gold set:
- **DER (Diarization Error Rate)** — how good is "who spoke when" (ignores names). Compute with
  `pyannote.metrics` (`DiarizationErrorRate`) from gold RTTM vs predicted RTTM. It reports the
  missed-detection / false-alarm / confusion split for free.
- **Speaker-attribution accuracy** — after enrollment matching, what % of speech time (or segments)
  got the correct **player name**? Simple custom compare of predicted vs gold labels.
- (Optional) **WER** via `jiwer` if you also have a reference text.

```python
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
der = DiarizationErrorRate()
ref = load_rttm("gold.rttm").popitem()[1]; hyp = load_rttm("session.rttm").popitem()[1]
print(der(ref, hyp))   # lower is better
```

That's enough for an MVP. No baselines-grid, no corpus-scale scoring.

---

## 8. Concrete step-by-step (an afternoon)

**Install** (GPU box with CUDA; CPU works with Whisper `small`):
```bash
uv sync --locked
# ffmpeg must be on PATH (brew install ffmpeg / apt install ffmpeg)
# Get a free Hugging Face token, then accept terms on the model pages:
#   huggingface.co/pyannote/speaker-diarization-3.1  and  .../pyannote/embedding
export HF_TOKEN=hf_xxx
```

1. **Preprocess:** `ffmpeg -i session.m4a -ac 1 -ar 16000 session_16k.wav`.
2. **Diarize:** run pyannote `speaker-diarization-3.1` → `session.rttm` (anonymous clusters). Pass
   `num_speakers` if you know the cast size.
3. **ASR:** run Whisper (`small` on CPU, `large-v3` on GPU) → segments with start/end/text.
4. **Merge:** assign each Whisper segment the diarization label it overlaps most → JSON + SRT with
   `SPEAKER_xx` labels. **This is already a usable MVP deliverable.**
5. **(Optional) Enrollment:** if a roster with `enroll_wav`s exists, embed each player + each
   cluster, nearest-centroid cosine match, rename `SPEAKER_xx → player_name`, `unknown` below
   threshold.
6. **Evaluate:** DER (pyannote.metrics) + speaker-attribution accuracy on the 1–2 gold clips.

**Milestones:** M1 audio→anonymous-labeled transcript (steps 1–4). M2 enrollment renaming (step 5).
M3 DER + accuracy numbers on the gold clips (step 6).

**Hardware:** any single 8–12 GB GPU runs Whisper `large-v3` + pyannote comfortably; CPU-only works
with Whisper `small`/`base` (slower). Process offline, not realtime.

---

## 9. Risks + mitigations (essentials)

| Risk | Mitigation |
|---|---|
| **Overlap / crosstalk** (players talk over each other) → merged/confused clusters | pyannote-3.1 handles some overlap out of the box; if multitrack (Craig/Discord) recordings exist, use them — diarization becomes trivial. Accept imperfect overlap handling in MVP. |
| **Character voices split one player into multiple clusters** (the *Playing with Voices* finding: IC speech raises cluster confusion) | Enroll on the player's natural voice; allow `unknown` rather than a forced wrong name; note as a known MVP limitation. Deeper fixes (multi-centroid, cluster merge) are post-MVP. |
| **Wrong speaker count** | Pass `num_speakers` from the roster. |
| **Gold-set alignment / licensing** | Use short clips; hand-mark labels; keep CR/YouTube audio local (don't redistribute). |
| **Enrollment threshold off** → too many/few `unknown` | Expose `unknown_threshold` (default ~0.5); eyeball on the gold clip and adjust. |

---

### Verified links
- Whisper (MIT): https://github.com/openai/whisper
- pyannote.audio (MIT code; gated pretrained models): https://github.com/pyannote/pyannote-audio — pipeline `pyannote/speaker-diarization-3.1`
- CRD3 (CR transcripts, text only): https://github.com/RevanthRameshkumar/CRD3 · CritRoleStats: https://www.critrolestats.com
- Playing with Voices: paper https://arxiv.org/abs/2502.12714 · code https://github.com/LiRem101/playing-with-voices
- FIREBALL (D&D, text/game-state, no audio): https://github.com/zhudotexe/FIREBALL
- (reference only) NeMo: https://github.com/NVIDIA/NeMo · VoxConverse / AMI / VoxCeleb for harder eval later
