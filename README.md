# Legend Lore

D&D actual-play audio → **timestamped transcript labeled with which player is speaking**.
CS6352 project (Bradshaw, Walton, Jackson). This is the proof of concept: pretrained models
only, no training. See [`docs/plan.md`](docs/plan.md) for the design.

Pipeline: `preprocess → ASR → diarize → merge → (optional) enrollment → evaluate`

- **ASR** — OpenAI Whisper produces the text, with per-word timestamps. (It runs before
  diarization on purpose — see the note in `pipeline.py`.)
- **Diarization** — pyannote `speaker-diarization-3.1` answers *who spoke when* as anonymous
  clusters (`SPEAKER_00`, `SPEAKER_01`, …).
- **Merge** — each word gets the cluster it overlaps most in time, and lines are split where
  the speaker changes.
- **Enrollment (optional)** — renames clusters to real players by cosine-matching each cluster
  against a short voice sample per player; below `unknown_threshold` it says `unknown`.

In-character / out-of-character and per-character labels are not built yet — the output JSON
leaves `ic_ooc` and `character_name` as `null` for those later stages.

## Setup

Requires Python 3.9+ and `ffmpeg` on PATH (`brew install ffmpeg`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Diarization and enrollment use **gated** pyannote models, so each person needs a free
Hugging Face token:

1. Create a **Read** token at https://huggingface.co/settings/tokens
2. While logged in, accept the terms on all three model pages:
   [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) ·
   [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) ·
   [embedding](https://huggingface.co/pyannote/embedding)
3. `cp .env.example .env` and paste the token in (or `export HF_TOKEN=...`). `.env` is gitignored.

Without a token the pipeline still runs, but labels everything `SPEAKER_00`.

## Run

```bash
# synthetic 2-voice smoke test (macOS `say`, no downloads)
bash scripts/make_sample_audio.sh
.venv/bin/python -m dnd_notetaker.cli --audio data/sample_session.wav --model base \
    --roster config/roster.sample.json

# a real actual-play clip: <url> <start> <duration> <name>
bash scripts/fetch_clip.sh "https://www.youtube.com/watch?v=XXXX" 00:30:00 00:02:00 cr_clip
.venv/bin/python -m dnd_notetaker.cli --audio data/cr_clip.wav --model base
```

Outputs go to `outputs/`: `<name>.transcript.json` (full schema), `<name>.srt` (readable),
`<name>.rttm` (for scoring). Whisper sizes: `base`/`small` are fine on CPU; `large-v3` wants a GPU.

Options:
- `--drop-silence` — remove lines that are mostly silence (often Whisper inventing text). Off by
  default because it also removes real speech the diarizer missed. Lines with no detected
  speech are otherwise kept and labeled `unknown`.
- `--min-speakers N` — a floor on the speaker count. Unlike `--num-speakers` it doesn't force an
  exact count, which made results worse on AMI.

## Evaluate

Score a run against hand-labeled ground truth (format in [`docs/gold-labels.md`](docs/gold-labels.md)):

```bash
.venv/bin/python -m dnd_notetaker.evaluate \
    --pred-rttm outputs/cr_clip.rttm --gold-rttm data/gold/cr_clip.gold.rttm \
    --pred-json outputs/cr_clip.transcript.json --gold-labels data/gold/cr_clip.gold_labels.json
```

## Experiments (AMI benchmark)

AMI is a meeting corpus that ships ground-truth speaker labels, so we used it to verify the
diarization. Run `bash scripts/fetch_ami.sh` first, then from the repo root:

| Script | Question | Result |
|---|---|---|
| `python -m experiments.ami_compare` | Does forcing the speaker count help? | No — 19.4% DER auto vs 39.9% forced |
| `python -m experiments.overlap_eval` | How much error is overlapping speech? | ~8 points: 19.4% → 11.3% without overlap |
| `python -m experiments.vad_tuning` | Can tuning cut missed speech? | No — misses stuck at ~11% in every config |
| `python -m experiments.ami_identify` | Does identifying roster players directly beat clustering? | No — 20.4% vs 19.4% DER; a quiet speaker talking over others is still missed |

(Use `.venv/bin/python`. `ami_compare` and `vad_tuning` take ~20 min on CPU; `ami_identify` ~2 min.)

`dnd_notetaker/identify.py` is that experimental alternative to clustering: it labels each
1.5s window with the closest enrolled player (and allows several voice samples per player, e.g.
a character voice). It isn't wired into the CLI yet — we'll evaluate it on D&D audio first.

## Layout

```
dnd_notetaker/   pipeline package (preprocess, asr, diarize, merge, enroll, evaluate, cli, hf;
                 identify = experimental)
experiments/     one-off AMI experiments, each documents its own result
scripts/         sample audio, YouTube clip fetch, AMI download (+ ES2004b for enrollment)
config/          roster files (player names + enrollment wavs)
docs/            plan, gold-label format
data/, outputs/  local only — gitignored, never commit audio
```

Audio from actual-play shows is for local research only; don't commit or redistribute it.
