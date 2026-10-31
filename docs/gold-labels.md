# Gold set (hand-labeled)

Tiny human-verified ground truth to score the pipeline on REAL audio.

For each test clip `NAME`, put two files in `data/gold/` (local, gitignored):

## 1. `NAME.gold.rttm` — "who spoke when" (for DER)
One line per speech segment. Format (space-separated):
```
SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker_label> <NA> <NA>
```
- `<uri>` = the clip's base name (e.g. `cr_clip_16k` — match the predicted RTTM's uri).
- Times in seconds. `<speaker_label>` = the player's name (e.g. `Matt`).

Example:
```
SPEAKER cr_clip_16k 1 0.00 4.20 <NA> <NA> Matt <NA> <NA>
SPEAKER cr_clip_16k 1 4.20 3.10 <NA> <NA> Laura <NA> <NA>
```

## 2. `NAME.gold_labels.json` — speaker per time span (for attribution accuracy)
```json
[
  {"start": 0.0,  "end": 4.2, "player_name": "Matt"},
  {"start": 4.2,  "end": 7.3, "player_name": "Laura"}
]
```

## How to produce these
1. Open the clip's `.srt` (if you fetched subtitles) or just listen in Audacity.
2. Mark segment boundaries and who is talking. Keep it rough — a few-minute clip is fine.
3. If the fetched subtitles already carry speaker names, you can convert them to the
   formats above instead of labeling from scratch.

## Score it
```bash
uv run python -m dnd_notetaker.evaluate \
  --pred-rttm outputs/cr_clip.rttm \
  --gold-rttm data/gold/cr_clip.gold.rttm \
  --pred-json outputs/cr_clip.transcript.json \
  --gold-labels data/gold/cr_clip.gold_labels.json
```
