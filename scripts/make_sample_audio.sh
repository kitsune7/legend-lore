#!/usr/bin/env bash
# Generate a tiny synthetic two-"player" D&D clip using macOS `say` + ffmpeg.
# No downloads, no licensing issues — just a local smoke test for the pipeline.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data data/enroll

say -v Daniel  -o /tmp/dm1.aiff    "You arrive at the edge of a misty ravine. Roll me a perception check."
say -v Samantha -o /tmp/pl1.aiff   "I rolled a seventeen. Do I see anything moving in the fog below?"
say -v Daniel  -o /tmp/dm2.aiff    "You spot a rope bridge, half rotted, swaying in the wind."
say -v Samantha -o /tmp/pl2.aiff   "Great. I tie myself off and start across, carefully."

# concatenate into one "session"
for f in dm1 pl1 dm2 pl2; do ffmpeg -y -i /tmp/$f.aiff -ar 16000 -ac 1 /tmp/$f.wav >/dev/null 2>&1; done
printf "file '/tmp/dm1.wav'\nfile '/tmp/pl1.wav'\nfile '/tmp/dm2.wav'\nfile '/tmp/pl2.wav'\n" > /tmp/concat.txt
ffmpeg -y -f concat -safe 0 -i /tmp/concat.txt -ar 16000 -ac 1 data/sample_session.wav >/dev/null 2>&1

# enrollment samples (one clean clip per "player")
say -v Daniel   -o /tmp/enr_dm.aiff  "This is the dungeon master speaking in my normal voice for enrollment."
say -v Samantha -o /tmp/enr_pl.aiff  "This is the player speaking in my normal voice for enrollment."
ffmpeg -y -i /tmp/enr_dm.aiff -ar 16000 -ac 1 data/enroll/dm.wav   >/dev/null 2>&1
ffmpeg -y -i /tmp/enr_pl.aiff -ar 16000 -ac 1 data/enroll/player.wav >/dev/null 2>&1

echo "Wrote data/sample_session.wav and data/enroll/{dm,player}.wav"
