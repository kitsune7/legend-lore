#!/usr/bin/env bash
# Fetch a SHORT slice of a YouTube actual-play video as 16kHz mono WAV for LOCAL
# research/eval only (do not redistribute). Also grabs subtitles if available.
#
# Usage:
#   bash scripts/fetch_clip.sh <YOUTUBE_URL_OR_ID> <START> <DURATION> [name]
# Example (90s starting at 2 minutes):
#   bash scripts/fetch_clip.sh "https://youtu.be/XXXX" 00:02:00 00:01:30 cr_clip
set -euo pipefail
cd "$(dirname "$0")/.."

URL="${1:?need a YouTube URL or video id}"
START="${2:?need a start time, e.g. 00:02:00 or 120}"
DUR="${3:?need a duration, e.g. 00:01:30 or 90}"
NAME="${4:-clip}"
mkdir -p data

# player_client override avoids YouTube's SABR-streaming block (web client fails).
PY="uv run --locked python -m yt_dlp --extractor-args youtube:player_client=ios,android,tv"

# Compute the end time for --download-sections (accepts seconds or hh:mm:ss).
END=$(uv run --locked python - "$START" "$DUR" <<'PY'
import sys
def sec(t):
    if ":" in t:
        p=[float(x) for x in t.split(":")]
        while len(p)<3: p.insert(0,0.0)
        return p[0]*3600+p[1]*60+p[2]
    return float(t)
print(f"{sec(sys.argv[1])+sec(sys.argv[2]):.0f}")
PY
)
START_S=$(uv run --locked python - "$START" <<'PY'
import sys
t=sys.argv[1]
if ":" in t:
    p=[float(x) for x in t.split(":")]
    while len(p)<3: p.insert(0,0.0)
    print(f"{p[0]*3600+p[1]*60+p[2]:.0f}")
else:
    print(f"{float(t):.0f}")
PY
)

echo "Fetching section ${START_S}s-${END}s from $URL ..."
# Download only the requested section as wav, 16kHz mono.
$PY -x --audio-format wav \
    --download-sections "*${START_S}-${END}" \
    --force-keyframes-at-cuts \
    --postprocessor-args "ffmpeg:-ac 1 -ar 16000" \
    -o "data/${NAME}.%(ext)s" \
    "$URL"

# Try to grab manual (human) English subtitles — useful for speaker ground truth.
$PY --skip-download --write-subs --sub-langs "en.*" --convert-subs srt \
    -o "data/${NAME}.%(ext)s" "$URL" || echo "(no manual subs available)"

echo "Done. Audio: data/${NAME}.wav"
ls -la data/${NAME}.* 2>/dev/null || true
