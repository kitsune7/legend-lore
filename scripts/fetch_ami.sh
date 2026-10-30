#!/usr/bin/env bash
# Download AMI meeting ES2004a (audio + pyannote's ground-truth RTTM) and make the
# 5-minute slice used by experiments/vad_tuning.py. Local use only.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/ami outputs

curl -sL -o data/ami/ES2004a.rttm \
  "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/ES2004a.rttm"
curl -sL -o data/ami/ES2004a.Mix-Headset.wav \
  "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/ES2004a.Mix-Headset.wav"

# 5-minute slice: trim audio, and clip gold segments to <= 300s.
ffmpeg -y -i data/ami/ES2004a.Mix-Headset.wav -t 300 -ac 1 -ar 16000 \
  data/ami/ES2004a_5min.wav >/dev/null 2>&1
awk '$4 < 300 { d = $5; if ($4 + d > 300) d = 300 - $4; $5 = sprintf("%.2f", d); print }' \
  data/ami/ES2004a.rttm > data/ami/ES2004a_5min.gold.rttm

echo "AMI ready: $(wc -l < data/ami/ES2004a.rttm) gold segments (full), $(wc -l < data/ami/ES2004a_5min.gold.rttm) (5 min)"

# ES2004b: same four people, different meeting. Used ONLY for enrollment voice
# samples in experiments/ami_identify.py, so the test file never leaks into them.
curl -sL -o data/ami/ES2004b.rttm \
  "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/ES2004b.rttm"
curl -sL -o data/ami/ES2004b.Mix-Headset.wav \
  "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004b/audio/ES2004b.Mix-Headset.wav"
echo "Enrollment meeting ready: data/ami/ES2004b.*"
