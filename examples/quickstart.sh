#!/usr/bin/env bash
# Run every capability of QuanAI with no API key and no GPU.
# Usage: bash examples/quickstart.sh
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m quanai doctor

echo '\n=== 1. deck ==='
python3 -m quanai run "create a 7 slide deck about solar energy adoption in cities"

echo '\n=== 2. image ==='
python3 -m quanai run "generate an abstract poster image about deep ocean exploration" | head -20

echo '\n=== 3. video ==='
python3 -m quanai run "make a short promo video about our AI agent" | head -25

echo '\n=== 4. desktop app ==='
python3 -m quanai run "build a desktop app for tracking my daily habits" | head -20

echo '\n=== 5. mobile app ==='
python3 -m quanai run "create a mobile app for habit tracking" | head -20

echo '\n=== 6. multimodal input ==='
# make a tiny sample clip if one is not already present
mkdir -p samples
if [ ! -f samples/demo.wav ]; then
  ffmpeg -y -loglevel error -f lavfi -i "sine=frequency=440:duration=2" -ac 1 samples/demo.wav 2>/dev/null || \
    python3 -c "from quanai.util import ffmpeg_bin, run_cmd; run_cmd([ffmpeg_bin(),'-y','-loglevel','error','-f','lavfi','-i','sine=frequency=440:duration=2','-ac','1','samples/demo.wav'])"
fi
python3 -m quanai run "analyse this audio and summarise what is in it" --input samples/demo.wav | head -20

echo '\nDone. Artifacts live under ./workspace/obj-*/ and each run has REPORT.md.'
