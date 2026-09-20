#!/usr/bin/env bash
# NEELDRIK — one-command start (macOS / Linux)
set -e
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt 2>/dev/null || true
if [ ! -f models/unet_oilspill.npz ]; then
  echo "No trained model yet — training now (about 25 minutes)…"
  python3 train.py
fi
python3 run.py "$@"
