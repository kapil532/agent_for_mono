#!/usr/bin/env bash
# Launch MonoceptGPT
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "No .venv found. Run ./install.sh first."
  exit 1
fi

echo "Starting MonoceptGPT on http://localhost:8080 ..."
./.venv/bin/python app.py
