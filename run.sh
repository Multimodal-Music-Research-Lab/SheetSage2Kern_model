#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash run.sh INPUT_DIR OUTPUT_DIR" >&2
  exit 1
fi

input_dir=$(cd "$1" && pwd)
mkdir -p "$2"
output_dir=$(cd "$2" && pwd)
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

docker build \
  --build-arg HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  -t mirex_a2s "$repo_dir"

docker run --rm --gpus all \
  -v "$input_dir:/input:ro" \
  -v "$output_dir:/output" \
  mirex_a2s \
  --input_dir /input \
  --output_dir /output
