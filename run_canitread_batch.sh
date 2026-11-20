#!/usr/bin/env bash
# Run the reading test on a list of models, two at a time on one GPU. A model that fails to load is skipped.
# Usage: bash run_canitread_batch.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results/canitread logs
queue() {  # name, then "model|extra args" entries
  local name="$1"; shift
  for entry in "$@"; do
    model="${entry%%|*}"; extra="${entry#*|}"; [ "$extra" = "$entry" ] && extra=""
    echo "=== $model $extra" >> "logs/$name.txt"
    python3 canitread.py --model "$model" $extra --save-answers >> "logs/$name.txt" 2>&1 || echo "FAILED $model" >> "logs/$name.txt"
  done
  echo "QUEUE DONE" >> "logs/$name.txt"
}
queue a "Qwen/Qwen3.5-2B" "zai-org/GLM-OCR" "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" "smolvlm2-500m-64tok" \
        "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen3.5-2B|--precision 4bit" &
queue b "Qwen/Qwen2.5-VL-3B-Instruct" "PaddlePaddle/PaddleOCR-VL-1.6" "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" \
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct" "OpenGVLab/InternVL3-2B-hf" "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
        "zai-org/GLM-OCR|--precision 4bit" &
wait
