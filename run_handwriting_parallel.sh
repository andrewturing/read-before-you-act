#!/usr/bin/env bash
# Run the handwriting probe as one worker per model so the four models overlap on the two GPUs.
# Usage: bash run_handwriting_parallel.sh [lines per dataset, default 120]
set -euo pipefail
cd "$(dirname "$0")"
N="${1:-120}"
pkill -f '^python3 eval_handwriting' || true
sleep 2
rm -f results/handwriting.jsonl results/hw_*.jsonl
launch() {  # gpu model
  CUDA_VISIBLE_DEVICES="$1" nohup python3 eval_handwriting.py --n "$N" --models "$2" \
    --out "results/hw_$2.jsonl" > "logs_hw_$2.txt" 2>&1 &
}
launch 0 paddleocr-vl-1.6
launch 0 glm-ocr
launch 0 qwen2.5-vl-3b
launch 1 qwen3.5-2b
echo "started 4 handwriting workers with n=$N"
