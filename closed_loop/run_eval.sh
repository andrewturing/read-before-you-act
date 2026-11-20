#!/usr/bin/env bash
# Closed-loop evaluation of the four trained policies on the 200 test episodes, spread over four GPUs.
# Usage: bash run_eval.sh [checkpoint name, default "last"]
set -uo pipefail
cd "$(dirname "$0")"
CK="${1:-last}"
mkdir -p results/eval logs
rm -f logs_eval_done.txt
ev() {  # gpu name policy mode specs [pointer file]
  local gpu="$1" name="$2" pol="$3" mode="$4" specs="$5" pointer="${6:-oracle}"
  CUDA_VISIBLE_DEVICES="$gpu" python3 eval_policy.py --policy "runs/$pol/checkpoints/$CK/pretrained_model" --mode "$mode" \
    --specs "data/$specs.json" --pointer "$pointer" --name "$name" --out "results/eval/$name.json" \
    --save-frames data/frames > "logs/eval_$name.txt" 2>&1 || echo "FAILED $name" >> "logs/eval_$name.txt"
}
( ev 0 text_far text_far text test_specs_far; ev 0 pointer_oracle pointer_far pointer test_specs_far ) &
( ev 1 text_near text_near text test_specs_near; ev 1 pointer_small_oracle pointer_far pointer test_specs_far_small ) &
( ev 2 text_far_vlm text_far_vlm text test_specs_far; ev 2 pointer_small_first_view pointer_far pointer test_specs_far_small results/decisions_small/fixed_view_1.json ) &
( ev 3 pointer_small_stop_when_confident pointer_far pointer test_specs_far_small results/decisions_small/stop_when_confident.json; ev 3 pointer_small_stop_a10 pointer_far pointer test_specs_far_small results/decisions_small_a10/stop_when_confident.json ) &
wait
grep -h -E "\[(all|seen|held_out)\]|FAILED" logs/eval_*.txt > results/eval/summary.txt
echo "EVAL DONE" > logs_eval_done.txt
