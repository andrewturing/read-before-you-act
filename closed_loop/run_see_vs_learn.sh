#!/usr/bin/env bash
# Second closed-loop study: does the policy fail because it cannot see the text, or because it does
# not learn to use it? The text_crops runs give the policy, besides the usual camera view, a close-up of
# each label as two extra camera inputs, with the text far larger than the backbone needs to read it.
#
#   text_crops           look-alike names, 600 demonstrations
#   text_crops_distinct  clearly different names, 600 demonstrations (can it use text at all?)
#   text_far_distinct    the same episodes without the close-ups (does seeing the text matter?)
#   text_crops_more      look-alike names, 1500 demonstrations
#   text_crops_vlm       look-alike names, 600 demonstrations, vision-language backbone also trained
#                        (batch 12: batch 16 runs out of memory on a 24 GB L4 with three cameras)
#   text_crops_step2000  text_crops at step 2000 (64K samples), to compare with text_crops_vlm (72K samples)
#
# Usage: bash run_see_vs_learn.sh [training steps, default 6000]    (needs four GPUs)
set -uo pipefail
cd "$(dirname "$0")"
STEPS="${1:-6000}"
TRAIN="${LEROBOT_TRAIN:-$HOME/.local/bin/lerobot-train}"
mkdir -p logs results/eval
rm -f logs_see_vs_learn_done.txt
python3 patch_lerobot_image_query.py
python3 make_dataset.py --test-specs data > logs/test_specs.txt 2>&1

gen() {  # condition episodes
  rm -rf "data/reach_$1"
  python3 make_dataset.py --cond "$1" --episodes "$2" --out "data/reach_$1" --preview "data/preview_$1.png" > "logs/gen_$1.txt" 2>&1
}
train() {  # gpu name dataset batch [extra options]
  local gpu="$1" name="$2" ds="$3" bs="$4"; shift 4
  rm -rf "runs/$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$TRAIN" --policy.path=lerobot/smolvla_base --dataset.repo_id="local/reach_$ds" \
    --dataset.root="data/reach_$ds" --batch_size="$bs" --steps="$STEPS" --save_freq=2000 --log_freq=100 \
    --output_dir="runs/$name" --job_name="$name" --policy.device=cuda --policy.push_to_hub=false \
    --wandb.enable=false --num_workers=10 "$@" > "logs/train_$name.txt" 2>&1
  echo "EXIT $?" >> "logs/train_$name.txt"
}
ev() {  # gpu name specs [checkpoint, default last] [result name, default name]; text_crops* runs get the close-ups
  local out="${5:-$2}" crops=""
  [[ "$2" == text_crops* ]] && crops="--crops"
  CUDA_VISIBLE_DEVICES="$1" python3 eval_policy.py --policy "runs/$2/checkpoints/${4:-last}/pretrained_model" --mode text \
    $crops --specs "data/$3.json" --name "$out" --out "results/eval/$out.json" --save-frames data/frames \
    > "logs/eval_$out.txt" 2>&1 || echo "FAILED $out" >> "logs/eval_$out.txt"
}

( gen text_crops 600
  ( train 0 text_crops text_crops 32; ev 0 text_crops test_specs_far
    ev 0 text_crops test_specs_far 002000 text_crops_step2000 ) &
  ( PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True train 3 text_crops_vlm text_crops 12 \
      --policy.train_expert_only=false --policy.freeze_vision_encoder=false
    ev 3 text_crops_vlm test_specs_far ) &
  wait ) &
( gen text_crops_distinct 600; gen text_far_distinct 600
  train 1 text_crops_distinct text_crops_distinct 32; ev 1 text_crops_distinct test_specs_far_distinct
  train 1 text_far_distinct text_far_distinct 32; ev 1 text_far_distinct test_specs_far_distinct ) &
( gen text_crops_more 1500
  train 2 text_crops_more text_crops_more 32; ev 2 text_crops_more test_specs_far ) &
wait
grep -h -E "\[(all|seen|held_out)\]|FAILED" logs/eval_text_crops*.txt logs/eval_text_far_distinct.txt \
  > results/eval/summary_see_vs_learn.txt
echo "DONE" > logs_see_vs_learn_done.txt
