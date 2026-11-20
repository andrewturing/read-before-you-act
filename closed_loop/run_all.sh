#!/usr/bin/env bash
# Generate the three demonstration sets, then fine-tune SmolVLA on each (one GPU per run).
# The fourth run repeats the far-camera text condition with the vision-language backbone unfrozen.
# Usage: bash run_all.sh [training steps, default 6000]
set -uo pipefail
cd "$(dirname "$0")"
STEPS="${1:-6000}"
TRAIN="${LEROBOT_TRAIN:-$HOME/.local/bin/lerobot-train}"
rm -rf runs/text_far runs/text_near runs/pointer_far runs/text_far_vlm logs_all_done.txt
python3 patch_lerobot_image_query.py
if [ -z "${SKIP_GEN:-}" ]; then  # set SKIP_GEN=1 to reuse demonstrations that are already on disk
  rm -rf data/reach_*
  for c in text_far text_near pointer_far; do
    python3 make_dataset.py --cond "$c" --episodes 600 --out "data/reach_$c" --preview "data/preview_$c.png" > "logs_gen_$c.txt" 2>&1 &
  done
  wait
fi
train() {  # gpu name dataset batch [extra options]
  local gpu="$1" name="$2" ds="$3" bs="$4"; shift 4
  CUDA_VISIBLE_DEVICES="$gpu" "$TRAIN" --policy.path=lerobot/smolvla_base --dataset.repo_id="local/reach_$ds" \
    --dataset.root="data/reach_$ds" --batch_size="$bs" --steps="$STEPS" --save_freq=2000 --log_freq=100 \
    --output_dir="runs/$name" --job_name="$name" --policy.device=cuda --policy.push_to_hub=false \
    --wandb.enable=false --num_workers=10 "$@" > "logs_train_$name.txt" 2>&1
  echo "EXIT $?" >> "logs_train_$name.txt"
}
train 0 text_far text_far 64 &
train 1 text_near text_near 64 &
train 2 pointer_far pointer_far 64 &
train 3 text_far_vlm text_far 32 --policy.train_expert_only=false --policy.freeze_vision_encoder=false &
wait
echo "ALL DONE" > logs_all_done.txt
