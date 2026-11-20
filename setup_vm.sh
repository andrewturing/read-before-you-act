#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu GPU machine: Python packages, system fonts, two open handwriting-style fonts.
set -euo pipefail
cd "$(dirname "$0")"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fonts-noto-core fonts-noto-cjk libegl1 libgl1 libosmesa6 >/dev/null
python3 -m pip install -q -U -r requirements-scenes.txt
mkdir -p fonts data results
# Open Font License fonts from the Google Fonts repository.
curl -fsSL -o fonts/PatrickHand-Regular.ttf https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf
curl -fsSL -o fonts/MaShanZheng-Regular.ttf https://github.com/google/fonts/raw/main/ofl/mashanzheng/MaShanZheng-Regular.ttf
python3 - <<'EOF'
import torch, transformers, mujoco
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "mujoco", mujoco.__version__)
EOF
ls -la fonts
