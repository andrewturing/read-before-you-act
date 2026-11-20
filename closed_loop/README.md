# Closed loop: can a robot policy reach the box named by its label?

Two identical cartons with look-alike labels sit on a table. A kinematic end-effector must move onto the one named in the instruction. The cartons differ only in their labels, so reading is the only cue.

## Results (20 November 2026)

SmolVLA fine-tuned from `lerobot/smolvla_base` on 600 scripted demonstrations per condition, 6,000 steps, one NVIDIA L4 per run. 200 test episodes per row; half use label pairs never seen in training.

| Policy is told | Print, camera | Correct | Wrong |
|---|---|---|---|
| Label text | 6 mm, 0.60 m | 54.5% [48–61] | 90 |
| Label text | 6 mm, 0.25 m | 50% [43–57] | 100 |
| Label text, backbone also trained | 6 mm, 0.60 m | 47% [40–54] | 106 |
| Green outline on the true target | 6 mm, 0.60 m | 100% [98–100] | 0 |
| Green outline on the true target | 2 mm, 0.60 m | 100% [98–100] | 0 |
| Outline placed by a reader, first view only | 2 mm, 0.60 m | 63.5% [57–70] | 73 |
| Outline placed by a reader that moves closer until confident, certified for 10% error | 2 mm, from 0.60 m | 96% [92–98] | 8 |
| The same, certified for 5% error | 2 mm, from 0.60 m | 96.5% [93–98] | 5, plus 2 declined |

- **Told the text, the policy guesses.** It goes left and right about equally often, on seen and held-out pairs alike. Moving the camera to 0.25 m and training the vision-language backbone do not help.
- **Told by a pointer, it never misses.** Same architecture, same amount of data and training. The motor part is easy; the reading is not.
- **A small reader fills the gap.** Qwen3.5-2B reads the labels from the full-resolution camera frame and marks its choice for the policy. With 6 mm print it was right in all 200 test episodes from the first view. With 2 mm print it is wrong 40% of the time from the first view, so it moves the camera closer until its confidence clears a threshold calibrated on 400 separate episodes.
- **The certified rates held on test:** 2.5% wrong among executed picks against a 5% target, and 4.0% against 10%.

Brackets are 95% Wilson intervals. Everything is in `results/`: per-episode outcomes in `results/eval`, the reader's transcriptions in `results/reads_*.json`, the threshold search in `results/decisions_*/report.json`, and training logs in `results/logs`.

## What this does not show

- One policy, 600 demonstrations per condition, 6,000 training steps.
- A kinematic end-effector with no contact or grasping.
- Label locations are given to the reader.
- These runs cannot separate "the policy cannot see the text" from "it does not learn to use it". A higher-resolution policy and more data would.
- Simulation only.

## Run it

Needs one or more NVIDIA GPUs with 24 GB, Python 3.10, PyTorch 2.9 and MuJoCo with EGL. Expect about two hours per training run on an L4.

```bash
sudo apt-get install -y python3-dev build-essential libegl1 ffmpeg
```

```bash
pip install "lerobot[smolvla]==0.4.4" mujoco rapidfuzz
```

The reader needs a newer `transformers` than LeRobot allows, so install it into its own folder:

```bash
pip install --target ~/reader_libs "transformers>=5.0.0" num2words "jinja2>=3.1"
```

Then, from this folder:

```bash
python make_dataset.py --test-specs data
```

```bash
bash run_all.sh 6000
```

```bash
PYTHONPATH=~/reader_libs python reader_decide.py read --specs data/test_specs_far_small.json --out results/reads_test_small.json
```

```bash
PYTHONPATH=~/reader_libs python reader_decide.py read --specs data/calibration2_specs_far_small.json --out results/reads_cal2_small.json
```

```bash
python reader_decide.py decide --calibration results/reads_cal2_small.json --test results/reads_test_small.json --out results/decisions_small
```

```bash
bash run_eval.sh
```

`run_all.sh` generates the demonstrations, applies `patch_lerobot_image_query.py` (a speed fix for LeRobot 0.4.4 when frames are stored as images; it returns the same values) and trains the four policies on four GPUs. `run_eval.sh` expects the reader's decision files in `results/decisions_small` and `results/decisions_small_a10` (the second from `decide --alpha 0.10`).

## Files

| File | Purpose |
|---|---|
| `reach_env.py` | The MuJoCo scene, the scripted expert, and what the policy and the reader each see |
| `make_dataset.py` | Demonstrations as LeRobot datasets, plus fixed test and calibration episodes |
| `run_all.sh` | Data generation and the four training runs |
| `reader_decide.py` | The reader's looks at each distance, and the calibrated stop-when-confident rule |
| `eval_policy.py` | Closed-loop evaluation of a trained policy |
| `run_eval.sh` | All evaluations in the table above |
| `patch_lerobot_image_query.py` | Data-loading speed fix for LeRobot 0.4.4 |
