#!/usr/bin/env python
"""Closed-loop evaluation of a trained SmolVLA policy on reach-by-label.

  --mode text      the instruction names the label text
  --mode pointer   the instruction says "reach the box marked in green"; the outline is put on the
                   true target (--pointer oracle) or on the carton a reader chose (--pointer FILE,
                   a JSON list with 0, 1 or -1 per test episode, where -1 means the reader declined)

An episode ends as correct (end-effector on the target carton), wrong (on the other carton),
none (on neither) or declined (no attempt, pointer mode only).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from reach_env import EPISODE_STEPS, POINTER_TASK, ReachEnv, outcome, text_task


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="folder holding the trained policy (…/pretrained_model)")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--mode", choices=["text", "pointer"], required=True)
    ap.add_argument("--pointer", default="oracle")
    ap.add_argument("--replan", type=int, default=10, help="actions executed before the policy looks again")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--save-frames", metavar="DIR", help="save what the policy saw at the start of the first episodes")
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(args.policy)
    policy.to(device).eval()
    policy.config.n_action_steps = args.replan
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.policy)

    specs = json.load(open(args.specs))
    chosen = None if args.pointer == "oracle" else json.load(open(args.pointer))
    if args.limit:
        specs = specs[: args.limit]
    env = ReachEnv(os.path.join(os.path.dirname(os.path.abspath(args.specs)), "_tex"))
    rows, t0 = [], time.time()
    for i, spec in enumerate(specs):
        env.reset(spec)
        policy.reset()
        if args.mode == "pointer":
            side = spec["target_side"] if chosen is None else chosen[i]
            task = POINTER_TASK
        else:
            side, task = None, text_task(env.target_text(spec))
        if side == -1:
            rows.append(dict(i=i, split=spec.get("split", ""), result="declined"))
            continue
        for t in range(EPISODE_STEPS):
            frame = env.render()
            image = env.observation(frame, side)
            if args.save_frames and i < 4 and t == 0:
                os.makedirs(args.save_frames, exist_ok=True)
                from PIL import Image
                Image.fromarray(image).save(os.path.join(args.save_frames, f"{args.name or args.mode}_{i}.png"))
            obs = {"observation.images.camera1": image, "observation.state": env.pos()}
            action = predict_action(obs, policy, device, pre, post, use_amp=False, task=task,
                                    robot_type="kinematic_gripper")
            env.step(np.asarray(action.detach().cpu(), dtype=np.float64).reshape(-1)[:3])
        got = outcome(env.pos())
        result = "correct" if got == spec["target_side"] else ("none" if got == -1 else "wrong")
        rows.append(dict(i=i, split=spec.get("split", ""), result=result, final=[round(float(v), 4) for v in env.pos()]))
        if (i + 1) % 25 == 0:
            done = [r["result"] for r in rows]
            print(f"  {i + 1}/{len(specs)}  correct {done.count('correct')}  wrong {done.count('wrong')}  "
                  f"none {done.count('none')}  declined {done.count('declined')}", flush=True)
    env.close()

    summary = {}
    for split in ("all", "seen", "held_out"):
        g = [r for r in rows if split == "all" or r["split"] == split]
        if not g:
            continue
        n = len(g)
        counts = {k: sum(r["result"] == k for r in g) for k in ("correct", "wrong", "none", "declined")}
        acted = counts["correct"] + counts["wrong"]
        lo, hi = wilson(counts["correct"], n)
        summary[split] = dict(n=n, **counts, correct_rate=round(counts["correct"] / n, 4), low=round(lo, 4), high=round(hi, 4),
                              wrong_among_reached=round(counts["wrong"] / acted, 4) if acted else None)
    out = dict(name=args.name, policy=args.policy, mode=args.mode, pointer=args.pointer, specs=args.specs,
               replan=args.replan, seconds=round(time.time() - t0), summary=summary, episodes=rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for split, s in summary.items():
        print(f"{args.name or args.mode} [{split}] n={s['n']}: correct {100 * s['correct_rate']:.0f}% "
              f"[{100 * s['low']:.0f}–{100 * s['high']:.0f}], wrong {s['wrong']}, none {s['none']}, declined {s['declined']}")


if __name__ == "__main__":
    main()
