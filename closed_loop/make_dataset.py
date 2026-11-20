#!/usr/bin/env python
"""Scripted demonstrations for reach-by-label, written as a LeRobot dataset, plus test episode lists.

Conditions:
  text_far      the instruction names the label text; camera 0.6 m away (text about 9 px on the sensor)
  text_near     the same, camera 0.25 m away (text about 22 px on the sensor)
  pointer_far   the target carton has a green outline and the instruction says to reach the marked box

Usage: python make_dataset.py --cond text_far --episodes 600 --out data/reach_text_far
       python make_dataset.py --test-specs data
"""
import argparse
import json
import os

import numpy as np

from reach_env import (EPISODE_STEPS, FPS, OBS_H, OBS_W, POINTER_TASK, TEST_PAIRS, TRAIN_PAIRS, ReachEnv, expert_path,
                       make_specs, text_task)

# condition: (camera, pointer, label close-ups as extra cameras, pairing, seed)
CONDS = {
    "text_far": ("far", False, False, "lookalike", 11),
    "text_near": ("near", False, False, "lookalike", 12),
    "pointer_far": ("far", True, False, "lookalike", 13),
    "text_crops": ("far", False, True, "lookalike", 14),
    "text_crops_distinct": ("far", False, True, "distinct", 15),
    "text_crops_more": ("far", False, True, "lookalike", 16),
    "text_far_distinct": ("far", False, False, "distinct", 15),  # text_crops_distinct without the close-ups
}


def write_test_specs(folder):
    """200 fixed test episodes per camera distance: 100 with label pairs seen in training, 100 with held-out pairs."""
    os.makedirs(folder, exist_ok=True)
    for view, seed in (("far", 101), ("near", 102)):
        specs = [dict(s, split="seen") for s in make_specs(100, TRAIN_PAIRS, view, seed)]
        specs += [dict(s, split="held_out") for s in make_specs(100, TEST_PAIRS, view, seed + 50)]
        json.dump(specs, open(os.path.join(folder, f"test_specs_{view}.json"), "w"), indent=0)
        print("wrote", len(specs), "test episodes for the", view, "camera")
    # Episodes used only to calibrate the reader's commit threshold: all pairs, far camera.
    cal = make_specs(400, TRAIN_PAIRS + TEST_PAIRS, "far", 301)
    json.dump(cal, open(os.path.join(folder, "calibration_specs_far.json"), "w"), indent=0)
    print("wrote", len(cal), "calibration episodes")
    # The same far-camera episodes with 2 mm print, where one look is not always enough.
    for name in ("test_specs_far", "calibration_specs_far"):
        specs = [dict(s, cap_m=0.002) for s in json.load(open(os.path.join(folder, name + ".json")))]
        json.dump(specs, open(os.path.join(folder, name + "_small.json"), "w"), indent=0)
    print("wrote the small-print copies")
    specs = [dict(s, split="seen") for s in make_specs(100, TRAIN_PAIRS, "far", 103, "distinct")]
    specs += [dict(s, split="held_out") for s in make_specs(100, TEST_PAIRS, "far", 153, "distinct")]
    json.dump(specs, open(os.path.join(folder, "test_specs_far_distinct.json"), "w"), indent=0)
    print("wrote the distinct-name test episodes")
    # A second, independent small-print calibration set (see results notes: the first one was used
    # to find a flaw in the threshold search and is not used for the reported threshold).
    cal2 = [dict(s, cap_m=0.002) for s in make_specs(400, TRAIN_PAIRS + TEST_PAIRS, "far", 302)]
    json.dump(cal2, open(os.path.join(folder, "calibration2_specs_far_small.json"), "w"), indent=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=list(CONDS))
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--out")
    ap.add_argument("--test-specs", metavar="FOLDER")
    ap.add_argument("--preview", metavar="PNG", help="also save the first frame of the first episode")
    args = ap.parse_args()
    if args.test_specs:
        write_test_specs(args.test_specs)
        return
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    view, pointer, crops, pairing, seed = CONDS[args.cond]
    features = {
        "observation.images.camera1": {"dtype": "image", "shape": (OBS_H, OBS_W, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "action": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    }
    if crops:  # close-ups of the left and right label as two more cameras
        for cam in ("camera2", "camera3"):
            features[f"observation.images.{cam}"] = {"dtype": "image", "shape": (OBS_H, OBS_W, 3),
                                                     "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=f"local/reach_{args.cond}", fps=FPS, features=features, root=args.out,
                               robot_type="kinematic_gripper", use_videos=False, image_writer_threads=8)
    env = ReachEnv(os.path.join(os.path.dirname(os.path.abspath(args.out)), "_tex"))
    for ep, spec in enumerate(make_specs(args.episodes, TRAIN_PAIRS, view, seed, pairing)):
        env.reset(spec)
        path = expert_path(spec)
        task = POINTER_TASK if pointer else text_task(env.target_text(spec))
        for t in range(EPISODE_STEPS):
            frame = env.render()
            image = env.observation(frame, spec["target_side"] if pointer else None)
            item = {"observation.images.camera1": image, "observation.state": env.pos(), "action": path[t + 1],
                    "task": task}
            if crops:
                item["observation.images.camera2"], item["observation.images.camera3"] = env.label_views(frame)
            if ep == 0 and t == 0 and args.preview:  # exactly what the policy is given, side by side
                from PIL import Image
                imgs = [Image.fromarray(v) for k, v in item.items() if k.startswith("observation.images")]
                sheet = Image.new("RGB", (OBS_W * len(imgs), OBS_H), "white")
                for i, im in enumerate(imgs):
                    sheet.paste(im, (i * OBS_W, 0))
                sheet.save(args.preview)
            ds.add_frame(item)
            env.step(path[t + 1])
        ds.save_episode()
        if (ep + 1) % 50 == 0:
            print(f"{args.cond}: {ep + 1}/{args.episodes} episodes", flush=True)
    ds.finalize()
    env.close()
    extra = f" and {env.text_px_in_label_view():.0f} px in each label close-up" if crops else ""
    print(f"wrote {args.episodes} episodes to {args.out}; text is about {env.text_px():.0f} px tall on the sensor, "
          f"{env.text_px_in_input():.0f} px in the policy's main view{extra}")


if __name__ == "__main__":
    main()
