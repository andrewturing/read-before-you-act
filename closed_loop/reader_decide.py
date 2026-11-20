#!/usr/bin/env python
"""Reader side of the closed loop.

  read    for every episode, a movable camera steps closer to the cartons; at each step a small
          reader transcribes both labels from the camera frame and the carton whose text is closest
          to the target is noted, with the reader's confidence;
  decide  a commit threshold is calibrated on one set of episodes so that wrong choices among
          committed episodes stay under a target, then applied to the test episodes: commit at the
          first view whose confidence reaches the threshold, decline if none does.

`read` needs the reader's own environment (transformers 5); `decide` needs nothing but Python.

Usage:
  python reader_decide.py read --specs data/test_specs_far.json --out results/reads_test.json
  python reader_decide.py decide --calibration results/reads_cal.json --test results/reads_test.json \
      --out results/decisions
"""
import argparse
import json
import math
import os
import sys

LADDER = [1.0, 0.75, 0.55, 0.42]  # camera distance as a share of the starting distance (0.60 m -> 0.25 m)
GRID = [-0.02, -0.03, -0.05, -0.07, -0.10, -0.15, -0.20, -0.30, -0.40, -0.60, -0.80, -1.00, -1.50, -2.50, -1e9]


def read(args):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from rapidfuzz.distance import Levenshtein

    from common import Reader, norm_text, upscale_min_side
    from reach_env import ReachEnv

    specs = json.load(open(args.specs))
    specs = specs[args.start: args.start + args.limit] if args.limit else specs[args.start:]
    reader = Reader(args.model)
    env = ReachEnv(os.path.join(os.path.dirname(os.path.abspath(args.specs)), "_tex"))
    out = []
    for i, spec in enumerate(specs):
        target = None
        looks = []
        for ratio in LADDER:
            s = dict(spec, cam_dist=round(spec["cam_dist"] * ratio, 4))
            env.reset(s)
            target = env.target_text(s)
            preds, confs = [], []
            for crop in env.label_crops(env.render()):
                text, conf = reader.generate(upscale_min_side(crop, 64), reader.read_prompt(), max_new_tokens=32)
                preds.append(text)
                confs.append(conf["mean_lp"] if any(ch.isalnum() for ch in text) else -1e9)  # empty = not read
            dist = [Levenshtein.normalized_distance(norm_text(p), norm_text(target)) for p in preds]
            side = -1 if dist[0] == dist[1] else (0 if dist[0] < dist[1] else 1)
            looks.append(dict(cam_dist=s["cam_dist"], text_px=round(env.text_px(), 1), side=side, conf=round(min(confs), 4),
                              pred=preds))
        out.append(dict(i=i, split=spec.get("split", ""), target_side=spec["target_side"], target=target, looks=looks))
        if (i + 1) % 25 == 0:
            first = sum(o["looks"][0]["side"] == o["target_side"] for o in out)
            print(f"  {i + 1}/{len(specs)}  right from the first view: {first}", flush=True)
    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(model=args.model, ladder=LADDER, episodes=out), open(args.out, "w"), ensure_ascii=False, indent=0)
    print("wrote", args.out)


def run_rule(episode, tau):
    """Commit at the first view whose confidence reaches tau. Returns (chosen side or -1, views used)."""
    for k, look in enumerate(episode["looks"]):
        if look["conf"] >= tau and look["side"] != -1:
            return look["side"], k + 1
    return -1, len(episode["looks"])


def binom_cdf(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))


def calibrate(episodes, alpha, delta):
    """Fixed-sequence testing from strict to loose thresholds (Learn then Test). The p-value for
    'wrong among committed exceeds alpha' is a binomial tail, valid given which episodes commit."""
    # Testing starts at the strictest threshold with enough commits for a clean record to certify it:
    # the smallest n with (1 - alpha)^n <= delta. This depends on alpha and delta only, not on errors.
    min_committed = math.ceil(math.log(delta) / math.log(1 - alpha))
    rows, chosen, started, stopped = [], None, False, False
    for tau in GRID:
        picks = [run_rule(e, tau)[0] for e in episodes]
        n = sum(p != -1 for p in picks)
        err = sum(p != -1 and p != e["target_side"] for p, e in zip(picks, episodes))
        p_value = binom_cdf(err, n, alpha) if n else 1.0
        if not started and n >= min_committed:
            started = True
        if started and not stopped:
            if p_value <= delta:
                chosen = tau
            else:
                stopped = True
        rows.append(dict(tau=tau, committed=n, wrong=err, p_value=round(p_value, 4), accepted=chosen == tau))
    return chosen, rows


def apply(episodes, tau):
    picks = [run_rule(e, tau) for e in episodes]
    n = len(episodes)
    committed = [(p, e) for (p, _), e in zip(picks, episodes) if p != -1]
    wrong = sum(p != e["target_side"] for p, e in committed)
    return dict(n=n, committed=len(committed), declined=n - len(committed), wrong=wrong,
                wrong_among_committed=round(wrong / len(committed), 4) if committed else None,
                mean_views=round(sum(v for _, v in picks) / n, 3),
                decided_at_first_view=sum(v == 1 and p != -1 for p, v in picks)), [p for p, _ in picks]


def decide(args):
    cal = [e for path in args.calibration.split(",") for e in json.load(open(path))["episodes"]]
    test = json.load(open(args.test))["episodes"]
    os.makedirs(args.out, exist_ok=True)
    report = dict(alpha=args.alpha, delta=args.delta, calibration_episodes=len(cal), test_episodes=len(test), rules={})

    # Baselines: always act on a fixed view.
    for k, ratio in enumerate(LADDER):
        picks = [e["looks"][k]["side"] for e in test]
        wrong = sum(p != e["target_side"] for p, e in zip(picks, test))
        report["rules"][f"fixed_view_{k + 1}"] = dict(n=len(test), views=k + 1, wrong=wrong, wrong_rate=round(wrong / len(test), 4))
        json.dump([p if p != -1 else 0 for p in picks], open(os.path.join(args.out, f"fixed_view_{k + 1}.json"), "w"))

    tau, rows = calibrate(cal, args.alpha, args.delta)
    report["calibration_table"] = rows
    report["tau"] = tau
    if tau is None:
        print("no threshold could be certified on the calibration episodes")
    else:
        stats, picks = apply(test, tau)
        report["rules"]["stop_when_confident"] = stats
        json.dump(picks, open(os.path.join(args.out, "stop_when_confident.json"), "w"))
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "calibration_table"}, indent=1))
    for r in rows:
        print(r)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--specs", required=True)
    r.add_argument("--model", default="Qwen/Qwen3.5-2B")
    r.add_argument("--out", required=True)
    r.add_argument("--start", type=int, default=0, help="first episode to read, to split the work across GPUs")
    r.add_argument("--limit", type=int, default=0)
    d = sub.add_parser("decide")
    d.add_argument("--calibration", required=True, help="one or more files from `read`, comma-separated")
    d.add_argument("--test", required=True)
    d.add_argument("--alpha", type=float, default=0.05, help="target share of wrong choices among committed episodes")
    d.add_argument("--delta", type=float, default=0.10, help="allowed chance that calibration certifies a bad threshold")
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    read(args) if args.cmd == "read" else decide(args)


if __name__ == "__main__":
    main()
