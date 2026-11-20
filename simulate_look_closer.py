#!/usr/bin/env python
"""Offline simulation of "look closer until confident" on the label-reading results.

Every label pair was rendered at five camera distances. A trajectory starts at the farthest view.
At each view the reader transcribes both labels; if its confidence (the lower of the two mean token
log-probabilities) reaches a threshold it commits to the box whose text is closest to the target,
otherwise the camera moves one step closer. If it is still unsure at the closest view it abstains.

The threshold is chosen on half the trajectories for a target error among committed answers and
scored on the other half, over many random splits. There is no finite-sample correction, so this
shows what the data allow, not a guarantee.

A trajectory here is the same pair of labels in the same font at the five distances. The renderer
re-draws the box angles, the left/right order and which label is the target at every distance, so
successive looks are less alike than they would be on one real approach to a fixed scene.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from summarize import auroc

DISTS = [1.0, 0.7, 0.5, 0.35, 0.22]


def alnum(s):
    return sum(ch.isalnum() for ch in (s or ""))


def load(results, scenes):
    meta = {s["id"]: s for s in json.load(open(os.path.join(scenes, "meta.json")))}
    rows = []
    for line in open(results):
        r = json.loads(line)
        if r["task"] != "crop":
            continue
        s = meta[r["id"]]
        # An empty transcription is "not read", however confident the model was in saying nothing.
        conf = min(lp if alnum(p) > 0 else -np.inf for lp, p in zip(r["mean_lp"], r["pred"]))
        rows.append(dict(model=r["model"], key=f'{s["kind"]}/{s["pair"]}/{s["style"]}', dist=r["dist"],
                         conf=conf, ok=int(r["correct"])))
    return pd.DataFrame(rows)


def to_trajs(g):
    """[n, 5, 2] array of (confidence, correct), ordered far to near; only complete trajectories."""
    out = []
    for _, t in g.groupby("key"):
        t = t.drop_duplicates("dist").set_index("dist")
        if all(d in t.index for d in DISTS):
            out.append([[t.loc[d, "conf"], t.loc[d, "ok"]] for d in DISTS])
    return np.asarray(out, float)


def simulate(trajs, tau):
    conf, ok = trajs[..., 0], trajs[..., 1]
    hit = conf >= tau
    committed = hit.any(1)
    first = np.where(committed, hit.argmax(1), conf.shape[1] - 1)
    return committed, ok[np.arange(len(ok)), first], first


def pick_tau(cal, alpha, min_commit=10):
    """Walk from the strictest threshold down and stop at the first one that misses the target."""
    cands = np.unique(cal[..., 0][np.isfinite(cal[..., 0])])[::-1]
    best = np.inf
    for tau in cands:
        c, okc, _ = simulate(cal, tau)
        if c.sum() < min_commit:
            continue
        if 1 - okc[c].mean() <= alpha:
            best = tau
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/legibility.jsonl")
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--splits", type=int, default=300)
    ap.add_argument("--out", default="results/look_closer.md")
    args = ap.parse_args()
    df = load(args.results, args.scenes)
    out = ["## Look closer until confident: offline simulation on the label-reading data\n"]

    out.append("### Does confidence separate right from wrong at a fixed distance?\n")
    out.append("AUROC of the reader's confidence for choosing the correct box, within one camera distance. "
               "`–` means every choice at that distance was correct (or every one wrong).\n")
    rows = []
    for model, g in df.groupby("model"):
        row = {"model": model}
        for d in DISTS:
            gg = g[g.dist == d]
            a = auroc(gg.conf.replace(-np.inf, -1e9), gg.ok)
            row[f"{d} m"] = "–" if np.isnan(a) else f"{a:.2f} (acc {100 * gg.ok.mean():.0f}%)"
        rows.append(row)
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    out.append("### Adaptive stopping against fixed distances\n")
    out.append("`moves` counts camera steps towards the label (0 = decided from 1.0 m, 4 = went to 0.22 m). "
               "Intervals are the 5th to 95th percentile over random half/half splits of the trajectories.\n")
    rows = []
    rng = np.random.default_rng(0)
    for model, g in df.groupby("model"):
        tr = to_trajs(g)
        n = len(tr)
        fixed_err = [1 - tr[:, k, 1].mean() for k in range(len(DISTS))]
        for alpha in (0.05, 0.10):
            errs, abst, moves, first_view, none = [], [], [], [], 0
            for _ in range(args.splits):
                perm = rng.permutation(n)
                cal, test = tr[perm[: n // 2]], tr[perm[n // 2:]]
                tau = pick_tau(cal, alpha)
                if not np.isfinite(tau):
                    none += 1
                    continue
                c, okc, first = simulate(test, tau)
                errs.append(1 - okc[c].mean() if c.any() else 0.0)
                abst.append(1 - c.mean())
                moves.append(first.mean())
                first_view.append((c & (first == 0)).mean())
            cheapest = next((k for k, e in enumerate(fixed_err) if e <= alpha), None)
            row = {"model": model, "target error": f"{alpha:.0%}", "trajectories": n,
                   "error at 1.0 m, no moves": f"{100 * fixed_err[0]:.0f}%",
                   "error at 0.22 m, 4 moves": f"{100 * fixed_err[-1]:.0f}%",
                   "fixed distance meeting target": "none" if cheapest is None else f"{DISTS[cheapest]} m ({cheapest} moves)"}
            if errs:
                q = lambda v: f"{100 * np.mean(v):.0f}% [{100 * np.percentile(v, 5):.0f}–{100 * np.percentile(v, 95):.0f}]"
                row["adaptive: error among committed"] = q(errs)
                row["adaptive: splits over target"] = f"{100 * np.mean(np.asarray(errs) > alpha):.0f}%"
                row["adaptive: abstains"] = q(abst)
                row["adaptive: moves"] = f"{np.mean(moves):.2f} [{np.percentile(moves, 5):.2f}–{np.percentile(moves, 95):.2f}]"
                row["adaptive: decided at 1.0 m"] = f"{100 * np.mean(first_view):.0f}%"
            row["splits with no threshold"] = f"{100 * none / args.splits:.0f}%"
            rows.append(row)
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    text = "\n".join(out)
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
