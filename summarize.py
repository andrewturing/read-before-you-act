#!/usr/bin/env python
"""Turn the pilot's JSONL outputs into markdown tables with sample sizes and intervals."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import norm_text, wilson

OK_CER = 0.10  # a line counts as read correctly when its character error rate is at most 10%
ECHO = "the text in the image is"  # start of the stock answer one model gives when it sees no text
SECOND_VI = "uit_hwdb_vi"  # second Vietnamese source, reported in its own table


def is_echo(pred):
    """True when an answer only repeats the instruction instead of offering text it claims to have read."""
    t = norm_text(pred)
    if ECHO not in t:
        return False
    rest = t.replace(ECHO, " ")
    return sum(ch.isalnum() for ch in rest) < 3 or "placeholder" in rest


def looped(pred):
    """Crude detector of a repetition loop: many space-separated tokens, very few distinct ones."""
    t = (pred or "").split()
    return len(t) >= 8 and len(set(t)) <= max(2, len(t) // 6)


def pct(k, n):
    if n == 0:
        return "–"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}] (n={n})"


def auroc(scores, labels):
    """Probability that a positive item has a higher score than a negative one (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def selective(df, alpha, seed=0):
    """Pick the lowest confidence threshold on half the data whose error among answered items is
    <= alpha; report error and coverage on the other half. A plain split, no finite-sample correction."""
    d = df.sample(frac=1.0, random_state=seed)
    cal, test = d.iloc[: len(d) // 2], d.iloc[len(d) // 2:]
    best = None
    for t in np.sort(cal["mean_lp"].unique()):
        kept = cal[cal["mean_lp"] >= t]
        if len(kept) >= 10 and (1 - kept["ok"].mean()) <= alpha:
            best = t
            break
    if best is None:
        return None
    kept = test[test["mean_lp"] >= best]
    if len(kept) == 0:
        return dict(coverage=0.0, err=float("nan"), n=0)
    return dict(coverage=len(kept) / len(test), err=1 - kept["ok"].mean(), n=len(kept))


def legibility(path, out):
    df = pd.DataFrame([json.loads(line) for line in open(path)])
    out.append("## Hypothesis 2 — can the visual pipeline tell look-alike labels apart?\n")
    out.append("Chance is 50%. `px` is the height of the main label text in the native 1280x720 frame; "
               "at a 224 px input it is about 5.7 times smaller, at 512 px about 2.5 times smaller. "
               "`zoom` is the native frame cropped to the two labels (their location is given).\n")
    ch = df[df.task == "choice"]
    if len(ch):
        out.append("### Whole-frame choice (general VLMs), by input size and camera distance\n")
        rows = []
        for (model, view), g in ch.groupby(["model", "view"]):
            row = {"model": model, "input": view}
            for dist, gg in sorted(g.groupby("dist"), reverse=True):
                row[f"{dist} m ({gg.text_px.median():.0f} px)"] = pct(int(gg.correct.sum()), len(gg))
            row["no answer"] = f"{100 * (g.side == -1).mean():.0f}%"
            row["says left"] = f"{100 * (g.side == 0).mean():.0f}%"
            rows.append(row)
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    cr = df[df.task == "crop"]
    if len(cr):
        out.append("### Oracle label crop from the native frame: transcribe, then match to the target\n")
        out.append("`exact` is the share of labels transcribed exactly.\n")
        rows = []
        for model, g in cr.groupby("model"):
            row = {"model": model}
            for dist, gg in sorted(g.groupby("dist"), reverse=True):
                ex = np.concatenate(gg.exact.values)
                row[f"{dist} m ({gg.text_px.median():.0f} px)"] = (
                    f"{pct(int(gg.correct.sum()), len(gg))}; exact {100 * ex.mean():.0f}%")
            rows.append(row)
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
        out.append("### Same, by label type (all distances pooled)\n")
        rows = []
        for (model, kind, style), g in cr.groupby(["model", "kind", "style"]):
            rows.append({"model": model, "label": kind, "font": style, "choice": pct(int(g.correct.sum()), len(g)),
                         "exact": f"{100 * np.concatenate(g.exact.values).mean():.0f}%"})
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
        out.append("### Does confidence flag the unreadable crops?\n")
        out.append("AUROC of mean token log-probability for exact vs. inexact transcriptions of a crop.\n")
        rows = []
        for model, g in cr.groupby("model"):
            lp, ex = np.concatenate(g.mean_lp.values), np.concatenate(g.exact.values)
            rows.append({"model": model, "AUROC": f"{auroc(lp, ex):.2f}", "exact overall": f"{100 * ex.mean():.0f}%",
                         "crops": len(ex)})
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")


def handwriting(path, out):
    df = pd.DataFrame([json.loads(line) for line in open(path)])
    df = df.drop_duplicates(["model", "precision", "dataset", "idx", "cond"])  # workers may overlap after a restart
    df["ok"] = (df["cer"] <= OK_CER).astype(float)
    extra = df[df.dataset == SECOND_VI]  # run at full precision only; kept out of the pooled comparisons
    df = df[df.dataset != SECOND_VI]
    # Pooled comparisons between precisions use only the lines a model read at both precisions.
    both = df.groupby(["model", "dataset", "idx", "cond"]).precision.transform("nunique")
    pooled = df[both == df.groupby("model").precision.transform("nunique")]
    out.append("## Hypothesis 1 — fabrication, confidence and compression on handwriting\n")
    out.append("Tables that pool languages use only the lines a model read at both precisions, so the two "
               "precisions are compared on the same lines.\n")
    read = df[df.cond.isin(["clean", "h24", "h14", "h9"])]
    out.append(f"### Reading accuracy by legibility (a line is correct when its character error rate is at most {OK_CER:.0%})\n")
    out.append("`h24`, `h14`, `h9` are the same lines reduced to a height of 24, 14 and 9 pixels.\n")
    rows = []
    for (model, prec, ds), g in read.groupby(["model", "precision", "dataset"]):
        row = {"model": model, "precision": prec, "data": ds}
        for cond in ("clean", "h24", "h14", "h9"):
            gg = g[g.cond == cond]
            if len(gg):
                row[cond] = f"{100 * gg.ok.mean():.0f}% correct; median CER {gg.cer.median():.2f} (n={len(gg)})"
        rows.append(row)
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    out.append("### Fabrication: output when nothing is legible\n")
    out.append("`blur` and `blank` contain no readable text. `returns text` is an answer with at least 3 letters, "
               "digits or CJK characters. `invented text` leaves out answers that only repeat the instruction "
               "(\"The text in the image is: ...\"). `h9 wrong but fluent` is the share of 9-pixel lines "
               "that were read wrongly yet answered with at least half as many characters as the true text.\n")
    rows = []
    for (model, prec), g in pooled.groupby(["model", "precision"]):
        row = {"model": model, "precision": prec}
        for cond in ("blur", "blank"):
            gg = g[g.cond == cond]
            if len(gg):
                text = gg.out_alnum >= 3
                echo = gg.pred.map(is_echo)
                row[f"{cond}: returns text"] = pct(int(text.sum()), len(gg))
                row[f"{cond}: invented text"] = pct(int((text & ~echo).sum()), len(gg))
        h9 = g[g.cond == "h9"]
        if len(h9):
            row["h9 wrong but fluent"] = pct(int(((h9.ok == 0) & (h9.out_alnum >= 0.5 * h9.gold_alnum)).sum()), len(h9))
        rows.append(row)
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    out.append("### Repetition loops\n")
    out.append("A loop is an answer of at least 8 space-separated tokens with very few distinct ones (a crude "
               "detector that misses loops in unspaced Chinese). `as confident as a correct line` counts loops "
               "whose mean token log-probability is at least the median of that model's correctly read lines.\n")
    rows = []
    for (model, prec), g in pooled.groupby(["model", "precision"]):
        r = g[g.cond.isin(["clean", "h24", "h14", "h9"])]
        loops = r[r.pred.map(looped)]
        med = r[r.ok == 1].mean_lp.median()
        rows.append({"model": model, "precision": prec, "loops": pct(len(loops), len(r)),
                     "as confident as a correct line": f"{int((loops.mean_lp >= med).sum())} of {len(loops)}"})
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    out.append("### Does confidence separate right from wrong?\n")
    out.append("AUROC of mean token log-probability over all readable lines (clean and reduced): 0.5 is useless, "
               "1.0 is perfect. `selective` picks a threshold on half the lines for a 10% or 20% error target and "
               "reports the error and the share of lines still answered on the other half.\n")
    rows = []
    for (model, prec), g in pooled.groupby(["model", "precision"]):
        r = g[g.cond.isin(["clean", "h24", "h14", "h9"])]
        row = {"model": model, "precision": prec, "lines": len(r),
               "accuracy answering everything": f"{100 * r.ok.mean():.0f}%",
               "AUROC right vs wrong": f"{auroc(r.mean_lp, r.ok):.2f}"}
        for a in (0.10, 0.20):
            sel = selective(r, a)
            row[f"selective @{a:.0%}"] = "no threshold found" if sel is None else (
                f"error {100 * sel['err']:.0f}%, answers {100 * sel['coverage']:.0f}% (n={sel['n']})")
        rows.append(row)
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    out.append("### Confidence within one legibility level\n")
    out.append("The pooled AUROC above partly reflects that small lines are both harder and less confident. "
               "This is the AUROC among lines of one language at one size, with the share read correctly.\n")
    rows = []
    for (model, prec, ds), g in df[df.cond.isin(["clean", "h24", "h14", "h9"])].groupby(["model", "precision", "dataset"]):
        row = {"model": model, "precision": prec, "data": ds}
        for cond in ("clean", "h24", "h14", "h9"):
            gg = g[g.cond == cond]
            a = auroc(gg.mean_lp, gg.ok) if len(gg) else float("nan")
            row[cond] = "–" if np.isnan(a) else f"{a:.2f} ({100 * gg.ok.mean():.0f}% correct)"
        rows.append(row)
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    out.append("### Same, per language\n")
    rows = []
    for (model, prec, ds), g in df[df.cond.isin(["clean", "h24", "h14", "h9"])].groupby(["model", "precision", "dataset"]):
        sel = selective(g, 0.10)
        rows.append({"model": model, "precision": prec, "data": ds, "accuracy": f"{100 * g.ok.mean():.0f}%",
                     "AUROC": f"{auroc(g.mean_lp, g.ok):.2f}",
                     "selective @10%": "no threshold found" if sel is None else
                     f"error {100 * sel['err']:.0f}%, answers {100 * sel['coverage']:.0f}% (n={sel['n']})"})
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    if len(extra):
        out.append("### Second Vietnamese source: scanned UIT-HWDB lines, full precision\n")
        out.append("A check that the Vietnamese result is not specific to the pen-stroke renderings of VNOnDB.\n")
        rows = []
        for model, g in extra.groupby("model"):
            row = {"model": model}
            for cond in ("clean", "h24", "h14", "h9"):
                gg = g[g.cond == cond]
                if len(gg):
                    row[cond] = f"{100 * gg.ok.mean():.0f}% correct; median CER {gg.cer.median():.2f} (n={len(gg)})"
            r = g[g.cond.isin(["clean", "h24", "h14", "h9"])]
            row["AUROC right vs wrong"] = f"{auroc(r.mean_lp, r.ok):.2f}"
            sel = selective(r, 0.10)
            row["selective @10%"] = "no threshold found" if sel is None else (
                f"error {100 * sel['err']:.0f}%, answers {100 * sel['coverage']:.0f}% (n={sel['n']})")
            blank = g[g.cond == "blank"]
            if len(blank):
                row["blank: returns text"] = pct(int((blank.out_alnum >= 3).sum()), len(blank))
            rows.append(row)
        out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--legibility", default="legibility.jsonl")
    ap.add_argument("--handwriting", default="handwriting.jsonl")
    args = ap.parse_args()
    out = ["# Pilot results\n"]
    for name, fn in ((args.legibility, legibility), (args.handwriting, handwriting)):
        p = os.path.join(args.dir, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            fn(p, out)
    text = "\n".join(out)
    open(os.path.join(args.dir, "summary.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
