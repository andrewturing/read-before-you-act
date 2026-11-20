#!/usr/bin/env python
"""Hypothesis 2 probe: at the image sizes robot policies use, can a model tell two look-alike labels apart?

For each rendered scene:
  - general VLMs answer "which box is labelled X: left or right?" on the whole frame at
    224 px, 512 px (policy-style inputs) and at native 1280x720, and on a digital zoom
    (the native frame cropped to the two labels), which checks that a model failing on the
    whole frame can answer the question at all when the text is large;
  - every model transcribes an oracle crop of each label from the native frame, and the box
    whose transcription is closest to the target string is chosen.
This probes what the visual pipeline can recover. It is not a closed-loop test of a robot policy.
"""
import argparse
import json
import os
import time

from PIL import Image
from rapidfuzz.distance import Levenshtein

from common import MODEL_SPECS, Reader, norm_text, resize_with_pad, upscale_min_side

FC_PROMPT = ('There are two medicine boxes, one on the left and one on the right. '
             'Which box is labelled "{target}"? Answer with one word: left or right.')


def parse_side(text):
    t = text.lower()
    li, ri = t.find("left"), t.find("right")
    if li == -1 and ri == -1:
        return -1
    if ri == -1 or (li != -1 and li < ri):
        return 0
    return 1


def zoom_view(frame, boxes, margin=0.15):
    """Digital zoom: the native frame cropped to the region holding both labels (location given)."""
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    box = (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(frame.width, int(x1 + mx)), min(frame.height, int(y1 + my)))
    return upscale_min_side(frame.crop(box), 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes")
    ap.add_argument("--views", default="224,512,native,zoom")
    ap.add_argument("--models", default="smolvlm2-500m-64tok,smolvlm2-500m,qwen2.5-vl-3b,qwen3.5-2b,glm-ocr,paddleocr-vl-1.6")
    ap.add_argument("--out", default="results/legibility.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = json.load(open(os.path.join(args.scenes, "meta.json")))
    if args.limit:
        step = max(1, len(meta) // args.limit)
        meta = meta[::step][: args.limit]
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["model"], r["id"], r["task"], r.get("view", "")))
    fout = open(args.out, "a")
    for key in args.models.split(","):
        assert key in MODEL_SPECS, key
        try:
            m = Reader(key, device=args.device)
        except Exception as e:  # keep going if one model fails to load
            print(f"[skip] {key}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        t0, n, errs = time.time(), 0, 0
        for s in meta:
            frame = Image.open(os.path.join(args.scenes, s["file"])).convert("RGB")
            target = s["texts"][s["target_side"]]
            if m.kind == "vlm":
                for view in args.views.split(","):
                    if (key, s["id"], "choice", view) in done:
                        continue
                    if view == "native":
                        img = frame
                    elif view == "zoom":
                        img = zoom_view(frame, s["boxes"])
                    else:
                        img = resize_with_pad(frame, int(view))
                    try:
                        text, conf = m.generate(img, FC_PROMPT.format(target=target), max_new_tokens=8)
                    except Exception as e:
                        errs += 1
                        if errs <= 3:
                            print(f"[gen error] {key} scene {s['id']} {view}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                        continue
                    side = parse_side(text)
                    fout.write(json.dumps(dict(model=key, id=s["id"], task="choice", view=view, kind=s["kind"],
                                               style=s["style"], dist=s["dist"], text_px=s["text_px"][s["target_side"]],
                                               pred=text, side=side, correct=int(side == s["target_side"]), **conf),
                                          ensure_ascii=False) + "\n")
                    n += 1
            if (key, s["id"], "crop", "") in done:
                continue
            trans, confs = [], []
            try:
                for b in s["boxes"]:
                    crop = upscale_min_side(frame.crop(tuple(int(round(v)) for v in b)), 64)
                    text, conf = m.generate(crop, m.read_prompt(), max_new_tokens=48)
                    trans.append(text)
                    confs.append(conf)
            except Exception as e:
                errs += 1
                if errs <= 3:
                    print(f"[gen error] {key} scene {s['id']} crop: {type(e).__name__}: {str(e)[:160]}", flush=True)
                continue
            nt = norm_text(target)
            dists = [Levenshtein.normalized_distance(norm_text(t), nt) for t in trans]
            side = -1 if dists[0] == dists[1] else (0 if dists[0] < dists[1] else 1)
            exact = [int(norm_text(t) == norm_text(g)) for t, g in zip(trans, s["texts"])]
            fout.write(json.dumps(dict(model=key, id=s["id"], task="crop", view="", kind=s["kind"], style=s["style"],
                                       dist=s["dist"], text_px=s["text_px"][s["target_side"]], pred=trans,
                                       gold=s["texts"], side=side, correct=int(side == s["target_side"]),
                                       exact=exact, mean_lp=[c["mean_lp"] for c in confs],
                                       min_lp=[c["min_lp"] for c in confs]), ensure_ascii=False) + "\n")
            n += 1
            fout.flush()
        print(f"[done] {key}: {n} queries, {errs} errors, {time.time() - t0:.0f}s", flush=True)
        m.close()
    fout.close()


if __name__ == "__main__":
    main()
