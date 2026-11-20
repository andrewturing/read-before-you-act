#!/usr/bin/env python
"""Hypothesis 1 probe: do small readers fabricate on illegible handwriting, does their confidence
track legibility, and does 4-bit compression change either?

For each handwriting line (English, Vietnamese, Chinese) the model transcribes:
  clean          the line as released
  h24 / h14 / h9 the line downscaled so its height is 24 / 14 / 9 px (then upscaled back): a legibility ladder
  blur           heavy Gaussian blur: nothing legible remains
  blank          paper-coloured noise the same size as the line: there is no text at all
A reader that knows its limits should return little or nothing on blur and blank.
"""
import argparse
import io
import json
import os
import random
import time

import numpy as np
from PIL import Image, ImageFilter

from common import MODEL_SPECS, Reader, cer, count_alnum, norm_text

# Each entry lists candidate repositories, tried in order. The Vietnamese sets are third-party
# mirrors of UIT-HWDB / VNOnDB (the maintained 5CD-AI sets are gated); fine for a pilot, to be
# replaced by the original releases for anything published.
DATASETS = {
    "iam_en": [dict(repo="Teklia/IAM-line", split="test")],
    "uit_vi": [dict(repo="HungNguyen29/UIT-HWDB-line", split=None),
               dict(repo="blue7012/UIT_HWDB", split=None),
               dict(repo="vklinhhh/processed-vnondb-line", split=None)],
    "casia_zh": [dict(repo="Teklia/CASIA-HWDB2-line", split="test")],
    # A second Vietnamese source (scanned UIT-HWDB lines), to check that the Vietnamese result
    # is not specific to the pen-stroke renderings of VNOnDB.
    "uit_hwdb_vi": [dict(repo="HungNguyen29/UIT-HWDB-line", split=None),
                    dict(repo="blue7012/UIT_HWDB", split=None)],
}
TEXT_KEYS = ("text", "label", "transcription", "ground_truth", "answer", "caption", "sentence")


def _as_image(v):
    """Some repositories stream images as raw bytes instead of decoded images."""
    if isinstance(v, Image.Image):
        return v
    if isinstance(v, dict) and v.get("bytes"):
        try:
            return Image.open(io.BytesIO(v["bytes"]))
        except Exception:
            return None
    return None


def load_lines(name, n, seed=0):
    from datasets import get_dataset_split_names, load_dataset

    last = None
    for cfg in DATASETS[name]:
        try:
            split = cfg["split"]
            if split is None:
                splits = get_dataset_split_names(cfg["repo"])
                split = "test" if "test" in splits else ("validation" if "validation" in splits else splits[0])
            ds = load_dataset(cfg["repo"], split=split, streaming=True).shuffle(seed=seed, buffer_size=2000)
            out = []
            for ex in ds:
                img = next((im for im in map(_as_image, ex.values()) if im is not None), None)
                txt = next((ex[k] for k in TEXT_KEYS if k in ex and isinstance(ex[k], str)), None)
                if txt is None:
                    txt = next((v for v in ex.values() if isinstance(v, str) and len(v.strip()) >= 3
                                and not v.lower().endswith((".png", ".jpg", ".jpeg"))), None)
                if img is None or not txt or len(txt.strip()) < 3 or img.size[1] < 12:
                    continue
                out.append((img.convert("RGB"), txt.strip()))
                if len(out) >= n:
                    break
            if out:
                return out, f"{cfg['repo']}:{split}"
        except Exception as e:
            last = e
    raise RuntimeError(f"no usable source for {name}: {last}")


def to_height(img, h):
    w0, h0 = img.size
    small = img.resize((max(1, round(w0 * h / h0)), h), Image.BICUBIC)
    return small.resize((w0, h0), Image.BICUBIC)


def variants(img, rng):
    w, h = img.size
    noise = np.clip(np.random.default_rng(rng.randrange(1 << 30)).normal(236, 6, (h, w, 3)), 0, 255).astype(np.uint8)
    return {
        "clean": img,
        "h24": to_height(img, 24),
        "h14": to_height(img, 14),
        "h9": to_height(img, 9),
        "blur": img.filter(ImageFilter.GaussianBlur(radius=max(6, h * 0.35))),
        "blank": Image.fromarray(noise),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="glm-ocr,paddleocr-vl-1.6,qwen3.5-2b,qwen2.5-vl-3b")
    ap.add_argument("--precisions", default="bf16,4bit")
    ap.add_argument("--datasets", default="iam_en,uit_vi,casia_zh")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="results/handwriting.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--idx-mod", default="0,1", help="r,m: only lines whose index leaves remainder r modulo m, "
                                                     "to split one model across workers")
    args = ap.parse_args()
    mod_r, mod_m = (int(v) for v in args.idx_mod.split(","))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = {}
    for name in args.datasets.split(","):
        try:
            lines, src = load_lines(name, args.n)
            data[name] = lines
            hs = sorted(im.size[1] for im, _ in lines)
            print(f"[data] {name}: {len(lines)} lines from {src}; median height {hs[len(hs) // 2]} px", flush=True)
        except Exception as e:
            print(f"[skip data] {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["model"], r["precision"], r["dataset"], r["idx"], r["cond"]))
    fout = open(args.out, "a")
    for key in args.models.split(","):
        assert key in MODEL_SPECS, key
        for prec in args.precisions.split(","):
            try:
                m = Reader(key, precision=prec, device=args.device)
            except Exception as e:
                print(f"[skip] {key}/{prec}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            t0, n, errs = time.time(), 0, 0
            for name, lines in data.items():
                rng = random.Random(7)
                for idx, (img, gold) in enumerate(lines):
                    for cond, v in variants(img, rng).items():
                        if idx % mod_m != mod_r or (key, prec, name, idx, cond) in done:
                            continue
                        try:
                            text, conf = m.generate(v, m.read_prompt(), max_new_tokens=96)
                        except Exception as e:
                            errs += 1
                            if errs <= 3:
                                print(f"[gen error] {key}/{prec} {name} {idx} {cond}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                            continue
                        has_text = cond not in ("blur", "blank")
                        fout.write(json.dumps(dict(
                            model=key, precision=prec, dataset=name, idx=idx, cond=cond, gold=gold, pred=text,
                            cer=cer(text, gold) if has_text else None,
                            exact=int(norm_text(text) == norm_text(gold)) if has_text else None,
                            out_alnum=count_alnum(text), gold_alnum=count_alnum(gold), **conf),
                            ensure_ascii=False) + "\n")
                        n += 1
                    fout.flush()
            print(f"[done] {key}/{prec}: {n} queries, {errs} errors, {time.time() - t0:.0f}s", flush=True)
            m.close()
    fout.close()


if __name__ == "__main__":
    main()
