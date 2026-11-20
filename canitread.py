#!/usr/bin/env python
"""Can it read? A reading test for vision-language models.

Two tests on pairs of look-alike labels (Hydroxyzine / Hydralazine, 5 mg / 50 mg):

  crops   the model transcribes each label on its own, at shrinking text sizes; the result is the
          smallest text at which it still tells the pair apart (its acuity);
  frame   the model sees both labels in one 1280x720 camera frame, at camera resolution and
          resized to 512 and 224 pixels as robot policies do, and says which box is the target.

    python canitread.py --model Qwen/Qwen3.5-2B

The labels are drawn with Pillow's built-in font, so the test is the same on every machine and
needs no simulator, data set or font files.
"""
import argparse
import glob
import json
import os
import sys
import time

from PIL import Image, ImageDraw, ImageFont

from labels import DOSE_NAMES, DOSE_PAIRS, DOSES, NAME_PAIRS

CHART_VERSION = "0.1"
SIZES = [24, 16, 12, 10, 8, 6, 5, 4, 3, 2]  # height of a capital letter, in pixels
PASS = 0.90  # a line of the chart is "read" when at least this share of pairs is told apart
MASTER_CAP = 96  # labels are drawn large, then reduced the way a sensor would sample them
PAPER, INK, EDGE = (250, 250, 246), (15, 15, 20), (40, 60, 120)

# Reference camera: 1280x720, 42 degree vertical field of view, label text 6 mm tall.
REF_FOCAL_PX, REF_SENSOR_W, REF_TEXT_M = 938.0, 1280, 0.006
FRAME_W, FRAME_H = 1280, 720
FRAME_TEXT = [24, 12]  # text height on the sensor: 6 mm print at about 23 cm and 47 cm
FRAME_INPUTS = [("camera", None), ("512", 512), ("224", 224)]
FC_PROMPT = ('There are two medicine boxes, one on the left and one on the right. '
             'Which box is labelled "{target}"? Answer with one word: left or right.')


def chart_pairs(n=None):
    """The fixed list of look-alike pairs: 32 drug names, then 8 doses."""
    pairs = []
    for i, (a, b) in enumerate(NAME_PAIRS):
        dose = DOSES[i % len(DOSES)]
        pairs.append(dict(kind="name", lines=([a, dose], [b, dose])))
    for (a, b), name in zip(DOSE_PAIRS, DOSE_NAMES):
        pairs.append(dict(kind="dose", lines=([name, a], [name, b])))
    if n and n < len(pairs):  # keep both kinds in a shortened chart
        step = len(pairs) / n
        pairs = [pairs[int(i * step)] for i in range(n)]
    return pairs


_FONT = None


def master_font():
    """Pillow's built-in scalable font, sized so that a capital letter is MASTER_CAP pixels tall."""
    global _FONT
    if _FONT is None:
        try:
            size = int(MASTER_CAP / 0.7)
            for _ in range(40):
                font = ImageFont.load_default(size=size)
                l, t, r, b = font.getbbox("H")
                if b - t == MASTER_CAP:
                    break
                size += 1 if b - t < MASTER_CAP else -1
            _FONT = font
        except TypeError:
            sys.exit("canitread needs Pillow 10.1 or newer for its built-in scalable font: pip install -U pillow")
    return _FONT


def draw_label(lines, cap_px):
    """A two-line label whose capital letters are cap_px pixels tall."""
    font = master_font()
    w, h = 1320, 470
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, w - 11, h - 11], outline=EDGE, width=10)
    top = font.getbbox("H")[1]
    y = (h - (2 * MASTER_CAP + 80)) // 2
    for text in lines:
        l, _, r, _ = font.getbbox(text)
        d.text(((w - (r - l)) // 2 - l, y - top), text, font=font, fill=INK)
        y += MASTER_CAP + 80
    s = cap_px / MASTER_CAP
    return img.resize((max(8, round(w * s)), max(4, round(h * s))), Image.LANCZOS)


def draw_frame(pair_lines, cap_px):
    """A 1280x720 camera frame: two cartons on a table, each with a label whose capitals are cap_px tall."""
    frame = Image.new("RGB", (FRAME_W, FRAME_H), (214, 218, 222))
    d = ImageDraw.Draw(frame)
    d.rectangle([0, int(FRAME_H * 0.56), FRAME_W, FRAME_H], fill=(178, 153, 122))
    labels = [draw_label(lines, cap_px) for lines in pair_lines]
    w, h = labels[0].size
    gap, m = int(0.35 * w), max(2, int(0.12 * h))
    y = int(FRAME_H * 0.56) - h - m - int(0.25 * h)
    for i, lab in enumerate(labels):
        x = FRAME_W // 2 + (-(gap // 2) - w if i == 0 else gap // 2)
        d.rectangle([x - m + 3, y - m + 4, x + w + m + 3, y + h + m + int(0.25 * h) + 4], fill=(150, 128, 100))
        d.rectangle([x - m, y - m, x + w + m, y + h + m + int(0.25 * h)], fill=(238, 238, 232), outline=(150, 150, 146))
        frame.paste(lab, (x, y))
    return frame


def parse_side(text):
    t = text.lower()
    li, ri = t.find("left"), t.find("right")
    if li == -1 and ri == -1:
        return -1
    return 0 if ri == -1 or (li != -1 and li < ri) else 1


def frame_test(reader, pairs, wilson, resize_with_pad):
    """Forced choice on the whole frame, as a robot policy would receive it."""
    out, rows = [], []
    for ti, cap in enumerate(FRAME_TEXT):
        for name, width in FRAME_INPUTS:
            k = left = 0
            for pi, pair in enumerate(pairs):
                target_side = (pi + ti) % 2
                target = " ".join(pair["lines"][target_side])
                frame = draw_frame(pair["lines"], cap)
                view = frame if width is None else resize_with_pad(frame, width)
                text, _ = reader.generate(view, FC_PROMPT.format(target=target), max_new_tokens=8)
                side = parse_side(text)
                k += int(side == target_side)
                left += int(side == 0)
                rows.append(dict(test="frame", sensor_px=cap, input=name, pair=pi, target=target, pred=text,
                                 correct=int(side == target_side)))
            lo, hi = wilson(k, len(pairs))
            out.append(dict(sensor_px=cap, input=name, n=len(pairs), correct=k, accuracy=round(k / len(pairs), 4),
                            low=round(lo, 4), high=round(hi, 4), says_left=round(left / len(pairs), 4)))
            print(f"  frame, text {cap:>2} px, {name:>6} input  {k}/{len(pairs)}", flush=True)
    return out, rows


def run(args):
    from rapidfuzz.distance import Levenshtein

    from common import Reader, norm_text, resize_with_pad, upscale_min_side, wilson

    sizes = [int(v) for v in args.sizes.split(",")] if args.sizes else SIZES
    pairs = chart_pairs(12 if args.quick else args.pairs)
    print(f"loading {args.model} ...", flush=True)
    reader = Reader(args.model, precision=args.precision, device=args.device, kind=args.kind, prompt=args.prompt)
    t0, rows = time.time(), []
    for si, size in enumerate(sizes):
        ok = 0
        for pi, pair in enumerate(pairs):
            target_side = (pi + si) % 2
            target = " ".join(pair["lines"][target_side])
            preds, lps = [], []
            for lines in pair["lines"]:
                text, conf = reader.generate(upscale_min_side(draw_label(lines, size), 64), reader.read_prompt(),
                                             max_new_tokens=32)
                preds.append(text)
                lps.append(conf["mean_lp"])
            dist = [Levenshtein.normalized_distance(norm_text(p), norm_text(target)) for p in preds]
            side = -1 if dist[0] == dist[1] else (0 if dist[0] < dist[1] else 1)  # a tie is a miss
            correct = int(side == target_side)
            ok += correct
            rows.append(dict(size=size, pair=pi, kind=pair["kind"], target=target, pred=preds, correct=correct,
                             exact=int(norm_text(preds[target_side]) == norm_text(target)), mean_lp=lps))
        print(f"  {size:>2} px  {ok}/{len(pairs)}", flush=True)
    frame, frame_rows = (frame_test(reader, pairs, wilson, resize_with_pad) if reader.kind == "vlm" else ([], []))
    seconds = time.time() - t0

    lines_out, by_size, acuity, still_reading = [], [], None, True
    for size in sizes:
        g = [r for r in rows if r["size"] == size]
        k, n = sum(r["correct"] for r in g), len(g)
        lo, hi = wilson(k, n)
        by_size.append(dict(px=size, n=n, correct=k, accuracy=round(k / n, 4), low=round(lo, 4), high=round(hi, 4),
                            exact=round(sum(r["exact"] for r in g) / n, 4)))
        if still_reading and k / n >= PASS:
            acuity = size
        else:
            still_reading = False
        lines_out.append(f"  {size:>2} px  {'█' * round(24 * k / n):<24} {100 * k / n:3.0f}%   "
                         f"[{100 * lo:.0f}–{100 * hi:.0f}]")
    print(f"\n{args.model}" + (" (4-bit)" if args.precision == "4bit" else ""))
    print("\n  LABEL CROPS: each label on its own")
    print("  text    told apart" + " " * 19 + "95% interval")
    print("\n".join(lines_out))
    if acuity is None:
        print(f"\n  acuity: none. It never told {PASS:.0%} of look-alike pairs apart, even at {sizes[0]} px.")
    else:
        reach = REF_FOCAL_PX * REF_TEXT_M / acuity
        print(f"\n  acuity: {acuity} px. That is the smallest text at which it tells look-alike labels apart.")
        print(f"  Handed the label crop from a 1280x720 camera, it could read 6 mm print from about {100 * reach:.0f} cm at best.")
    if frame:
        print("\n  WHOLE FRAME: both labels in one camera frame, as a robot policy receives it (chance is 50%)")
        for cap in FRAME_TEXT:
            dist = REF_FOCAL_PX * REF_TEXT_M / cap
            print(f"  6 mm print at {100 * dist:.0f} cm, {cap} px tall on the sensor")
            for f in [f for f in frame if f["sensor_px"] == cap]:
                name = "camera resolution" if f["input"] == "camera" else f"{f['input']} px input"
                print(f"    {name:<18} {'█' * round(24 * f['accuracy']):<24} {100 * f['accuracy']:3.0f}%   "
                      f"[{100 * f['low']:.0f}–{100 * f['high']:.0f}]")
    elif reader.kind != "vlm":
        print("\n  WHOLE FRAME: skipped, because this model only transcribes.")
    print(f"\n  {len(rows) * 2 + len(frame_rows)} readings in {seconds:.0f} s on {reader.device}; chart {CHART_VERSION}, "
          f"{len(pairs)} pairs" + (" (quick chart: use the full one for the leaderboard)" if args.quick else ""))

    os.makedirs(args.out, exist_ok=True)
    name = args.model.replace("/", "__") + ("-4bit" if args.precision == "4bit" else "")
    result = dict(model=args.model, precision="4bit" if args.precision == "4bit" else str(reader.dtype).replace("torch.", ""),
                  device=reader.device.split(":")[0], chart=CHART_VERSION, pairs=len(pairs), quick=bool(args.quick),
                  acuity_px=acuity, seconds=round(seconds), sizes=by_size, frame=frame)
    path = os.path.join(args.out, name + ".json")
    json.dump(result, open(path, "w"), indent=1)
    if args.save_answers:
        json.dump(rows + frame_rows, open(os.path.join(args.out, name + ".answers.json"), "w"), ensure_ascii=False, indent=1)
    print(f"  saved {path}")


def add_frame(args):
    """Run only the whole-frame test and add it to a result that was saved without one."""
    from common import Reader, resize_with_pad, wilson

    name = args.model.replace("/", "__") + ("-4bit" if args.precision == "4bit" else "")
    path = os.path.join(args.out, name + ".json")
    result = json.load(open(path))
    reader = Reader(args.model, precision=args.precision, device=args.device, kind=args.kind, prompt=args.prompt)
    if reader.kind != "vlm":
        print("whole-frame test skipped: this model only transcribes")
        return
    frame, frame_rows = frame_test(reader, chart_pairs(result["pairs"]), wilson, resize_with_pad)
    result["frame"] = frame
    json.dump(result, open(path, "w"), indent=1)
    answers = os.path.join(args.out, name + ".answers.json")
    if os.path.exists(answers):
        rows = [r for r in json.load(open(answers)) if r.get("test") != "frame"]
        json.dump(rows + frame_rows, open(answers, "w"), ensure_ascii=False, indent=1)
    print(f"  added the whole-frame test to {path}")


def leaderboard(args):
    """Markdown table from every full-chart result in the results folder."""
    res = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.out, "*.json"))) if not p.endswith(".answers.json")]
    res = [r for r in res if not r.get("quick")]
    px = [6, 4, 3, 2]

    def acc(r, size):
        return next((s["accuracy"] for s in r["sizes"] if s["px"] == size), None)

    def frame(r, name, cap=12):
        return next((f["accuracy"] for f in r.get("frame", []) if f["input"] == name and f["sensor_px"] == cap), None)

    def pct(v):
        return "–" if v is None else f"{100 * v:.0f}%"

    res.sort(key=lambda r: (r["acuity_px"] if r["acuity_px"] is not None else 999,
                            -sum(acc(r, s) or 0 for s in px)))
    print("| Model | Precision | Acuity on crops | " + " | ".join(f"{s} px" for s in px)
          + " | Frame: camera | Frame: 512 px | Frame: 224 px |")
    print("|---|---|---|" + "---|" * (len(px) + 3))
    for r in res:
        acuity = "none" if r["acuity_px"] is None else f"**{r['acuity_px']} px**"
        cells = " | ".join(pct(acc(r, s)) for s in px)
        print(f"| `{r['model']}` | {r['precision']} | {acuity} | {cells} | "
              f"{pct(frame(r, 'camera'))} | {pct(frame(r, '512'))} | {pct(frame(r, '224'))} |")


def save_chart(args):
    """Write the chart for one pair as an image, one row per text size, magnified so pixels are visible."""
    pair = chart_pairs()[0]
    sizes = [int(v) for v in args.sizes.split(",")] if args.sizes else SIZES
    tile_w, tile_h, gap = 660, 235, 30
    sheet = Image.new("RGB", (2 * tile_w + 3 * gap, len(sizes) * (tile_h + gap) + gap), (255, 255, 255))
    for i, size in enumerate(sizes):
        for j, lines in enumerate(pair["lines"]):
            tile = draw_label(lines, size).resize((tile_w, tile_h), Image.NEAREST)
            sheet.paste(tile, (gap + j * (tile_w + gap), gap + i * (tile_h + gap)))
    os.makedirs(os.path.dirname(os.path.abspath(args.save_chart)), exist_ok=True)
    sheet.save(args.save_chart)
    print("wrote", args.save_chart)


def main():
    ap = argparse.ArgumentParser(description="Can it read? A reading test for vision-language models.")
    ap.add_argument("--model", help="Hugging Face repository id, or a short name from common.MODEL_SPECS")
    ap.add_argument("--kind", choices=["vlm", "ocr"], help="ocr: the model takes a fixed task prompt (set it with --prompt)")
    ap.add_argument("--prompt", help="override the transcription prompt")
    ap.add_argument("--precision", choices=["auto", "4bit"], default="auto")
    ap.add_argument("--device", help="cuda:0, mps or cpu (default: the best available)")
    ap.add_argument("--pairs", type=int, default=0, help="use only N of the 40 look-alike pairs")
    ap.add_argument("--quick", action="store_true", help="12 pairs: a rough answer in about a third of the time")
    ap.add_argument("--sizes", help="comma-separated text heights in pixels, largest first")
    ap.add_argument("--out", default="results/canitread")
    ap.add_argument("--save-answers", action="store_true", help="also save every transcription")
    ap.add_argument("--frame-only", action="store_true", help="add the whole-frame test to an existing result")
    ap.add_argument("--leaderboard", action="store_true", help="print a markdown table of the saved results")
    ap.add_argument("--save-chart", metavar="PNG", help="write the chart for one pair as an image and exit")
    ap.add_argument("--save-frame", metavar="PNG", help="write one example camera frame as an image and exit")
    args = ap.parse_args()
    if args.leaderboard:
        leaderboard(args)
    elif args.save_chart:
        save_chart(args)
    elif args.save_frame:
        draw_frame(chart_pairs()[0]["lines"], FRAME_TEXT[-1]).save(args.save_frame)
        print("wrote", args.save_frame)
    elif args.model:
        args.precision = "4bit" if args.precision == "4bit" else "bf16"
        add_frame(args) if args.frame_only else run(args)
    else:
        ap.error("give --model, --leaderboard, --save-chart or --save-frame")


if __name__ == "__main__":
    main()
