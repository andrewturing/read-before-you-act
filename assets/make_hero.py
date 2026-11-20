#!/usr/bin/env python
"""Build the README hero image (light and dark) and the social preview from the saved results.

Each row shows the same pair of look-alike labels at the text size a model receives when a robot
looks at 6 mm print from about half a metre: on the camera sensor, through a 512 px input and
through a 224 px input. The dots count how many tested models still pick the right box in the
whole-frame test.

Usage: python assets/make_hero.py      (needs Pillow 10.1 or newer; run from the repository root)
"""
import glob
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from canitread import PASS, chart_pairs, draw_label  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "canitread")
W, H = 2560, 1280
SENSOR_PX = 12  # 6 mm print at about 47 cm on the reference camera
ROWS = [("camera", 1.0, "What the camera sees", "1280 x 720 sensor"),
        ("512", 512 / 1280, "Through a 512 px input", "what a compact robot policy receives"),
        ("224", 224 / 1280, "Through a 224 px input", "the most common policy input size")]
THEMES = {
    "light": dict(bg=(255, 255, 255), ink=(11, 11, 11), muted=(82, 81, 78), rule=(217, 216, 210),
                  on=(42, 120, 214), off=(176, 175, 168)),
    "dark": dict(bg=(13, 17, 23), ink=(240, 239, 236), muted=(160, 164, 170), rule=(48, 54, 61),
                 on=(57, 135, 229), off=(92, 97, 104)),
}


def font(size):
    return ImageFont.load_default(size=size)


def load_results():
    res = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RESULTS, "*.json")))
           if not p.endswith(".answers.json")]
    return [r for r in res if not r.get("quick") and r.get("frame")]  # models that took the whole-frame test


def readers_at(results, name):
    """How many models pick the right box for at least PASS of the pairs with this input."""
    n = 0
    for r in results:
        acc = next((f["accuracy"] for f in r["frame"] if f["input"] == name and f["sensor_px"] == SENSOR_PX), None)
        n += int(acc is not None and acc >= PASS)
    return n


def build(theme, results):
    c = THEMES[theme]
    img = Image.new("RGB", (W, H), c["bg"])
    d = ImageDraw.Draw(img)
    d.text((110, 78), "Can it read?", font=font(132), fill=c["ink"])
    d.text((114, 248), "Two medicine labels a robot must tell apart, at the text size the model actually receives.",
           font=font(47), fill=c["muted"])
    pair = chart_pairs()[0]  # Hydroxyzine / Hydralazine
    tile_w, tile_h, top, row_h = 560, 200, 372, 268
    x_tiles, x_dots = 860, 860 + 2 * tile_w + 40 + 80
    total = len(results)
    for i, (name, scale, title, sub) in enumerate(ROWS):
        px = SENSOR_PX * scale
        y = top + i * row_h
        d.line([(110, y - 34), (W - 110, y - 34)], fill=c["rule"], width=3)
        d.text((110, y + 8), title, font=font(60), fill=c["ink"])
        d.text((112, y + 92), sub, font=font(38), fill=c["muted"])
        d.text((112, y + 146), f"text {px:.0f} px tall", font=font(38), fill=c["muted"])
        for j, lines in enumerate(pair["lines"]):
            tile = draw_label(lines, px).resize((tile_w, tile_h), Image.NEAREST)
            img.paste(tile, (x_tiles + j * (tile_w + 40), y))
        k = readers_at(results, name)
        d.text((x_dots, y + 2), f"{k} of {total}", font=font(84), fill=c["ink"])
        d.text((x_dots + 4, y + 106), "models tell them apart", font=font(34), fill=c["muted"])
        for m in range(total):  # one dot per model: filled when it reads this line, hollow when not
            cx, cy, r = x_dots + 14 + m * 30, y + 178, 11
            if m < k:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c["on"])
            else:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c["off"], width=3)
    d.line([(110, top + 3 * row_h - 34), (W - 110, top + 3 * row_h - 34)], fill=c["rule"], width=3)
    d.text((110, H - 116), "6 mm print seen from about half a metre, both labels in one camera frame. Labels are magnified "
           "here; the model gets the whole frame.", font=font(34), fill=c["muted"])
    d.text((110, H - 68), f"A model passes a row when it picks the right box for at least {PASS:.0%} of 40 look-alike pairs.",
           font=font(34), fill=c["muted"])
    return img


def main():
    results = load_results()
    if not results:
        sys.exit("no results in results/canitread yet: run canitread.py on at least one model first")
    for theme in THEMES:
        img = build(theme, results)
        img.save(os.path.join(HERE, f"hero-{theme}.png"), optimize=True)
        if theme == "light":
            img.resize((1280, 640), Image.LANCZOS).save(os.path.join(HERE, "social-preview.png"), optimize=True)
    print(f"wrote hero-light.png, hero-dark.png and social-preview.png from {len(results)} models")


if __name__ == "__main__":
    main()
