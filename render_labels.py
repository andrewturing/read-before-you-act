#!/usr/bin/env python
"""Render pick-by-label scenes in MuJoCo: two look-alike medicine boxes facing a camera.

Each scene has a left and a right box whose labels differ by a confusable drug name,
a dose digit, or a short Vietnamese / Chinese instruction. The camera distance is swept
so that the text height in pixels is known. Output: one 1280x720 frame per scene plus
metadata (texts, label boxes in the image, text height in pixels).
"""
import argparse
import json
import math
import os
import random

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from labels import DOSE_NAMES, DOSE_PAIRS, DOSES, NAME_PAIRS, VI_PAIRS, ZH_PAIRS  # noqa: E402

W, H, FOVY = 1280, 720, 42.0
F_PX = (H / 2) / math.tan(math.radians(FOVY / 2))
BOX_HALF = (0.06, 0.015, 0.0375)  # half width, depth, height (m): a 12 x 3 x 7.5 cm carton
LABEL_HALF = (0.055, 0.0325)  # half width, height (m)
PX_PER_MM = 10
BOX_X = 0.068
ELEV_DEG = 12.0
DISTANCES = [1.0, 0.7, 0.5, 0.35, 0.22]

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONTS = {
    ("latin", "print"): "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ("latin", "hand"): os.path.join(FONT_DIR, "PatrickHand-Regular.ttf"),
    ("zh", "print"): "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ("zh", "hand"): os.path.join(FONT_DIR, "MaShanZheng-Regular.ttf"),
}


def fit_font(path, text, target_h_px, max_w_px):
    """Largest font whose ink height is <= target and whose width fits; returns font and ink bbox."""
    size = max(8, int(target_h_px / 0.6))
    while size > 8:
        font = ImageFont.truetype(path, size)
        l, t, r, b = font.getbbox(text)
        if (b - t) <= target_h_px and (r - l) <= max_w_px:
            return font, (l, t, r, b)
        size -= 2
    font = ImageFont.truetype(path, 8)
    return font, font.getbbox(text)


def make_label(lines, script, style):
    """lines: list of (text, target ink height in mm). Returns image and measured ink height (mm) of line 0."""
    wpx, hpx = int(LABEL_HALF[0] * 2000 * PX_PER_MM), int(LABEL_HALF[1] * 2000 * PX_PER_MM)
    img = Image.new("RGB", (wpx, hpx), (250, 250, 246))
    d = ImageDraw.Draw(img)
    d.rectangle([6, 6, wpx - 7, hpx - 7], outline=(40, 60, 120), width=6)
    path = FONTS[(script, style)]
    total_h = sum(mm * PX_PER_MM for _, mm in lines) + 60 * (len(lines) - 1)
    y = (hpx - total_h) // 2
    ink0 = None
    for i, (text, mm) in enumerate(lines):
        font, (l, t, r, b) = fit_font(path, text, mm * PX_PER_MM, wpx - 80)
        x = (wpx - (r - l)) // 2 - l
        d.text((x, y - t), text, font=font, fill=(15, 15, 20))
        if i == 0:
            ink0 = (b - t) / PX_PER_MM
        y += int(mm * PX_PER_MM) + 60
    return img, ink0


def scene_xml(tex_paths, yaws, dist, light, table_rgb):
    a = math.radians(ELEV_DEG)
    tgt = np.array([0.0, -BOX_HALF[1], BOX_HALF[2]])
    cam = tgt + dist * np.array([0.0, -math.cos(a), math.sin(a)])
    geoms, assets, corners = [], [], []
    for i, (tex, yaw) in enumerate(zip(tex_paths, yaws)):
        c, s = math.cos(yaw), math.sin(yaw)
        bx = np.array([(-1 if i == 0 else 1) * BOX_X, 0.0, BOX_HALF[2]])
        normal = np.array([s, -c, 0.0])
        lc = bx + normal * (BOX_HALF[1] + 0.001)
        ex, ez = np.array([c, s, 0.0]), np.array([0.0, 0.0, 1.0])
        assets.append(f'<texture name="t{i}" type="2d" file="{tex}"/>'
                      f'<material name="m{i}" texture="t{i}" texrepeat="1 1" texuniform="false" specular="0" shininess="0"/>')
        geoms.append(f'<geom type="box" size="{BOX_HALF[0]} {BOX_HALF[1]} {BOX_HALF[2]}" pos="{bx[0]} {bx[1]} {bx[2]}" '
                     f'xyaxes="{c} {s} 0 {-s} {c} 0" rgba="0.93 0.93 0.9 1"/>')
        geoms.append(f'<geom type="plane" size="{LABEL_HALF[0]} {LABEL_HALF[1]} 0.01" pos="{lc[0]} {lc[1]} {lc[2]}" '
                     f'xyaxes="{c} {s} 0 0 0 1" material="m{i}"/>')
        corners.append([lc + sx * LABEL_HALF[0] * ex + sz * LABEL_HALF[1] * ez for sx in (-1, 1) for sz in (-1, 1)])
    xml = f"""
<mujoco>
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <quality offsamples="8" shadowsize="2048"/>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.45 0.45 0.45" specular="0 0 0"/>
  </visual>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <light pos="{light[0]} {light[1]} {light[2]}" dir="{-light[0]} {-light[1]} {-light[2]}" diffuse="0.55 0.55 0.55" castshadow="true"/>
    <geom type="plane" size="2 2 0.01" rgba="{table_rgb[0]} {table_rgb[1]} {table_rgb[2]} 1"/>
    <geom type="plane" pos="0 0.5 0" xyaxes="1 0 0 0 0 1" size="2 1.5 0.01" rgba="0.82 0.84 0.86 1"/>
    {''.join(geoms)}
    <camera name="cam" pos="{cam[0]} {cam[1]} {cam[2]}" xyaxes="1 0 0 0 {math.sin(a)} {math.cos(a)}" fovy="{FOVY}"/>
  </worldbody>
</mujoco>"""
    return xml, cam, corners


def project(p, cam):
    a = math.radians(ELEV_DEG)
    xc, yc, zc = np.array([1.0, 0, 0]), np.array([0, math.sin(a), math.cos(a)]), np.array([0, -math.cos(a), math.sin(a)])
    v = np.asarray(p) - cam
    depth = -float(v @ zc)
    return (W / 2 + F_PX * float(v @ xc) / depth, H / 2 - F_PX * float(v @ yc) / depth, depth)


def build_items():
    rng = random.Random(0)
    items = []
    for a, b in NAME_PAIRS:
        dose = rng.choice(DOSES)
        items.append(dict(kind="name", script="latin", texts=(f"{a} {dose}", f"{b} {dose}"),
                          lines=([(a, 6.0), (dose, 4.5)], [(b, 6.0), (dose, 4.5)])))
    for (a, b), name in zip(DOSE_PAIRS, DOSE_NAMES):
        items.append(dict(kind="dose", script="latin", texts=(f"{name} {a}", f"{name} {b}"),
                          lines=([(name, 6.0), (a, 4.5)], [(name, 6.0), (b, 4.5)])))
    for a, b in VI_PAIRS:
        items.append(dict(kind="vi", script="latin", texts=(a, b), lines=([(a, 6.0)], [(b, 6.0)])))
    for a, b in ZH_PAIRS:
        items.append(dict(kind="zh", script="zh", texts=(a, b), lines=([(a, 7.0)], [(b, 7.0)])))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N scenes (smoke test)")
    ap.add_argument("--flip", default="", help="texture fix: any of 'v','h' if labels render mirrored")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tex_dir = os.path.abspath(os.path.join(args.out, "_tex"))
    os.makedirs(tex_dir, exist_ok=True)
    rng = random.Random(1)
    meta, n = [], 0
    for it_idx, it in enumerate(build_items()):
        for style in ("print", "hand"):
            imgs = [make_label(lines, it["script"], style) for lines in it["lines"]]
            for dist in DISTANCES:
                if args.limit and n >= args.limit:
                    break
                order = [0, 1] if rng.random() < 0.5 else [1, 0]  # which text goes on the left box
                tex_paths, inks = [], []
                for slot, k in enumerate(order):
                    im, ink = imgs[k]
                    if "v" in args.flip:
                        im = im.transpose(Image.FLIP_TOP_BOTTOM)
                    if "h" in args.flip:
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    p = os.path.join(tex_dir, f"s{n:04d}_{slot}.png")
                    im.save(p)
                    tex_paths.append(p)
                    inks.append(ink)
                yaws = [math.radians(rng.uniform(-18, 18)) for _ in range(2)]
                light = (rng.uniform(-0.8, 0.8), rng.uniform(-1.2, -0.4), rng.uniform(1.0, 1.8))
                shade = rng.uniform(-0.06, 0.06)
                xml, cam, corners = scene_xml(tex_paths, yaws, dist, light, (0.70 + shade, 0.60 + shade, 0.48 + shade))
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                r = mujoco.Renderer(model, height=H, width=W)
                r.update_scene(data, camera="cam")
                frame = Image.fromarray(r.render())
                r.close()
                fname = f"s{n:04d}.png"
                frame.save(os.path.join(args.out, fname))
                boxes = []
                for cs in corners:
                    uv = [project(c, cam) for c in cs]
                    us, vs = [u for u, _, _ in uv], [v for _, v, _ in uv]
                    mx, my = 0.06 * (max(us) - min(us)), 0.06 * (max(vs) - min(vs))
                    boxes.append([max(0, min(us) - mx), max(0, min(vs) - my), min(W, max(us) + mx), min(H, max(vs) + my)])
                tgt_side = rng.randrange(2)
                meta.append(dict(
                    id=n, file=fname, pair=it_idx, kind=it["kind"], script=it["script"], style=style, dist=dist,
                    texts=[it["texts"][order[0]], it["texts"][order[1]]],
                    target_side=tgt_side, boxes=boxes,
                    text_px=[round(i / 1000 * F_PX / dist, 2) for i in inks],
                    yaw_deg=[round(math.degrees(y), 1) for y in yaws],
                ))
                n += 1
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"rendered {n} scenes to {args.out}")


if __name__ == "__main__":
    main()
