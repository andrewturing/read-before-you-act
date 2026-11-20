"""Reach-by-label: two labelled cartons on a table and a kinematic end-effector, in MuJoCo.

The task is to move the end-effector onto the carton named in the instruction. The cartons are
identical except for their labels, so reading is the only cue. The end-effector is kinematic (it
goes where the action says), which keeps the question on the decision, not on grasping.
"""
import math
import os
import random
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from canitread import MASTER_CAP, chart_pairs, draw_label  # noqa: E402

W, H, FOVY = 1280, 720, 42.0
F_PX = (H / 2) / math.tan(math.radians(FOVY / 2))
OBS_W, OBS_H = 640, 360  # stored frame size; the policy resizes it to 512 with padding
ELEV = math.radians(12.0)
BOX_HALF = (0.06, 0.015, 0.0375)  # a 12 x 3 x 7.5 cm carton
BOX_X = 0.085
CAP_M = 0.006  # capital letters on the label are 6 mm tall
LABEL_HALF = (0.5 * 1320 / MASTER_CAP * CAP_M, 0.5 * 470 / MASTER_CAP * CAP_M)
STEPS_MOVE, STEPS_HOLD = 30, 6
EPISODE_STEPS = STEPS_MOVE + STEPS_HOLD
FPS = 10
REACH_TOL = 0.025
CAM_DIST = {"far": 0.60, "near": 0.25}
POINTER_TASK = "reach the box marked in green"

PAIRS = chart_pairs()
TEST_PAIRS = [i for i in range(len(PAIRS)) if i % 4 == 3]  # ten pairs never shown in training
TRAIN_PAIRS = [i for i in range(len(PAIRS)) if i % 4 != 3]
# "distinct" pairings put the first label of pair p beside the first label of an unrelated pair,
# so the two names differ in most letters (a control: can the policy use text at all?).
DISTINCT_OFFSET = 16


def pair_lines(pair, pairing="lookalike"):
    """The two label texts (each a list of lines) for a pair under a pairing."""
    if pairing == "distinct":
        return [PAIRS[pair]["lines"][0], PAIRS[(pair + DISTINCT_OFFSET) % len(PAIRS)]["lines"][0]]
    return PAIRS[pair]["lines"]


def text_task(target):
    return f'reach the box labelled "{target}"'


def sample_spec(rng, pair_ids, view, pairing="lookalike"):
    """Everything that defines one episode, as plain numbers so it can be saved and replayed."""
    return dict(
        pair=rng.choice(pair_ids), pairing=pairing, order=rng.randrange(2), target_side=rng.randrange(2), view=view,
        cam_dist=round(CAM_DIST[view] * rng.uniform(0.95, 1.05), 4),
        yaws=[round(rng.uniform(-12, 12), 2) for _ in range(2)],
        light=[round(rng.uniform(-0.8, 0.8), 3), round(rng.uniform(-1.2, -0.4), 3), round(rng.uniform(1.0, 1.8), 3)],
        shade=round(rng.uniform(-0.06, 0.06), 3),
        start=[round(rng.uniform(-0.03, 0.03), 4), round(rng.uniform(-0.12, -0.08), 4), round(rng.uniform(0.20, 0.24), 4)],
    )


def goal(side):
    """Where the end-effector should end up for the left (0) or right (1) carton: just above its top."""
    return np.array([(-1 if side == 0 else 1) * BOX_X, 0.0, 2 * BOX_HALF[2] + 0.02])


def expert_path(spec):
    """Minimum-jerk path from the start to the target carton, then hold. Returns EPISODE_STEPS + 1 positions."""
    p0, p1 = np.array(spec["start"]), goal(spec["target_side"])
    pts = []
    for t in range(EPISODE_STEPS + 1):
        s = min(1.0, t / STEPS_MOVE)
        pts.append(p0 + (p1 - p0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))
    return np.array(pts, dtype=np.float32)


def outcome(pos):
    d = [float(np.linalg.norm(np.asarray(pos) - goal(k))) for k in (0, 1)]
    return 0 if d[0] <= REACH_TOL else (1 if d[1] <= REACH_TOL else -1)


class ReachEnv:
    def __init__(self, tex_dir):
        self.tex_dir = os.path.abspath(tex_dir)
        os.makedirs(self.tex_dir, exist_ok=True)
        self.renderer = None

    def _texture(self, pair, which, pairing="lookalike"):
        path = os.path.join(self.tex_dir, f"label_{pairing}_{pair}_{which}.png")
        if not os.path.exists(path):
            draw_label(pair_lines(pair, pairing)[which], MASTER_CAP).save(path)
        return path

    def texts(self, spec):
        """Label text on the left and on the right carton."""
        lines = pair_lines(spec["pair"], spec.get("pairing", "lookalike"))
        idx = [0, 1] if spec["order"] == 0 else [1, 0]
        return [" ".join(lines[i]) for i in idx]

    def target_text(self, spec):
        return self.texts(spec)[spec["target_side"]]

    def reset(self, spec):
        self.spec = spec
        order = [0, 1] if spec["order"] == 0 else [1, 0]
        tgt = np.array([0.0, -BOX_HALF[1], BOX_HALF[2]])
        self.cam = tgt + spec["cam_dist"] * np.array([0.0, -math.cos(ELEV), math.sin(ELEV)])
        assets, geoms, self.label_corners, self.face_corners = [], [], [], []
        scale = spec.get("cap_m", CAP_M) / CAP_M  # smaller print shrinks the whole label
        lab = (LABEL_HALF[0] * scale, LABEL_HALF[1] * scale)
        for i in range(2):
            yaw = math.radians(spec["yaws"][i])
            c, s = math.cos(yaw), math.sin(yaw)
            bx = np.array([(-1 if i == 0 else 1) * BOX_X, 0.0, BOX_HALF[2]])
            normal, ex, ez = np.array([s, -c, 0.0]), np.array([c, s, 0.0]), np.array([0.0, 0.0, 1.0])
            face = bx + normal * BOX_HALF[1]
            lc = bx + normal * (BOX_HALF[1] + 0.0004)
            tex = self._texture(spec["pair"], order[i], spec.get("pairing", "lookalike"))
            assets.append(f'<texture name="t{i}" type="2d" file="{tex}"/>'
                          f'<material name="m{i}" texture="t{i}" texrepeat="1 1" texuniform="false" specular="0" shininess="0"/>')
            geoms.append(f'<geom type="box" size="{BOX_HALF[0]} {BOX_HALF[1]} {BOX_HALF[2]}" pos="{bx[0]} {bx[1]} {bx[2]}" '
                         f'xyaxes="{c} {s} 0 {-s} {c} 0" rgba="0.93 0.93 0.9 1"/>')
            geoms.append(f'<geom type="plane" size="{lab[0]} {lab[1]} 0.01" pos="{lc[0]} {lc[1]} {lc[2]}" '
                         f'xyaxes="{c} {s} 0 0 0 1" material="m{i}"/>')
            self.label_corners.append([lc + sx * lab[0] * ex + sz * lab[1] * ez for sx in (-1, 1) for sz in (-1, 1)])
            self.face_corners.append([face + sx * BOX_HALF[0] * ex + sz * BOX_HALF[2] * ez for sx in (-1, 1) for sz in (-1, 1)])
        lx, ly, lz = spec["light"]
        tr = 0.70 + spec["shade"]
        p0 = spec["start"]
        a = ELEV
        xml = f"""
<mujoco>
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <quality offsamples="8" shadowsize="8192"/>
    <headlight ambient="0.42 0.42 0.42" diffuse="0.45 0.45 0.45" specular="0 0 0"/>
  </visual>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <light pos="{lx} {ly} {lz}" dir="{-lx} {-ly} {-lz}" diffuse="0.55 0.55 0.55" castshadow="false"/>
    <geom type="plane" size="2 2 0.01" rgba="{tr} {tr - 0.10} {tr - 0.22} 1"/>
    <geom type="plane" pos="0 0.5 0" xyaxes="1 0 0 0 0 1" size="2 1.5 0.01" rgba="0.82 0.84 0.86 1"/>
    {''.join(geoms)}
    <body name="ee" mocap="true" pos="{p0[0]} {p0[1]} {p0[2]}">
      <geom type="sphere" size="0.013" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.004 0.004 0.014" pos="-0.013 0 -0.022" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>
      <geom type="box" size="0.004 0.004 0.014" pos="0.013 0 -0.022" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>
    </body>
    <camera name="cam" pos="{self.cam[0]} {self.cam[1]} {self.cam[2]}" xyaxes="1 0 0 0 {math.sin(a)} {math.cos(a)}" fovy="{FOVY}"/>
  </worldbody>
</mujoco>"""
        if self.renderer is not None:
            self.renderer.close()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=H, width=W)
        return self.pos()

    def pos(self):
        return np.array(self.data.mocap_pos[0], dtype=np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), [-0.3, -0.3, 0.0], [0.3, 0.3, 0.4])
        self.data.mocap_pos[0] = a
        mujoco.mj_forward(self.model, self.data)
        return self.pos()

    def render(self):
        """The 1280x720 camera frame."""
        self.renderer.update_scene(self.data, camera="cam")
        return Image.fromarray(self.renderer.render())

    def _project(self, p):
        xc, yc, zc = np.array([1.0, 0, 0]), np.array([0, math.sin(ELEV), math.cos(ELEV)]), np.array([0, -math.cos(ELEV), math.sin(ELEV)])
        v = np.asarray(p) - self.cam
        depth = -float(v @ zc)
        return W / 2 + F_PX * float(v @ xc) / depth, H / 2 - F_PX * float(v @ yc) / depth

    def _bbox(self, corners, margin):
        uv = [self._project(c) for c in corners]
        us, vs = [u for u, _ in uv], [v for _, v in uv]
        mx, my = margin * (max(us) - min(us)), margin * (max(vs) - min(vs))
        return [max(0, min(us) - mx), max(0, min(vs) - my), min(W, max(us) + mx), min(H, max(vs) + my)]

    def label_crops(self, frame):
        """The two labels cut from the camera frame at sensor resolution, for a reader."""
        return [frame.crop(tuple(int(round(v)) for v in self._bbox(c, 0.06))) for c in self.label_corners]

    def zoom_box(self):
        """A 16:9 region of the frame around both cartons, used for the zoomed policy input."""
        boxes = [self._bbox(c, 0.0) for c in self.face_corners]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max((x1 - x0) * 1.25, (y1 - y0) * 1.25 * 16 / 9)
        h = w * 9 / 16
        x0, y0 = min(max(0, cx - w / 2), W - w), min(max(0, cy - h / 2), H - h)
        return (x0, y0, x0 + w, y0 + h)

    def observation(self, frame, pointer_side=None, zoom=False):
        """What the policy receives: the frame at 640x360, with a green outline on one carton if asked.
        With zoom=True the frame is first cropped to the cartons, so the label text is larger."""
        if pointer_side is not None:
            frame = frame.copy()
            ImageDraw.Draw(frame).rectangle(self._bbox(self.face_corners[pointer_side], 0.05), outline=(0, 200, 60), width=10)
        if zoom:
            frame = frame.crop(tuple(int(round(v)) for v in self.zoom_box()))
        return np.asarray(frame.resize((OBS_W, OBS_H), Image.LANCZOS), dtype=np.uint8)

    def label_views(self, frame):
        """Two extra camera inputs: a close-up of each label, as a zoom or wrist camera would give.
        Each label crop is scaled to fill a 640x360 image (aspect kept, grey padding)."""
        views = []
        for crop in self.label_crops(frame):
            s = min(OBS_W / crop.width, OBS_H / crop.height)
            small = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
            canvas = Image.new("RGB", (OBS_W, OBS_H), (128, 128, 128))
            canvas.paste(small, ((OBS_W - small.width) // 2, (OBS_H - small.height) // 2))
            views.append(np.asarray(canvas, dtype=np.uint8))
        return views

    def text_px_in_label_view(self, policy_width=512):
        """Height of a capital letter in a label close-up after the policy's resize to 512."""
        crop_w = self._bbox(self.label_corners[0], 0.06)
        s = min(OBS_W / (crop_w[2] - crop_w[0]), OBS_H / (crop_w[3] - crop_w[1]))
        return self.text_px() * s * policy_width / OBS_W

    def text_px_in_input(self, zoom=False, policy_width=512):
        """Height of a capital letter in the image the policy finally sees (after its resize to 512)."""
        width = (self.zoom_box()[2] - self.zoom_box()[0]) if zoom else W
        return self.text_px() * policy_width / width

    def text_px(self):
        """Height of a capital letter on the sensor, in pixels."""
        return F_PX * self.spec.get("cap_m", CAP_M) / self.spec["cam_dist"]

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def make_specs(n, pair_ids, view, seed, pairing="lookalike"):
    rng = random.Random(seed)
    return [sample_spec(rng, pair_ids, view, pairing) for _ in range(n)]
