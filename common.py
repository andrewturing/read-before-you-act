"""Shared helpers for the read-before-you-act pilot: model wrapper, text metrics, image views."""
import math
import re
import unicodedata

import numpy as np
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

# kind: "ocr" = specialist reader with a fixed task prompt; "vlm" = general small VLM.
# Notes on why each model is here:
#   smolvlm2-500m        backbone family of SmolVLA (compact robot policy)
#   smolvlm2-500m-64tok  same, with tiling off -> one 512px image = 64 visual tokens, as SmolVLA feeds it
#   qwen2.5-vl-3b        backbone of NORA-class robot policies
#   qwen3.5-2b           current small general VLM
#   glm-ocr, paddleocr   current ~0.9B specialist readers
MODEL_SPECS = {
    "smolvlm2-500m": dict(repo="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", kind="vlm"),
    "smolvlm2-500m-64tok": dict(repo="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", kind="vlm", no_split=True),
    "smolvlm2-2.2b": dict(repo="HuggingFaceTB/SmolVLM2-2.2B-Instruct", kind="vlm"),
    "qwen2.5-vl-3b": dict(repo="Qwen/Qwen2.5-VL-3B-Instruct", kind="vlm"),
    "qwen3.5-2b": dict(repo="Qwen/Qwen3.5-2B", kind="vlm"),
    "glm-ocr": dict(repo="zai-org/GLM-OCR", kind="ocr", ocr_prompt="Text Recognition:"),
    "paddleocr-vl-1.6": dict(repo="PaddlePaddle/PaddleOCR-VL-1.6", kind="ocr", ocr_prompt="OCR:"),
}

VLM_READ_PROMPT = "Read the text in this image and output only that text, exactly as written."


def norm_text(s: str) -> str:
    """Normalise so that scoring ignores differences that are not reading errors: full-width vs
    ASCII punctuation (NFKC), case, line breaks, and the spaces some corpora put before punctuation."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\n", " ").strip().lower()
    s = re.sub(r"\s+([.,;:!?%)\]}'\"])", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s


def strip_wrappers(s: str) -> str:
    """Drop a markdown code fence around the answer (some readers wrap output in ```markdown ... ```)."""
    s = (s or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def count_alnum(s: str) -> int:
    """Letters, digits and CJK characters only, so placeholders such as '---' do not count as text."""
    return sum(ch.isalnum() for ch in unicodedata.normalize("NFKC", s or ""))


def cer(pred: str, ref: str) -> float:
    p, r = norm_text(pred), norm_text(ref)
    if not r:
        return 0.0 if not p else 1.0
    return Levenshtein.distance(p, r) / len(r)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def resize_with_pad(img: Image.Image, size: int) -> Image.Image:
    """Aspect-preserving resize onto a black square, as robot-policy pipelines do."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = img.resize((nw, nh), Image.BICUBIC)
    out = Image.new("RGB", (size, size), (0, 0, 0))
    out.paste(small, ((size - nw) // 2, (size - nh) // 2))
    return out


def upscale_min_side(img: Image.Image, min_side: int = 64) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    if m >= min_side:
        return img
    f = math.ceil(min_side / m)
    return img.resize((w * f, h * f), Image.BICUBIC)


class _LinearAsConv3d(torch.nn.Module):
    """A Conv3d whose kernel equals its stride is a linear map over each flattened patch.
    Qwen3.5 and GLM-OCR embed patches with such a Conv3d, which ran ~10x slower than the
    rest of the model on an L4; this computes the same thing with one matrix product."""

    def __init__(self, conv):
        super().__init__()
        self.weight, self.bias, self.out = conv.weight, conv.bias, conv.out_channels

    def forward(self, x):
        y = torch.nn.functional.linear(x.flatten(1), self.weight.flatten(1), self.bias)
        return y.view(x.shape[0], self.out, 1, 1, 1)


def _speed_up_patch_embed(model):
    for name, mod in list(model.named_modules()):
        if (isinstance(mod, torch.nn.Conv3d) and tuple(mod.kernel_size) == tuple(mod.stride)
                and tuple(mod.padding) == (0, 0, 0)):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], _LinearAsConv3d(mod))


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class Reader:
    """Thin wrapper: one image + one prompt -> text and token log-prob confidence.

    `key` is a short name from MODEL_SPECS or any Hugging Face repository id of an
    image-text-to-text model that ships a chat template."""

    def __init__(self, key: str, precision: str = "bf16", device: str = None, kind: str = None, prompt: str = None):
        import transformers
        from transformers import AutoProcessor

        self.key, self.precision = key, precision
        known = MODEL_SPECS.get(key) or next(
            (v for v in MODEL_SPECS.values() if v["repo"] == key and not v.get("no_split")), None)
        self.spec = dict(known) if known else dict(repo=key, kind="vlm")
        if kind:
            self.spec["kind"] = kind
        if prompt:
            self.spec["ocr_prompt"] = prompt
        device = device or pick_device()
        # bfloat16 on NVIDIA GPUs, half precision on Apple GPUs, full precision on CPU
        self.dtype = torch.bfloat16 if device.startswith("cuda") else (torch.float16 if device == "mps" else torch.float32)
        repo = self.spec["repo"]
        kw = {}
        if precision == "4bit":
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            kw["device_map"] = {"": device}
        self.processor = AutoProcessor.from_pretrained(repo)
        if self.spec.get("no_split"):
            self.processor.image_processor.do_image_splitting = False
        last = None
        self.model = None
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            for dkw in ({"dtype": self.dtype}, {"torch_dtype": self.dtype}):
                try:
                    self.model = cls.from_pretrained(repo, **dkw, **kw)
                    break
                except TypeError as e:  # dtype kwarg name differs across versions
                    last = e
                except ValueError as e:  # model type not mapped for this auto class
                    last = e
                    break
            if self.model is not None:
                break
        if self.model is None:
            raise RuntimeError(f"could not load {repo}: {last}")
        if precision != "4bit":
            self.model = self.model.to(device)
        _speed_up_patch_embed(self.model)
        self.model.eval()
        self.device = device
        # Stop at any end-of-turn token. Without this, a model that correctly returns nothing can run
        # on into a new turn, and that spill-over would be scored as fabricated text.
        tok = getattr(self.processor, "tokenizer", self.processor)
        stop = set()
        eos = self.model.generation_config.eos_token_id
        for e in (eos if isinstance(eos, (list, tuple)) else [eos]):
            if e is not None:
                stop.add(int(e))
        if tok.eos_token_id is not None:
            stop.add(int(tok.eos_token_id))
        vocab = tok.get_vocab()
        for t in ("<|im_end|>", "<|endoftext|>", "<end_of_utterance>", "<|eot_id|>", "<|user|>", "<|im_start|>", "</s>"):
            if t in vocab:
                stop.add(int(vocab[t]))
        self.stop_ids = sorted(stop)

    @property
    def kind(self):
        return self.spec["kind"]

    def read_prompt(self):
        return self.spec.get("ocr_prompt", VLM_READ_PROMPT)

    @torch.inference_mode()
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 64):
        messages = [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device, dtype=self.dtype)
        n_in = inputs["input_ids"].shape[-1]
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_scores=True, return_dict_in_generate=True, eos_token_id=self.stop_ids,
        )
        seq = out.sequences[0, n_in:]
        text = strip_wrappers(self.processor.decode(seq, skip_special_tokens=True))
        lp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
        lp = lp.float().cpu().numpy()
        lp = lp[np.isfinite(lp)]
        conf = dict(
            mean_lp=float(lp.mean()) if len(lp) else 0.0,
            min_lp=float(lp.min()) if len(lp) else 0.0,
            n_tok=int(len(lp)),
            n_img_tok=int(n_in),
        )
        return text, conf

    def close(self):
        del self.model
        torch.cuda.empty_cache()
