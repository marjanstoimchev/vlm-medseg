"""Frozen feature encoders for nucleus crops.

Pluggable via :func:`build_encoder` / the ``--encoder`` flag:

  - ``uni2h``  — MahmoodLab UNI2-h, a pathology foundation ViT-H (default; H&E-domain).
  - ``dinov3`` — facebook DINOv3 ViT-B/16, a newer general-purpose SSL model
                 (smaller/faster, natural-image pretraining — a strong baseline).

Each maps a list of RGB crops to an ``(N, dim)`` feature matrix; the encoder is
frozen (only a linear probe is trained on top).
"""

from __future__ import annotations

import functools

import numpy as np
from PIL import Image

from ..segment.sam2 import resolve_device

# Documented UNI2-h creation kwargs (ViT-H/14 + SwiGLU + 8 register tokens).
_UNI2H_KWARGS = dict(
    img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
    embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
    no_embed_class=True, reg_tokens=8, dynamic_img_size=True,
)


class _CropEncoder:
    """Shared resize/normalize + batched embedding. Subclasses load the model
    and implement :meth:`_forward` (a numpy batch -> ``(B, dim)`` features)."""

    name = "encoder"
    dim = 0

    def __init__(self, device: str = "auto", batch_size: int = 64):
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet defaults
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        im = Image.fromarray(crop).convert("RGB").resize((224, 224), Image.BICUBIC)
        return ((np.asarray(im, np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)

    def _forward(self, batch: np.ndarray) -> np.ndarray:  # (B,3,224,224) -> (B,dim)
        raise NotImplementedError

    def embed(self, crops: list[np.ndarray], *, progress: bool = False) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        starts = list(range(0, len(crops), self.batch_size))
        if progress:
            try:
                from tqdm.auto import tqdm
                starts = tqdm(starts, desc=f"embed {self.name}", unit="batch")
            except Exception:
                pass
        out = [self._forward(np.stack([self._prep(c) for c in crops[i:i + self.batch_size]]))
               for i in starts]
        return np.concatenate(out, 0)


class Uni2hEncoder(_CropEncoder):
    name = "uni2h"
    model_id = "MahmoodLab/UNI2-h"
    dim = 1536

    def __init__(self, device: str = "auto", batch_size: int = 64):
        super().__init__(device, batch_size)
        import timm
        import torch

        self.torch = torch
        kw = dict(_UNI2H_KWARGS, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        self.model = (
            timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw)
            .eval().to(self.device)
        )
        cfg = timm.data.resolve_model_data_config(self.model)
        self.mean = np.asarray(cfg["mean"], np.float32)
        self.std = np.asarray(cfg["std"], np.float32)

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(batch).to(self.device)
        with self.torch.inference_mode():
            return self.model(t).float().cpu().numpy()


class Dinov3Encoder(_CropEncoder):
    name = "dinov3"
    model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # default DINOv3 = ViT-L/16

    def __init__(self, device: str = "auto", batch_size: int = 64,
                 model_id: str | None = None, name: str | None = None):
        super().__init__(device, batch_size)
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        if name:
            self.name = name
        mid = model_id or self.model_id
        self.model = AutoModel.from_pretrained(mid).eval().to(self.device)
        self.dim = int(self.model.config.hidden_size)
        try:
            proc = AutoImageProcessor.from_pretrained(mid)
            self.mean = np.asarray(proc.image_mean, np.float32)
            self.std = np.asarray(proc.image_std, np.float32)
        except Exception:
            pass  # keep ImageNet defaults

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(batch).to(self.device)
        with self.torch.inference_mode():
            out = self.model(pixel_values=t)
        feat = getattr(out, "pooler_output", None)
        if feat is None:
            feat = out.last_hidden_state[:, 0]  # CLS token
        return feat.float().cpu().numpy()


_DINOV3_VARIANTS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
_ENCODERS = {"uni2h": Uni2hEncoder}
for _name, _mid in _DINOV3_VARIANTS.items():
    _ENCODERS[_name] = functools.partial(Dinov3Encoder, model_id=_mid, name=_name)
# Default DINOv3 alias -> ViT-L/16 (scale-matched to UNI2-h's ViT-H).
_ENCODERS["dinov3"] = _ENCODERS["dinov3_vitl16"]


def available_encoders() -> list[str]:
    return sorted(_ENCODERS)


def build_encoder(name: str = "uni2h", **kwargs):
    if name not in _ENCODERS:
        raise KeyError(f"unknown encoder '{name}'; available: {available_encoders()}")
    return _ENCODERS[name](**kwargs)
