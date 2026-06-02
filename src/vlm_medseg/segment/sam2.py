"""SAM2 prompted segmentation (HF ``Sam2Model`` + ``Sam2Processor``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from ..types import Box, Point

DEFAULT_MODEL_ID = "facebook/sam2-hiera-large"


def resolve_device(spec: str = "auto") -> str:
    import torch

    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Sam2Masker:
    """One model load serving both box- and point-prompted segmentation.

    ``masks_from_boxes`` / ``masks_from_points`` return one ``(H, W)`` boolean
    mask per prompt at the original resolution; all prompts for an image run in
    a single batched forward.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto", dtype=None) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = resolve_device(device)
        if dtype is None:
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.dtype = dtype
        self.model = Sam2Model.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.processor = Sam2Processor.from_pretrained(model_id)

    @staticmethod
    def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
        return image if isinstance(image, Image.Image) else Image.fromarray(image)

    def _post(self, outputs, inputs):
        return self.processor.post_process_masks(
            outputs.pred_masks.cpu().float(), inputs["original_sizes"]
        )

    def masks_from_boxes(
        self, image: np.ndarray | Image.Image, boxes: Sequence[Box]
    ) -> list[np.ndarray]:
        import torch

        if not boxes:
            return []
        input_boxes = [[list(map(float, b)) for b in boxes]]
        inputs = self.processor(
            images=self._to_pil(image), input_boxes=input_boxes, return_tensors="pt"
        ).to(self.device, self.dtype)
        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)
        return [m[0].numpy().astype(bool) for m in self._post(outputs, inputs)[0]]

    def masks_from_points(
        self,
        image: np.ndarray | Image.Image,
        points: Sequence[Point],
        *,
        multimask_output: bool = True,
        select: str = "score",
    ) -> tuple[list[np.ndarray], list[float]]:
        """One mask per point.

        SAM2 returns three masks per point at increasing scale. ``select`` keeps
        the ``"score"`` (highest predicted IoU; SAM2 default), ``"smallest"``, or
        ``"largest"`` candidate. ``"smallest"`` is usually the nucleus scale —
        the default ``"score"`` over-segments dense tiny nuclei.
        """
        import torch

        if not points:
            return [], []
        input_points = [[[[float(x), float(y)]] for x, y in points]]
        input_labels = [[[1] for _ in points]]
        inputs = self.processor(
            images=self._to_pil(image),
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device, self.dtype)
        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=multimask_output)
        post = self._post(outputs, inputs)[0]          # (N, num_masks, H, W)
        scores_all = outputs.iou_scores[0].cpu().numpy()
        masks_out, scores_out = [], []
        for i in range(post.shape[0]):
            cand = post[i].numpy().astype(bool)
            scs = scores_all[i]
            if select in ("smallest", "largest") and cand.shape[0] > 1:
                areas = cand.reshape(cand.shape[0], -1).sum(axis=1).astype(float)
                areas[areas == 0] = np.inf if select == "smallest" else -1
                idx = int(np.argmin(areas)) if select == "smallest" else int(np.argmax(areas))
            else:
                idx = int(np.argmax(scs))
            masks_out.append(cand[idx])
            scores_out.append(float(scs[idx]))
        return masks_out, scores_out
