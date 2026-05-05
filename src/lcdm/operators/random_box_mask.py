from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


def _stable_int(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


@dataclass
class RandomBoxMaskOperator:
    box_h: int = 100
    box_w: int = 100
    mask_seed: int = 1234

    def __post_init__(self):
        self.mask_o = None
        self.mask_m = None
        self.last_boxes = None

    def _sample_box(self, sample_id: str, H: int, W: int):
        if self.box_h > H or self.box_w > W:
            raise ValueError(
                f"Box size ({self.box_h},{self.box_w}) is larger than image size ({H},{W})"
            )

        seed = _stable_int(f"{self.mask_seed}:{sample_id}") % (2**31)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        top = torch.randint(0, H - self.box_h + 1, (1,), generator=g).item()
        left = torch.randint(0, W - self.box_w + 1, (1,), generator=g).item()
        return top, left

    def set_batch(self, x: torch.Tensor, sample_ids: list[str]):
        B, C, H, W = x.shape
        if len(sample_ids) != B:
            raise ValueError(f"len(sample_ids)={len(sample_ids)} but batch={B}")

        mask_o = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
        boxes = []

        for b, sid in enumerate(sample_ids):
            top, left = self._sample_box(str(sid), H, W)
            mask_o[b, :, top : top + self.box_h, left : left + self.box_w] = 0.0
            boxes.append(
                {
                    "sample_id": str(sid),
                    "top": int(top),
                    "left": int(left),
                    "height": int(self.box_h),
                    "width": int(self.box_w),
                }
            )

        self.mask_o = mask_o.repeat(1, C, 1, 1)
        self.mask_m = 1.0 - self.mask_o
        self.last_boxes = boxes

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_pn(x)

    def apply_pn(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_o is None:
            raise RuntimeError("Call operator.set_batch(...) before apply_pn.")
        return self.mask_o.to(x.device) * x

    def apply_pk(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.apply_pn(x)

    def lift_observation(self, y: torch.Tensor) -> torch.Tensor:
        # For inpainting, observation is already HR, so no lifting is needed.
        return y
