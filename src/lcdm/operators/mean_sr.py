from __future__ import annotations

import torch


class MeanSROperator:
    """
    Mean downsample / repeat-lift super-resolution operator.

    observe(x): returns low-resolution observation
    lift_observation(y): lifts LR to HR by repeating each pixel over an s x s block
    apply_pn(x): projection P = A^T (A A^T)^(-1) A
    apply_pk(x): I - P
    """

    def __init__(self, scale: int = 8):
        self.scale = int(scale)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        s = self.scale
        if h % s != 0 or w % s != 0:
            raise ValueError(f"Image shape {(h, w)} must be divisible by scale={s}.")
        xv = x.view(n, c, h // s, s, w // s, s)
        return xv.mean(dim=(3, 5))

    def upsample(self, x_lr: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x_lr.shape
        s = self.scale
        out = torch.zeros(n, c, h, s, w, s, device=x_lr.device, dtype=x_lr.dtype)
        out = out + x_lr.view(n, c, h, 1, w, 1)
        return out.view(n, c, s * h, s * w)

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)

    def lift_observation(self, y: torch.Tensor) -> torch.Tensor:
        return self.upsample(y)

    def apply_pn(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.downsample(x))

    def apply_pk(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.apply_pn(x)


def build_mean_sr_operator(cfg):
    return MeanSROperator(scale=cfg.get("scale", 8))
