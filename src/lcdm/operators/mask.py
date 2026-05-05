from __future__ import annotations

import torch


class BoxMaskOperator:
    """
    Inpainting operator using a rectangular missing region around the image center.

    Parameters are interpreted relative to the image center:
      y0 = cy - top
      y1 = cy + bottom
      x0 = cx - left
      x1 = cx + right

    The missing region is set to 0 in the observation mask.
    """

    def __init__(self, top: int, bottom: int, left: int, right: int):
        self.top = int(top)
        self.bottom = int(bottom)
        self.left = int(left)
        self.right = int(right)

    def build_masks(self, x: torch.Tensor):
        _, _, H, W = x.shape
        mask_obs = torch.ones_like(x)

        cy, cx = H // 2, W // 2

        y0 = max(0, cy - self.top)
        y1 = min(H, cy + self.bottom)
        x0 = max(0, cx - self.left)
        x1 = min(W, cx + self.right)

        mask_obs[..., y0:y1, x0:x1] = 0.0
        mask_miss = 1.0 - mask_obs
        return mask_obs, mask_miss

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        mask_obs, _ = self.build_masks(x)
        return mask_obs * x

    def apply_pn(self, x: torch.Tensor) -> torch.Tensor:
        mask_obs, _ = self.build_masks(x)
        return mask_obs * x

    def apply_pk(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.apply_pn(x)

    def lift_observation(self, y: torch.Tensor) -> torch.Tensor:
        return y


def build_mask_operator(cfg):
    return BoxMaskOperator(
        top=cfg["top"],
        bottom=cfg["bottom"],
        left=cfg["left"],
        right=cfg["right"],
    )
