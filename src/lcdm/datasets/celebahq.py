from __future__ import annotations

from lcdm.datasets.base import ImageFolderDataset


def build_celebahq_dataset(cfg):
    return ImageFolderDataset(
        input_dir=cfg["input_dir"],
        image_size=cfg.get("image_size", 256),
        file_ext=cfg.get("file_ext", "png"),
        max_images=cfg.get("max_images", 1000),
        center_crop=cfg.get("center_crop", False),
    )
