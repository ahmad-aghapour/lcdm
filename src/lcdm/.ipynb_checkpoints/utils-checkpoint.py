from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping/dict.")
    return data


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_for_process(base_seed: int, process_index: int) -> int:
    return int(base_seed) + int(process_index)


def normalize_images_to_model(x):
    """
    Convert [0, 1] image tensor to [-1, 1].
    """
    return 2.0 * x - 1.0


def denormalize_images_from_model(x):
    """
    Convert [-1, 1] tensor to [0, 1].
    """
    return (x / 2.0 + 0.5).clamp(0, 1)


def get_file_list(input_dir: str | Path, file_ext: str, max_images: int | None = None):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(f"*.{file_ext}"))
    if max_images is not None:
        files = files[: int(max_images)]
    return files