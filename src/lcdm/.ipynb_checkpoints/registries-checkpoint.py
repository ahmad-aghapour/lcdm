from __future__ import annotations

from lcdm.datasets.celebahq import build_celebahq_dataset
from lcdm.datasets.imagenet import build_imagenet_dataset
from lcdm.datasets.lsun_church import build_lsun_church_dataset

from lcdm.models.hf_diffusers import load_hf_diffusers_model
from lcdm.models.celeba_custom import load_celeba_custom_model
from lcdm.models.openai_unet import load_openai_unet_model

from lcdm.operators import BoxMaskOperator, MeanDownsampleOperator, RandomBoxMaskOperator

DATASET_BUILDERS = {
    "celebahq": build_celebahq_dataset,
    "imagenet": build_imagenet_dataset,
    "lsun_church": build_lsun_church_dataset,
}

MODEL_BUILDERS = {
    "hf_diffusers": load_hf_diffusers_model,
    "celeba_custom": load_celeba_custom_model,
    "openai_unet": load_openai_unet_model,
}

OPERATOR_BUILDERS = {
    "mask": build_mask_operator,
    "mean_sr": build_mean_sr_operator,
}


def build_dataset(cfg):
    name = cfg["dataset"]["name"]
    if name not in DATASET_BUILDERS:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_BUILDERS[name](cfg["dataset"])


def build_model(cfg, device):
    name = cfg["model"]["name"]
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_BUILDERS[name](cfg["model"], device=device)


def build_operator(cfg):
    name = cfg["name"]

    if name == "mask":
        return BoxMaskOperator(
            top=cfg["top"],
            bottom=cfg["bottom"],
            left=cfg["left"],
            right=cfg["right"],
        )

    if name == "mean_downsample":
        return MeanDownsampleOperator(
            scale=cfg.get("scale", 8)
        )

    if name == "random_box_mask":
        return RandomBoxMaskOperator(
            box_h=cfg.get("box_h", 100),
            box_w=cfg.get("box_w", 100),
            mask_seed=cfg.get("mask_seed", 1234),
        )

    raise ValueError(f"Unknown operator: {name}")