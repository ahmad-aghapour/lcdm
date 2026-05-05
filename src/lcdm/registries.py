from __future__ import annotations

from lcdm.datasets.celebahq import build_celebahq_dataset
from lcdm.datasets.imagenet import build_imagenet_dataset
from lcdm.datasets.lsun_church import build_lsun_church_dataset

from lcdm.models.hf_diffusers import load_hf_diffusers_model
from lcdm.models.celeba_custom import load_celeba_custom_model
from lcdm.models.openai_unet import load_openai_unet_model

from lcdm.operators.mask import build_mask_operator
from lcdm.operators.mean_sr import build_mean_sr_operator
from lcdm.operators.random_box_mask import RandomBoxMaskOperator


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
    name = cfg["operator"]["name"]

    if name in OPERATOR_BUILDERS:
        return OPERATOR_BUILDERS[name](cfg["operator"])

    if name == "random_box_mask":
        return RandomBoxMaskOperator(
            box_h=cfg["operator"].get("box_h", 100),
            box_w=cfg["operator"].get("box_w", 100),
            mask_seed=cfg["operator"].get("mask_seed", 1234),
        )

    raise ValueError(f"Unknown operator: {name}")
