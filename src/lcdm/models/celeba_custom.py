from __future__ import annotations

import types

import torch

from models import Model


def _dict_to_object(d):
    if not isinstance(d, dict):
        return d
    obj = types.SimpleNamespace()
    for k, v in d.items():
        setattr(obj, k, _dict_to_object(v))
    return obj


def default_celeba_model_config():
    return {
        "model": {
            "type": "simple",
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 1, 2, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "var_type": "fixedsmall",
            "ema_rate": 0.999,
            "ema": True,
            "resamp_with_conv": True,
        },
        "data": {
            "image_size": 256,
        },
        "diffusion": {
            "num_diffusion_timesteps": 1000,
        },
    }


def load_celeba_custom_model(cfg, device="cpu"):
    config_dict = cfg.get("arch_config", default_celeba_model_config())
    config = _dict_to_object(config_dict)

    model = Model(config)
    state_dict = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
