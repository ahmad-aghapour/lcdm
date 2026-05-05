from __future__ import annotations

import torch

from script_util import create_model


def default_openai_model_config():
    return {
        "type": "openai",
        "in_channels": 3,
        "out_channels": 3,
        "num_channels": 256,
        "num_heads": 4,
        "num_res_blocks": 2,
        "attention_resolutions": "32,16,8",
        "dropout": 0.0,
        "resamp_with_conv": True,
        "learn_sigma": True,
        "use_scale_shift_norm": True,
        "use_fp16": True,
        "resblock_updown": True,
        "num_heads_upsample": -1,
        "var_type": "fixedsmall",
        "num_head_channels": 64,
        "image_size": 256,
        "class_cond": False,
        "use_new_attention_order": False,
    }


def load_openai_unet_model(cfg, device="cpu"):
    arch_config = cfg.get("arch_config", default_openai_model_config())
    model = create_model(**arch_config)

    use_fp16 = bool(cfg.get("use_fp16", arch_config.get("use_fp16", False)))
    if use_fp16:
        model.convert_to_fp16()

    state_dict = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
