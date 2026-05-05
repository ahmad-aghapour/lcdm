from __future__ import annotations

import torch
from diffusers import DDIMPipeline


def load_hf_diffusers_model(cfg, device="cpu"):
    pretrained_id = cfg["pretrained_id"]
    use_fp16 = bool(cfg.get("use_fp16", False))

    torch_dtype = torch.float16 if use_fp16 else torch.float32

    pipe = DDIMPipeline.from_pretrained(
        pretrained_id,
        torch_dtype=torch_dtype,
    )

    unet = pipe.unet
    unet.to(device)
    unet.eval()
    return unet
