"""
Batch‑wise CelebA‑HQ‑256 in‑painting with projector‑aware Langevin + DPM‑Solver,
parallelised by 🤗 Accelerate.

• Splits the first 1 000 images across all visible GPUs/TPU cores  
• Each worker processes mini‑batches of BATCH=8 locally  
• Rank 0 gathers the completed samples and writes PNGs to OUT_DIR/

Requires:  accelerate ≥ 0.24, diffusers ≥ 0.26, torch ≥ 1.13, pillow, tqdm
"""


import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

import types



import torchvision.utils as tvu

from models import Model
from script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth
import math, torch
from typing import Callable
def _randn_like(t, generator=None):
    if generator is None:
        return torch.randn_like(t)
    return torch.randn(t.shape,
                       dtype=t.dtype,
                       device=t.device,
                       generator=generator)



# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch

from diffusers.models import UNet2DModel
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


import math, os, glob, numpy as np, torch, tqdm
from pathlib import Path
from PIL import Image
from accelerate import Accelerator



# ------------------------------------------------------------
# 0 · config (shared by all ranks)
# ------------------------------------------------------------
IN_DIR   = Path("image_net/imagenet")      # 1 000 PNGs
OUT_DIR  = Path("imagenet_val")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH    = 8
T_STEPS  = 100

MASK_H   = 120
MASK_W   = 120
def build_masks(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, H, W = x.shape
    mask_O = torch.ones_like(x)
    cy, cx = H // 2, W // 2
    mask_O[..., cy-MASK_W:cy, cx:cx+MASK_H] = 0
    return mask_O, 1 - mask_O
# ------------------------------------------------------------
# 1 · accelerator + pipeline (prepared once per rank)
# ------------------------------------------------------------
accelerator = Accelerator()       # set True if you want half‑precision
device      = accelerator.device


config_dict = {
    'model': {
        'type': "openai",
        'in_channels': 3,
        'out_channels': 3,
        'num_channels': 256,
        'num_heads': 4,
        'num_res_blocks': 2,
        'attention_resolutions': "32,16,8",
        'dropout': 0.0,
        'resamp_with_conv': True,
        'learn_sigma': True,
        'use_scale_shift_norm': True,
        'use_fp16': True,
        'resblock_updown': True,
        'num_heads_upsample': -1,
        'var_type': 'fixedsmall',
        'num_head_channels': 64,
        'image_size': 256,
        'class_cond': False,
        'use_new_attention_order': False
    },
}
def dict_to_object(d):
    """Recursively converts a dictionary and its nested dictionaries into a SimpleNamespace object."""
    if not isinstance(d, dict):
        return d
    obj = types.SimpleNamespace()
    for k, v in d.items():
        setattr(obj, k, dict_to_object(v))
    return obj

# config = dict_to_object(config_dict)




# config_dict = vars(config.model)
# model = create_model(**config_dict)

# if config.model.use_fp16:
#   model.convert_to_fp16()


# model.load_state_dict(torch.load('256x256_diffusion_uncond.pt', map_location=device))
# model.to(device)
# model.eval()

# scheduler = DDIMScheduler.from_pretrained(
#     "google/ddpm-celebahq-256", torch_dtype=torch.float16
# )
# pipe = DDIMPipelineDDNM(unet=model, scheduler=scheduler).to(device)


# # wrap the pipeline for DDP / AMP
# pipe = accelerator.prepare(pipe)

# ------------------------------------------------------------
# 2 · helper functions  (device‑agnostic)
# ------------------------------------------------------------
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from functools import partial
from pathlib import Path
from PIL import Image

class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__



# build the identical transform pipeline you pass to ImageFolder
crop_and_to_tensor = transforms.Compose([
                    CenterCropLongEdge(),
                    transforms.Resize(256),
                    transforms.ToTensor()
])

def tensor_from_png(fn: Path) -> torch.Tensor:
    """Load a PNG from disk, center‑crop to config.data.image_size, and return a Torch C×H×W float tensor."""
    img = Image.open(fn).convert("RGB")
    return crop_and_to_tensor(img)


# ------------------------------------------------------------
# 3 · distribute filenames across processes
# ------------------------------------------------------------
from accelerate.utils import gather_object   # add once at the top

# ------------------------------------------------------------
# 3 · distribute filenames across processes
# ------------------------------------------------------------
all_files   = sorted(IN_DIR.glob("*.jpg"))[:1000]
with accelerator.split_between_processes(all_files) as local_files:
    local_files = list(local_files)      # now a real list

# ------------------------------------------------------------
# 4 · main loop (each rank handles its subset)
# ------------------------------------------------------------
for i in tqdm.trange(
        0, len(local_files), BATCH,
        disable=not accelerator.is_local_main_process):
    batch_files = local_files[i:i+BATCH]
    if not batch_files:                      # last rank may have <BATCH imgs
        continue

    imgs = [tensor_from_png(f) for f in batch_files]   # CPU tensors
    y    = torch.stack(imgs).to(device)                # (B,3,H,W) float32

    mask_O, mask_M = build_masks(y)
    y_O   = 2 * y * mask_O - 1.0

    gen = torch.Generator(device=device).manual_seed(42 + accelerator.process_index)

    # out_pil = pipe(
    # y_O                 = y_O,
    # mask_M              = mask_M,
    # mask_O              = mask_O,
    # num_inference_steps = 100,
    # batch_size          = y_O.size(0),
    #  eta= 0.85 ,
    # generator           = gen,
    # ).images         
    # --------------------------------------------------------
    # gather to rank 0 and save
    # --------------------------------------------------------
    out_tensor = torch.tensor(
                 np.stack([np.array(p) for p in imgs]),
                 device=device, dtype=torch.float16)            # (b,H,W,3)

    gathered_img = accelerator.gather_for_metrics(out_tensor)      # OK now
    gathered_files = gather_object(batch_files)                   # Paths

    if accelerator.is_local_main_process:
        for arr, src_fn in zip(gathered_img.cpu().numpy(),
                               gathered_files):
            Image.fromarray(arr.astype("uint8"), mode="RGB").save(OUT_DIR / src_fn.name)

accelerator.wait_for_everyone()

# ------------------------------------------------------------
# 5 · only rank‑0 prints the summary
# ------------------------------------------------------------
if accelerator.is_local_main_process:
    total = len(list(OUT_DIR.glob("*.jpg")))
    print(f"Completed {total} images →  {OUT_DIR}/")

# ------------------------------------------------------------
# 6 · tear down the distributed process‑group explicitly
# ------------------------------------------------------------
accelerator.end_training()