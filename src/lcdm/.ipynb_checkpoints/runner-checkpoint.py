from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image

from diffusers.schedulers import DDIMScheduler

from lcdm.samplers.pipeline import ProjectedDDIMPipeline
from lcdm.utils import ensure_dir, normalize_images_to_model, seed_for_process


def run_experiment(cfg, dataset, model, operator):
    accelerator = Accelerator()
    device = accelerator.device

    scheduler = DDIMScheduler()
    pipe = ProjectedDDIMPipeline(unet=model, scheduler=scheduler).to(device)
    pipe = accelerator.prepare(pipe)

    output_dir = ensure_dir(cfg["output"]["dir"])
    batch_size = int(cfg["sampling"]["batch_size"])

    indexed_files = list(enumerate(dataset.files))
    with accelerator.split_between_processes(indexed_files) as local_pairs:
        local_pairs = list(local_pairs)

    for i in tqdm.trange(
        0,
        len(local_pairs),
        batch_size,
        disable=not accelerator.is_local_main_process,
    ):
        batch_pairs = local_pairs[i : i + batch_size]
        if not batch_pairs:
            continue

        idxs, batch_files = zip(*batch_pairs)
        samples = [dataset.load_tensor(p) for p in batch_files]
        x = torch.stack(samples)
        x_model = normalize_images_to_model(x)
        
        sample_ids = [f.name for f in batch_files]
        
        if hasattr(operator, "set_batch"):
            operator.set_batch(x_model, sample_ids)
        
        observation = operator.observe(x_model)

        gen = torch.Generator(device=device).manual_seed(
            seed_for_process(cfg["sampling"]["seed"], accelerator.process_index)
        )

        out_pil = pipe(
            observation=observation,
            operator=operator,
            batch_size=observation.size(0),
            generator=gen,
            eta=float(cfg["sampling"]["eta"]),
            num_inference_steps=int(cfg["sampling"]["num_inference_steps"]),
            sampler_cfg=cfg.get("sampler", {"name": "projected_ddim"}),
            warmup_cfg=cfg.get("warmup"),
        ).images

        arrays = [np.array(p) for p in out_pil]
        results_local = [
            {"gid": gid, "fname": f.name, "img": arr.astype("uint8")}
            for gid, f, arr in zip(idxs, batch_files, arrays)
        ]

        results_all = gather_object(results_local)

        if accelerator.is_main_process:
            for r in sorted(results_all, key=lambda z: z["gid"]):
                Image.fromarray(r["img"], mode="RGB").save(output_dir / r["fname"])

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        n_saved = len(list(Path(output_dir).glob("*")))
        print(f"Completed {n_saved} images -> {output_dir}")

    accelerator.end_training()
