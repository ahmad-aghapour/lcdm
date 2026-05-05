from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DDIMScheduler

from lcdm.samplers.warmup import build_warmup_fn, randn_like
from lcdm.utils import denormalize_images_from_model


class ProjectedDDIMPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler: DDIMScheduler):
        super().__init__()
        scheduler = DDIMScheduler.from_config(scheduler.config)
        self.register_modules(unet=unet, scheduler=scheduler)

    def _model_device(self):
        try:
            return next(self.unet.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _predict_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.unet(x, t)

        if hasattr(out, "sample"):
            out = out.sample

        if isinstance(out, tuple):
            out = out[0]

        if out.size(1) == 6:
            out = out[:, :3]

        return out

    def _default_warmup_cfg(self):
        return {
            "name": "underdamped",
            "splitting": "baoab",
            "start_fraction": 0.5,
            "n_steps": 50,
            "friction": 0.5,
            "h_mul": 1.0,
            "init_noise_std": 0.5,
        }

    @torch.no_grad()
    def __call__(
        self,
        observation: torch.Tensor,
        operator,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        sampler_cfg: Optional[dict] = None,
        warmup_cfg: Optional[dict] = None,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        device = self._model_device()

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generator list length {len(generator)} does not match batch size {batch_size}."
            )

        if sampler_cfg is None:
            sampler_cfg = {"name": "projected_ddim"}

        self.scheduler.set_timesteps(num_inference_steps)

        observation = observation.to(device)

        # For inpainting: identity
        # For super-resolution: LR -> HR lift
        y_init = operator.lift_observation(observation).to(device)

        sampler_name = sampler_cfg.get("name", "projected_ddim")

        if sampler_name == "projected_ddim":
            if warmup_cfg is None:
                warmup_cfg = self._default_warmup_cfg()

            image, time_start = self._init_with_warmup(
                y_init=y_init,
                operator=operator,
                generator=generator,
                warmup_cfg=warmup_cfg,
            )
        elif sampler_name == "ddnm":
            image, time_start = self._init_from_noise(
                y_init=y_init,
                generator=generator,
            )
        else:
            raise ValueError(f"Unknown sampler name: {sampler_name}")

        for t in self.progress_bar(self.scheduler.timesteps):
            if t > time_start:
                continue

            time = torch.ones(batch_size, device=device) * t
            model_output = self._predict_noise(image, time)

            sqrt_alpha_prod = (self.scheduler.alphas_cumprod[t] ** 0.5).to(device)
            sigma_t = ((1.0 - self.scheduler.alphas_cumprod[t]) ** 0.5).to(device)

            epsilon = (
                operator.apply_pk(model_output)
                - operator.apply_pn(sqrt_alpha_prod * y_init - image) / sigma_t
            )

            image = self.scheduler.step(
                epsilon,
                t,
                image,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
            ).prev_sample

        image = denormalize_images_from_model(image)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def _init_with_warmup(self, y_init, operator, generator, warmup_cfg):
        device = y_init.device

        start_fraction = float(warmup_cfg.get("start_fraction", 0.5))
        start_idx = int(len(self.scheduler.timesteps) * start_fraction)
        start_idx = min(max(start_idx, 0), len(self.scheduler.timesteps) - 1)
        time_start = self.scheduler.timesteps[start_idx]

        sigma_start = ((1.0 - self.scheduler.alphas_cumprod[time_start]) ** 0.5).to(device)

        init_noise_std = float(warmup_cfg.get("init_noise_std", 0.5))
        eps_obs = randn_like(y_init, generator=generator) * init_noise_std

        y_init_langevin = operator.apply_pn(y_init) + operator.apply_pk(eps_obs)
        y_init_post = y_init_langevin * self.scheduler.alphas_cumprod[time_start] ** 0.5

        warmup_fn = build_warmup_fn(
            warmup_cfg.get("name", "underdamped"),
            splitting=warmup_cfg.get("splitting", "baoab"),
        )

        image = warmup_fn(
            y_init_i=y_init_post,
            apply_pk=operator.apply_pk,
            score_fn=lambda x, t: -self._predict_noise(x, t) / sigma_start,
            alpha_cumprod=self.scheduler.alphas_cumprod,
            time=time_start,
            n_steps=int(warmup_cfg.get("n_steps", 50)),
            eta_mul=float(warmup_cfg.get("h_mul", 1.0)),
            h_mul=float(warmup_cfg.get("h_mul", 1.0)),
            friction=float(warmup_cfg.get("friction", 0.5)),
            generator=generator,
        )

        return image, time_start

    def _init_from_noise(self, y_init, generator):
        device = y_init.device
        image = randn_like(y_init, generator=generator).to(device)
        time_start = self.scheduler.timesteps[0]
        return image, time_start
