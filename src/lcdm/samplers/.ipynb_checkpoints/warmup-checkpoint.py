from __future__ import annotations

import math
from typing import Callable

import torch


def randn_like(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(
        x.shape,
        dtype=x.dtype,
        device=x.device,
        generator=generator,
    )


@torch.no_grad()
def projector_langevin_warmup(
    *,
    y_init_i: torch.Tensor,
    apply_pk: Callable[[torch.Tensor], torch.Tensor],
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    alpha_cumprod: torch.Tensor,
    time: int,
    n_steps: int = 50,
    eta_mul: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    sigma0 = (1.0 - alpha_cumprod[time]).sqrt().to(y_init_i.device)

    eps_obs = randn_like(y_init_i, generator=generator) * sigma0
    x = y_init_i + eps_obs

    eta = eta_mul * sigma0
    gamma = eta * eta
    noise_coef = eta

    for _ in range(n_steps):
        t_batch = torch.full((x.shape[0],), time, device=x.device)
        score_full = score_fn(x, t_batch)
        score_null = apply_pk(score_full)

        eps = randn_like(x, generator=generator)
        x = x + gamma * score_null + apply_pk(noise_coef * eps)

    return x


@torch.no_grad()
def projector_underdamped_warmup(
    *,
    y_init_i: torch.Tensor,
    apply_pk: Callable[[torch.Tensor], torch.Tensor],
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    alpha_cumprod: torch.Tensor,
    time: int,
    n_steps: int = 50,
    h_mul: float = 1.0,
    friction: float = 0.5,
    freeze_rowspace: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    BAOAB-style underdamped warmup in the null-space defined by apply_pk.
    """
    sigma0 = (1.0 - alpha_cumprod[time]).sqrt().to(y_init_i.device)

    h = h_mul * sigma0
    gamma = float(friction)

    eps_obs = randn_like(y_init_i, generator=generator) * sigma0
    init = y_init_i + eps_obs

    x = init.clone()
    v = apply_pk(torch.zeros_like(x))

    batch_size = x.shape[0]
    t_batch = torch.full((batch_size,), time, device=x.device)

    alpha_full = math.exp(-gamma * float(h))
    ou_std_full = (1.0 - alpha_full**2) ** 0.5

    grad = apply_pk(score_fn(x, t_batch))

    for j in range(n_steps):
        if j > max(0, n_steps - 10):
            h_local = min(float(sigma0), float(h))
            alpha_local = math.exp(-0.5 * h_local)
            ou_std_local = (1.0 - alpha_local**2) ** 0.5
        else:
            h_local = float(h)
            alpha_local = alpha_full
            ou_std_local = ou_std_full

        # B: half kick
        v = v + 0.5 * h_local * grad

        # A: half drift
        x = x + 0.5 * h_local * v
        if freeze_rowspace:
            x = init + apply_pk(x - init)

        # O: full OU
        eps = randn_like(v, generator=generator)
        v = alpha_local * v + ou_std_local * apply_pk(eps)

        # A: half drift
        x = x + 0.5 * h_local * v
        if freeze_rowspace:
            x = init + apply_pk(x - init)

        # B: half kick
        grad = apply_pk(score_fn(x, t_batch))
        v = v + 0.5 * h_local * grad

    return x


def build_warmup_fn(name: str):
    if name == "langevin":
        return projector_langevin_warmup
    if name == "underdamped":
        return projector_underdamped_warmup
    raise ValueError(f"Unknown warmup name: {name}")
