"""Differentiable 10-step DDIM sampler with gradient checkpointing and debug instrumentation."""

from __future__ import annotations

import pathlib
from typing import Dict, Iterable, List, Sequence
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint

from mia_logging import get_winston_logger


LOGGER = get_winston_logger(__name__)
EPS = 1e-12
SCALE_THRESHOLD = 50.0


def _tensor_debug_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Return nan-safe stats for logging."""

    with torch.no_grad():
        t = tensor.detach().float()
        if t.numel() == 0:
            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "norm": float("nan"),
                "absmax": float("nan"),
            }

        finite = t[torch.isfinite(t)]
        if finite.numel() == 0:
            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "norm": float("nan"),
                "absmax": float("nan"),
            }

        if t.dim() >= 1:
            flat = t.view(t.shape[0], -1)
            norm = torch.linalg.norm(flat, dim=1).mean() if flat.numel() > 0 else torch.tensor(float("nan"), device=t.device)
        else:
            flat = t.view(1, -1)
            norm = torch.linalg.norm(flat)

        return {
            "min": finite.min().item(),
            "max": finite.max().item(),
            "mean": finite.mean().item(),
            "std": finite.std().item(),
            "norm": norm.item(),
            "absmax": finite.abs().max().item(),
        }


def _small_slice(tensor: torch.Tensor) -> torch.Tensor:
    """Extract a compact slice for debug dumping."""

    if tensor.numel() == 0:
        return tensor.detach().cpu()
    with torch.no_grad():
        if tensor.dim() >= 4:
            return tensor.detach().cpu()[0, 0, :4, :4].clone()
        if tensor.dim() >= 2:
            return tensor.detach().cpu()[0, : min(4, tensor.shape[1])].clone()
        return tensor.detach().cpu().flatten()[:8].clone()


def build_linear_timesteps(T: int, K: int = 10, start: int | None = None) -> List[int]:
    """Return a descending list of K+1 timesteps spaced linearly in [start, 0].

    Ensures:
        - First timestep is start (default T-1).
        - Last timestep is 0.
        - Sequence is strictly decreasing (duplicates removed).
        - Length is exactly K+1 (inserts extra valid steps if rounding creates duplicates).
    """

    if K <= 0:
        raise ValueError("K must be positive")
    if T < 2:
        raise ValueError("T must be at least 2 for sampling")
    start_t = T - 1 if start is None else int(start)
    if start_t < 0 or start_t > T - 1:
        raise ValueError(f"start must be in [0, T-1], got {start_t}")
    if K + 1 > start_t + 1:
        raise ValueError(f"Requested {K + 1} timesteps but only {start_t + 1} unique steps are available between start and 0")

    float_points = torch.linspace(start_t, 0, steps=K + 1, dtype=torch.float64)
    rounded = torch.round(float_points).to(torch.long)

    timesteps: List[int] = []
    for val in rounded.tolist():
        step = int(val)
        step = min(max(step, 0), T - 1)
        if not timesteps:
            timesteps.append(step)
            continue
        if step == timesteps[-1]:
            continue
        if step > timesteps[-1]:
            step = max(timesteps[-1] - 1, 0)
            if step == timesteps[-1]:
                continue
        timesteps.append(step)

    if timesteps[0] != start_t:
        timesteps[0] = start_t

    # Enforce strict decrease and append 0 endpoint.
    cleaned: List[int] = [timesteps[0]]
    for step in timesteps[1:]:
        candidate = step
        if candidate >= cleaned[-1]:
            candidate = cleaned[-1] - 1
        candidate = max(candidate, 0)
        if candidate == cleaned[-1]:
            continue
        cleaned.append(candidate)

    if cleaned[-1] != 0:
        cleaned.append(0)

    if len(cleaned) < K + 1:
        next_step = cleaned[-1] - 1
        while len(cleaned) < K + 1 and next_step >= 0:
            if next_step < cleaned[-1]:
                cleaned.append(next_step)
            next_step -= 1
        if cleaned[-1] != 0:
            cleaned.append(0)

    if len(cleaned) < K + 1:
        raise ValueError(f"Unable to build strictly decreasing schedule of length {K + 1} with T={T}")

    return cleaned[: K + 1]


def _model_forward(model, x_t: torch.Tensor, t: int, use_checkpoint: bool) -> torch.Tensor:
    """Model wrapper that optionally applies gradient checkpointing."""

    def fn(input_tensor: torch.Tensor) -> torch.Tensor:
        t_batch = torch.full((input_tensor.shape[0],), t, device=input_tensor.device, dtype=torch.long)
        return model(input_tensor, t_batch)

    if use_checkpoint:
        return checkpoint(fn, x_t, use_reentrant=False)
    return fn(x_t)


def _prepare_timesteps(
    timesteps: Iterable[int],
    total_T: int,
    debug: bool = False,
    start: int | None = None,
) -> List[int]:
    """Validate and clean timesteps, enforcing descending order and endpoint at 0."""

    start_t = total_T - 1 if start is None else int(start)
    t_raw = [int(t) for t in timesteps]
    if len(t_raw) == 0:
        raise ValueError("timesteps must be non-empty")

    if t_raw[-1] != 0:
        t_raw.append(0)

    cleaned: List[int] = []
    for t in t_raw:
        if t < 0 or t > total_T - 1:
            raise ValueError(f"timestep {t} is out of bounds for T={total_T}")
        if cleaned and t > cleaned[-1]:
            raise ValueError(f"Timesteps must be non-increasing (got {cleaned[-1]} -> {t})")
        if cleaned and t == cleaned[-1]:
            continue
        cleaned.append(t)

    if cleaned[0] != start_t:
        raise ValueError(f"First timestep must be start={start_t}, got {cleaned[0]}")
    if cleaned[-1] != 0:
        cleaned.append(0)

    for idx, (t, t_prev) in enumerate(zip(cleaned[:-1], cleaned[1:])):
        if not (0 <= t_prev < t <= total_T - 1):
            raise ValueError(f"Invalid timestep pair at index {idx}: t={t}, t_prev={t_prev}")

    if debug:
        LOGGER.info("DDIM timesteps (len=%d): %s", len(cleaned), cleaned)
    return cleaned


def _dump_debug_bundle(
    stage: str,
    step_idx: int,
    t: int,
    t_prev: int,
    a_t: torch.Tensor,
    a_prev: torch.Tensor,
    debug_dir: pathlib.Path | None,
    x_t: torch.Tensor | None = None,
    eps: torch.Tensor | None = None,
    x0_hat: torch.Tensor | None = None,
    x_prev: torch.Tensor | None = None,
    eps_dir: torch.Tensor | None = None,
    extra: Dict[str, object] | None = None,
) -> pathlib.Path | None:
    """Persist a compact debug artifact on first failure."""

    if debug_dir is None:
        return None

    debug_dir.mkdir(parents=True, exist_ok=True)
    suffix = "nan" if "nan" in stage else stage
    path = debug_dir / f"ddim_{suffix}_step{step_idx}.pt"

    payload: Dict[str, object] = {
        "stage": stage,
        "step_idx": step_idx,
        "t": t,
        "t_prev": t_prev,
        "alpha_t": float(a_t),
        "alpha_prev": float(a_prev),
        "sqrt_alpha_t": float(torch.sqrt(a_t.clamp_min(EPS))),
        "sqrt_one_minus_t": float(torch.sqrt(torch.clamp(1 - a_t, min=0.0))),
        "sqrt_alpha_prev": float(torch.sqrt(a_prev.clamp_min(EPS))),
        "sqrt_one_minus_prev": float(torch.sqrt(torch.clamp(1 - a_prev, min=0.0))),
    }

    if x_t is not None:
        payload["x_t_stats"] = _tensor_debug_stats(x_t)
        payload["x_t_slice"] = _small_slice(x_t)
    if eps is not None:
        payload["eps_stats"] = _tensor_debug_stats(eps)
        payload["eps_slice"] = _small_slice(eps)
    if x0_hat is not None:
        payload["x0_hat_stats"] = _tensor_debug_stats(x0_hat)
        payload["x0_hat_slice"] = _small_slice(x0_hat)
    if x_prev is not None:
        payload["x_prev_stats"] = _tensor_debug_stats(x_prev)
        payload["x_prev_slice"] = _small_slice(x_prev)
    if eps_dir is not None:
        payload["eps_dir_stats"] = _tensor_debug_stats(eps_dir)
        payload["eps_dir_slice"] = _small_slice(eps_dir)
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    return path


def ddim_step(
    model,
    x_t: torch.Tensor,
    t: int,
    t_prev: int,
    alphas_bar: torch.Tensor,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Single deterministic DDIM step (η=0) that is autograd-friendly."""

    alpha_bar_t = alphas_bar[t].clamp(min=EPS, max=1.0)
    sqrt_alpha = alpha_bar_t.sqrt()
    sqrt_one_minus = (1.0 - alpha_bar_t).clamp(min=0.0).sqrt()

    eps = _model_forward(model, x_t, t, use_checkpoint)
    x0_hat = (x_t - sqrt_one_minus * eps) / sqrt_alpha

    if t_prev < 0 or t_prev > t:
        raise ValueError(f"t_prev must satisfy 0 <= t_prev <= t (got {t_prev}, {t})")
    if t_prev == 0:
        return x0_hat

    alpha_bar_prev = alphas_bar[t_prev].clamp(min=EPS, max=1.0)
    sqrt_alpha_prev = alpha_bar_prev.sqrt()
    sqrt_one_minus_prev = (1.0 - alpha_bar_prev).clamp(min=0.0).sqrt()
    x_prev = sqrt_alpha_prev * x0_hat + sqrt_one_minus_prev * eps
    return x_prev


def ddim_sample_differentiable(
    model,
    alphas_bar: torch.Tensor,
    shape: Sequence[int] | torch.Size,
    timesteps: Iterable[int],
    device: torch.device,
    use_checkpoint: bool = True,
    noise: torch.Tensor | None = None,
    debug_ddim: bool = False,
    debug_dir: pathlib.Path | None = None,
    ddim_fp32: bool = True,
    debug_scale: bool = False,
    scale_threshold: float = SCALE_THRESHOLD,
    fail_on_explode: bool = False,
) -> torch.Tensor:
    """Run a differentiable DDIM chain over the provided timesteps.

    Args:
        model: ε-prediction model.
        alphas_bar: Tensor of cumulative products `[T]`.
        shape: Output tensor shape, e.g., (B, 3, 32, 32).
        timesteps: Descending iterable of integers (length K+1 with endpoint 0).
        device: Compute device.
        use_checkpoint: Whether to wrap model forward with checkpointing.
        noise: Optional initial x_T. If None, draws N(0, I).
        debug_ddim: Enable per-step validation/logging and debug dumps.
        debug_dir: Directory to store debug artifacts when debug_ddim or debug_scale is True.
        ddim_fp32: Force DDIM algebra (and model forward) to run in float32.
        debug_scale: Enable per-step scale monitoring and dumps if exceeded.
        scale_threshold: Absolute-value threshold to trigger a debug dump.
        fail_on_explode: Raise if |x0| exceeds threshold at any step.

    Returns:
        x0 tensor (same shape) suitable for downstream losses.
    """

    x_t = noise if noise is not None else torch.randn(shape, device=device)
    x_t = x_t.to(device)
    if ddim_fp32:
        x_t = x_t.float()
    if use_checkpoint and not x_t.requires_grad:
        # checkpoint requires at least one input with grad to build graph
        x_t.requires_grad_(True)

    total_T = alphas_bar.shape[0]
    if total_T < 2:
        raise ValueError("alphas_bar must have length >= 2")
    if not torch.isfinite(alphas_bar).all():
        raise RuntimeError("alphas_bar contains non-finite values")
    if alphas_bar[0] < 0.9 or alphas_bar[0] > 1.1:
        LOGGER.warning("alphas_bar[0]=%.4e is unexpected (expected near 1.0)", alphas_bar[0].item())
    if alphas_bar[-1] <= 0:
        LOGGER.warning("alphas_bar[T-1]=%.4e is non-positive", alphas_bar[-1].item())

    timesteps_list = list(timesteps)
    start_t = timesteps_list[0] if timesteps_list else None
    t_list = _prepare_timesteps(timesteps_list, total_T, debug=debug_ddim, start=start_t)
    if len(t_list) < 2:
        raise ValueError("timesteps must contain at least two entries (including 0 endpoint)")

    alphas_bar = alphas_bar.to(device)
    if ddim_fp32:
        alphas_bar = alphas_bar.float()
    alphas_bar = alphas_bar.clamp(min=0.0, max=1.0)

    if x_t.is_cuda:
        def autocast_ctx(enabled: bool = True):  # type: ignore[override]
            return torch.amp.autocast(device_type="cuda", enabled=enabled)
    else:
        def autocast_ctx(enabled: bool = True):  # type: ignore[override]
            return nullcontext()

    autocast_enabled = x_t.is_cuda and (not debug_ddim) and (not ddim_fp32)

    alpha_schedule = [float(alphas_bar[int(t)].detach().cpu()) for t in t_list]
    LOGGER.info("DDIM timesteps (len=%d): %s", len(t_list), t_list)
    if len(t_list) != 11:
        LOGGER.warning("DDIM timesteps length is %d (expected 11 for K=10)", len(t_list))
    LOGGER.info(
        "alpha_bar schedule snippet: first=%.4e last=%.4e mid=%.4e",
        alpha_schedule[0],
        alpha_schedule[-1],
        alpha_schedule[len(alpha_schedule) // 2],
    )

    for idx, (t, t_prev) in enumerate(zip(t_list[:-1], t_list[1:])):
        a_t = alphas_bar[int(t)]
        a_prev = alphas_bar[int(t_prev)]

        sqrt_a_t_raw = torch.sqrt(torch.clamp(a_t, min=0.0))
        sqrt_one_minus_raw = torch.sqrt(torch.clamp(1 - a_t, min=0.0))
        sqrt_a_prev_raw = torch.sqrt(torch.clamp(a_prev, min=0.0))
        sqrt_one_minus_prev_raw = torch.sqrt(torch.clamp(1 - a_prev, min=0.0))

        sqrt_a_t = sqrt_a_t_raw.clamp_min(1e-4)
        sqrt_one_minus = sqrt_one_minus_raw.clamp_min(1e-4)
        sqrt_a_prev = sqrt_a_prev_raw.clamp_min(1e-4)
        sqrt_one_minus_prev = sqrt_one_minus_prev_raw.clamp_min(1e-4)

        if debug_ddim or debug_scale:
            LOGGER.info(
                "DDIM step %d: t=%d -> t_prev=%d | a_t=%.4e a_prev=%.4e sqrt(a_t)=%.4e sqrt(1-a_t)=%.4e sqrt(a_prev)=%.4e sqrt(1-a_prev)=%.4e",
                idx,
                t,
                t_prev,
                a_t.item(),
                a_prev.item(),
                sqrt_a_t.item(),
                sqrt_one_minus.item(),
                sqrt_a_prev.item(),
                sqrt_one_minus_prev.item(),
            )

        if not (0.0 < a_t <= 1.0) or not (0.0 < a_prev <= 1.0):
            raise RuntimeError(
                f"alpha_bar out of bounds at step {idx}: "
                f"a_t={a_t.item():.4e}, a_prev={a_prev.item():.4e} (t={t}, t_prev={t_prev})"
            )
        if torch.any(torch.isnan(a_t)) or torch.any(torch.isnan(a_prev)):
            raise RuntimeError(f"alpha_bar contains NaN at step {idx} (t={t}, t_prev={t_prev})")

        if debug_ddim or debug_scale:
            LOGGER.info("x_t stats: %s", _tensor_debug_stats(x_t))

        if not torch.isfinite(x_t).all():
            path = _dump_debug_bundle(
                "x_t",
                idx,
                t,
                t_prev,
                a_t,
                a_prev,
                debug_dir,
                x_t=x_t,
                extra={"alpha_schedule": alpha_schedule},
            )
            summary = (
                f"NaN/Inf detected in x_t at step {idx} (t={t}, t_prev={t_prev}); "
                f"alpha_t={a_t.item():.4e}, alpha_prev={a_prev.item():.4e}"
            )
            if path:
                summary += f" debug={path}"
            raise RuntimeError(summary)

        if debug_ddim:
            with torch.no_grad():
                with autocast_ctx(enabled=False):
                    eps_fp32 = _model_forward(model, x_t.float(), int(t), use_checkpoint=False).float()
                LOGGER.info("eps(fp32,no_grad) stats: %s", _tensor_debug_stats(eps_fp32))
                if not torch.isfinite(eps_fp32).all():
                    path = _dump_debug_bundle(
                        "eps_fp32",
                        idx,
                        t,
                        t_prev,
                        a_t,
                        a_prev,
                        debug_dir,
                        x_t=x_t,
                        eps=eps_fp32,
                        extra={"alpha_schedule": alpha_schedule},
                    )
                    msg = (
                        f"NaN/Inf in eps (fp32 debug) at step {idx} (t={t}, t_prev={t_prev}); "
                        f"alpha_t={a_t.item():.4e}, alpha_prev={a_prev.item():.4e}"
                    )
                    if path:
                        msg += f" debug={path}"
                    raise RuntimeError(msg)

        with autocast_ctx(enabled=autocast_enabled):
            eps = _model_forward(model, x_t if not ddim_fp32 else x_t.float(), int(t), use_checkpoint)
        eps = eps.float()

        if debug_ddim or debug_scale:
            LOGGER.info("eps stats: %s", _tensor_debug_stats(eps))

        if not torch.isfinite(eps).all():
            path = _dump_debug_bundle(
                "eps",
                idx,
                t,
                t_prev,
                a_t,
                a_prev,
                debug_dir,
                x_t=x_t,
                eps=eps,
                extra={"alpha_schedule": alpha_schedule},
            )
            summary = (
                f"NaN/Inf detected in eps at step {idx} (t={t}, t_prev={t_prev}); "
                f"alpha_t={a_t.item():.4e}, alpha_prev={a_prev.item():.4e}"
            )
            if path:
                summary += f" debug={path}"
            raise RuntimeError(summary)

        with autocast_ctx(enabled=autocast_enabled):
            numerator = x_t.float() - sqrt_one_minus * eps
            x0_hat = numerator / sqrt_a_t

            eps_dir = (x_t.float() - sqrt_a_t * x0_hat) / sqrt_one_minus

            if debug_ddim or debug_scale:
                LOGGER.info("x0_hat stats: %s", _tensor_debug_stats(x0_hat))
                LOGGER.info("eps_dir stats: %s", _tensor_debug_stats(eps_dir))
                LOGGER.info("eps_dir_vs_eps_mean_abs_diff=%.4e", (eps_dir - eps).abs().mean().item())

            if not torch.isfinite(x0_hat).all():
                path = _dump_debug_bundle(
                    "x0_hat",
                    idx,
                    t,
                    t_prev,
                    a_t,
                    a_prev,
                    debug_dir,
                    x_t=x_t,
                    eps=eps,
                    x0_hat=x0_hat,
                    eps_dir=eps_dir,
                    extra={"alpha_schedule": alpha_schedule},
                )
                summary = (
                    f"NaN/Inf detected in x0_hat at step {idx} (t={t}, t_prev={t_prev}); "
                    f"alpha_t={a_t.item():.4e}, alpha_prev={a_prev.item():.4e}"
                )
                if path:
                    summary += f" debug={path}"
                raise RuntimeError(summary)

            x_prev = sqrt_a_prev * x0_hat + sqrt_one_minus_prev * eps_dir

        if debug_ddim or debug_scale:
            LOGGER.info("x_prev stats: %s", _tensor_debug_stats(x_prev))

        # Scale invariant checks
        if debug_scale or fail_on_explode:
            absmax_x0 = x0_hat.detach().abs().max().item()
            absmax_prev = x_prev.detach().abs().max().item()
            if absmax_x0 > scale_threshold or absmax_prev > scale_threshold:
                path = _dump_debug_bundle(
                    "scale",
                    idx,
                    t,
                    t_prev,
                    a_t,
                    a_prev,
                    debug_dir,
                    x_t=x_t,
                    eps=eps,
                    x0_hat=x0_hat,
                    x_prev=x_prev,
                    eps_dir=eps_dir,
                    extra={
                        "alpha_schedule": alpha_schedule,
                        "absmax_x0": absmax_x0,
                        "absmax_prev": absmax_prev,
                    },
                )
                msg = (
                    f"Scale explosion at step {idx} (t={t}, t_prev={t_prev}): "
                    f"x0_hat_absmax={absmax_x0:.2f}, x_prev_absmax={absmax_prev:.2f}"
                )
                if path:
                    msg += f" debug={path}"
                if fail_on_explode:
                    raise RuntimeError(msg)
                else:
                    LOGGER.warning(msg)

        if not torch.isfinite(x_prev).all():
            path = _dump_debug_bundle(
                "x_prev",
                idx,
                t,
                t_prev,
                a_t,
                a_prev,
                debug_dir,
                x_t=x_t,
                eps=eps,
                x0_hat=x0_hat,
                x_prev=x_prev,
                eps_dir=eps_dir,
                extra={"alpha_schedule": alpha_schedule},
            )
            summary = (
                f"NaN/Inf detected in x_prev at step {idx} (t={t}, t_prev={t_prev}); "
                f"alpha_t={a_t.item():.4e}, alpha_prev={a_prev.item():.4e}"
            )
            if path:
                summary += f" debug={path}"
            raise RuntimeError(summary)

        x_t = x_prev

    if debug_scale or fail_on_explode:
        absmax_final = x_t.detach().abs().max().item()
        if absmax_final > scale_threshold:
            path = _dump_debug_bundle(
                "x0_final",
                len(t_list) - 1,
                t_list[-2] if len(t_list) >= 2 else -1,
                0,
                alphas_bar[int(t_list[-2])] if len(t_list) >= 2 else torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                debug_dir,
                x_prev=x_t,
                extra={"alpha_schedule": alpha_schedule, "absmax_final": absmax_final},
            )
            msg = f"Final x0 absmax={absmax_final:.2f} exceeds threshold {scale_threshold}"
            if path:
                msg += f" debug={path}"
            if fail_on_explode:
                raise RuntimeError(msg)
            else:
                LOGGER.warning(msg)

    return x_t


__all__ = ["build_linear_timesteps", "ddim_sample_differentiable", "ddim_step"]
