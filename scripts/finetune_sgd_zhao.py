#!/usr/bin/env python3
"""SGD vanilla fine-tuning adapter for Zhao et al. 2023 (EDM watermark baseline).

Implements sub-option (i') from Stage 1.5 Q8: a standalone training loop that
loads Zhao's EDMPrecond from pkl, uses EDM's native EDMLoss, and runs exact
500 iterations of SGD(lr=5e-6, momentum=0.9) on clean CIFAR-10. This matches
MiO SGD-FT (scripts/finetune_vanilla.py, 500 iter lr=5e-6 SGD momentum=0.9
on clean CIFAR-10) and WDM SGD-FT (scripts/finetune_sgd_wdm.py, same config)
as closely as Zhao's σ-parameterized EDM permits.

Protocol deviation vs MiO/WDM: the loss is EDM's native σ-weighted MSE on x̂₀
(training.loss.EDMLoss), not standard DDPM ε-prediction MSE — because
EDMPrecond does not output ε. Documented in the paper as a per-method-native
loss choice under a uniform "attacker uses the method's own training objective
with clean (non-watermarked) data at lr=5e-6 for 500 iter" protocol.

Source:
    /data/short/fjiang4/experiments/baseline_comparison/zhao/cifar10/edm/
    00000-images-uncond-ddpmpp-edm-gpus1-batch512-fp32/
    network-snapshot-015053.pkl

Output:
    experiments/baseline_comparison/results/robustness_attacks/cifar10/
    sgd_ft/zhao_sgd_ft_500.pkl
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EDM_DIR = PROJECT_ROOT / "experiments" / "baseline_comparison" / "watermarkdm_repo" / "edm"

# Add EDM's own tree to sys.path BEFORE pickle.load so
# torch_utils.persistence can resolve the custom layer classes.
for p in (str(PROJECT_ROOT), str(EDM_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Now safe to import EDM internals
from torch_utils import persistence  # noqa: F401 — side-effect import for pickle
from training.loss import EDMLoss


DEFAULT_CHECKPOINT = pathlib.Path(
    "/data/short/fjiang4/experiments/baseline_comparison/zhao/cifar10/edm/"
    "00000-images-uncond-ddpmpp-edm-gpus1-batch512-fp32/"
    "network-snapshot-015053.pkl"
)
DEFAULT_OUTPUT = pathlib.Path(
    "experiments/baseline_comparison/results/robustness_attacks/cifar10/"
    "sgd_ft/zhao_sgd_ft_500.pkl"
)

ITERATIONS = 500
BATCH_SIZE = 128
LR = 5e-6
MOMENTUM = 0.9
EMA_DECAY = 0.9999
SEED = 42


class EMA:
    """Deepcopy-based EMA tracker (matches MiO/WDM SGD-FT)."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def to(self, device) -> None:
        self.ema_model.to(device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)


def sha256_first_n_params(net: torch.nn.Module, n: int = 10) -> str:
    """SHA-256 of concatenated float bytes of first n named parameters of net.model (SongUNet), key-sorted.
    Applies to the EDMPrecond: uses net.model.named_parameters() so hashes are
    comparable with the Stage 1 Zhao source hash.
    """
    params = dict(net.model.named_parameters())
    keys = sorted(params.keys())[:n]
    h = hashlib.sha256()
    for k in keys:
        v = params[k]
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_zhao_checkpoint(path: pathlib.Path):
    """Load Zhao EDMPrecond from pkl; returns (net, aux_dict).
    aux_dict retains the other keys from the checkpoint ('loss_fn',
    'augment_pipe', 'dataset_kwargs') for optional use/logging.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    net = data["ema"]
    aux = {k: v for k, v in data.items() if k != "ema"}
    return net, aux


def build_clean_cifar10_dataloader(batch_size: int, num_workers: int = 4) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    root = PROJECT_ROOT / "data" / "cifar-10"
    dataset = datasets.CIFAR10(root=str(root), train=True, download=False, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # --- Load Zhao EDMPrecond + source hash ---
    print(f"[zhao-sgd] Loading Zhao EDMPrecond checkpoint {args.checkpoint}")
    net, aux = load_zhao_checkpoint(args.checkpoint)
    src_sha = sha256_first_n_params(net, n=10)
    print(f"[zhao-sgd] net type = {type(net).__name__}")
    print(f"[zhao-sgd] net.model type = {type(net.model).__name__}")
    print(f"[zhao-sgd] source SHA-256(first 10 SongUNet params) = {src_sha}")
    n_params_total = sum(p.numel() for p in net.parameters())
    print(f"[zhao-sgd] total params = {n_params_total:,}")

    net = net.to(device).train()
    # Ensure all inner params require grad (EDMPrecond's pickled state may
    # have frozen them)
    for p in net.parameters():
        p.requires_grad_(True)

    ema = EMA(net, decay=EMA_DECAY)
    ema.to(device)

    # --- Loss ---
    loss_fn = EDMLoss()  # default P_mean=-1.2, P_std=1.2, sigma_data=0.5
    print(f"[zhao-sgd] Loss: EDMLoss(P_mean={loss_fn.P_mean}, P_std={loss_fn.P_std}, sigma_data={loss_fn.sigma_data})")

    # --- Optimizer ---
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    print(f"[zhao-sgd] Optimizer: SGD(lr={args.lr}, momentum={args.momentum})")

    # --- Data ---
    loader = build_clean_cifar10_dataloader(args.batch_size)
    print(f"[zhao-sgd] Clean CIFAR-10 train samples = {len(loader.dataset):,}, batch={args.batch_size}")
    data_iter = iter(loader)

    # --- Pre-training sanity: single forward pass through loss_fn ---
    real_batch, _ = next(data_iter)
    real_batch = real_batch.to(device)
    with torch.no_grad():
        smoke_loss = loss_fn(net, real_batch, labels=None, augment_pipe=None).mean().item()
    assert np.isfinite(smoke_loss), f"Initial loss is NaN/Inf: {smoke_loss}"
    print(f"[zhao-sgd] pre-training smoke loss = {smoke_loss:.6f} (finite, OK)")
    data_iter = iter(loader)  # reset so step 1 uses fresh batch

    # --- Training loop ---
    initial_loss, last_loss, running_sum = None, None, 0.0
    t0 = time.time()
    for step in range(1, args.iterations + 1):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, _ = next(data_iter)
        images = images.to(device)

        loss = loss_fn(net, images, labels=None, augment_pipe=None).mean()
        assert torch.isfinite(loss).item(), f"NaN loss at step {step}"

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(net)

        lv = loss.item()
        if initial_loss is None:
            initial_loss = lv
        last_loss = lv
        running_sum += lv

        if step == 1 or step % args.log_interval == 0 or step == args.iterations:
            elapsed = time.time() - t0
            running_mean = running_sum / step
            print(f"[zhao-sgd] step={step}/{args.iterations} loss={lv:.6f} mean_so_far={running_mean:.6f} elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[zhao-sgd] done. initial={initial_loss:.6f} final={last_loss:.6f} elapsed={elapsed:.1f}s")

    # Sanity: final loss should be within 10x of initial
    if last_loss > 10 * initial_loss:
        raise RuntimeError(
            f"Final loss ({last_loss:.4f}) > 10x initial ({initial_loss:.4f}); aborting."
        )

    # --- Save EMA net ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_sha = sha256_first_n_params(ema.ema_model, n=10)
    if out_sha == src_sha:
        raise RuntimeError(
            "Output SHA-256 matches source SHA-256; weights were not modified. Aborting save."
        )

    # Move EMA net to CPU for pickle (smaller + no device dependence)
    ema_net_cpu = ema.ema_model.to("cpu")
    save_payload = {
        "ema": ema_net_cpu,
        "training_config": {
            "source_checkpoint": str(args.checkpoint),
            "source_sha256_first10": src_sha,
            "output_sha256_first10": out_sha,
            "protocol": "sgd-ft / EDMLoss (native σ-weighted) / clean CIFAR-10",
            "iterations": args.iterations,
            "lr": args.lr,
            "momentum": args.momentum,
            "batch_size": args.batch_size,
            "ema_decay": EMA_DECAY,
            "seed": args.seed,
            "loss_init": initial_loss,
            "loss_final": last_loss,
            "loss_mean_all_iters": running_sum / args.iterations,
            "wall_time_sec": elapsed,
        },
    }
    with open(args.output, "wb") as f:
        pickle.dump(save_payload, f)
    print(f"[zhao-sgd] saved → {args.output}")
    print(f"[zhao-sgd] output SHA-256(first 10 SongUNet params) = {out_sha}")


if __name__ == "__main__":
    main()
