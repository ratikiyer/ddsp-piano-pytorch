from __future__ import annotations

import argparse
import time

import torch

from ddsp_piano_pytorch.modules.losses import SpectralLoss, SpectralLossConfig
from ddsp_piano_pytorch.modules.piano_model import PianoModel


def run_benchmark(steps: int = 20, warmup: int = 5, device: str = "cpu") -> dict[str, float]:
    dev = torch.device(device)
    model = PianoModel(
        n_synths=16,
        n_harmonics=32,
        n_noise_bands=33,
        sample_rate=8000,
        frame_rate=100,
        context_dim=32,
    ).to(dev)
    loss_fn = SpectralLoss(SpectralLossConfig(fft_sizes=(512, 256), overlap=0.75)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    b, t, n = 2, 30, 16
    conditioning = torch.zeros(b, t, n, 2, device=dev)
    conditioning[:, :15, :, 0] = 60.0
    conditioning[:, :15, :, 1] = 0.8
    pedals = torch.zeros(b, t, 4, device=dev)
    piano_model = torch.zeros(b, dtype=torch.long, device=dev)
    target = torch.rand(b, int(8000 * (t / 100.0)), device=dev)

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        out = model(conditioning=conditioning, pedals=pedals, piano_model=piano_model)
        loss = loss_fn(target, out["audio"])
        loss.backward()
        optimizer.step()

    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(conditioning=conditioning, pedals=pedals, piano_model=piano_model)
        loss = loss_fn(target, out["audio"])
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    step_time = elapsed / float(steps)
    return {
        "steps": float(steps),
        "elapsed_sec": elapsed,
        "step_time_sec": step_time,
        "steps_per_sec": 1.0 / max(step_time, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DDSP-Piano PyTorch forward/backward throughput.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    stats = run_benchmark(steps=args.steps, warmup=args.warmup, device=args.device)
    print(stats)


if __name__ == "__main__":
    main()
