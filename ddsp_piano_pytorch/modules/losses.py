from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ddsp_piano_pytorch.core import multiscale_fft, safe_log


@dataclass
class SpectralLossConfig:
    fft_sizes: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)
    overlap: float = 0.75
    mag_weight: float = 1.0
    logmag_weight: float = 1.0
    cumulative_sums: int = 0


class SpectralLoss(nn.Module):
    def __init__(self, config: SpectralLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpectralLossConfig()

    def forward(self, target_audio: torch.Tensor, synth_audio: torch.Tensor) -> torch.Tensor:
        target_specs = multiscale_fft(target_audio, self.config.fft_sizes, overlap=self.config.overlap)
        synth_specs = multiscale_fft(synth_audio, self.config.fft_sizes, overlap=self.config.overlap)

        loss = target_audio.new_tensor(0.0)
        for target, synth in zip(target_specs, synth_specs):
            cur_t = target
            cur_s = synth
            for _ in range(max(0, self.config.cumulative_sums)):
                cur_t = torch.cumsum(cur_t, dim=-1)
                cur_s = torch.cumsum(cur_s, dim=-1)
            if self.config.mag_weight > 0:
                loss = loss + self.config.mag_weight * torch.mean(torch.abs(cur_t - cur_s))
            if self.config.logmag_weight > 0:
                loss = loss + self.config.logmag_weight * torch.mean(torch.abs(safe_log(cur_t) - safe_log(cur_s)))
        return loss


class ReverbRegularizer(nn.Module):
    def __init__(self, weight: float = 0.01, loss_type: str = "L1") -> None:
        super().__init__()
        self.weight = weight
        self.loss_type = loss_type.upper()

    def forward(self, reverb_ir: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "L2":
            loss = torch.sum(reverb_ir.square())
        else:
            loss = torch.sum(torch.abs(reverb_ir))
        return self.weight * loss / reverb_ir.shape[0]


class InharmonicityLoss(nn.Module):
    def __init__(self, weight: float = 10.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, inharm_coef: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(torch.clamp(-inharm_coef, min=0.0))
        return self.weight * loss / inharm_coef.shape[0]
