from __future__ import annotations

import torch
import torch.nn as nn

from ddsp_piano_pytorch.core import (
    get_inharmonic_frequencies,
    remove_above_nyquist,
    safe_divide,
    scale_function,
    upsample,
)


class InHarmonic(nn.Module):
    """Inharmonic additive synthesizer."""

    def __init__(
        self,
        frame_rate: int = 250,
        sample_rate: int = 24000,
        min_frequency: float = 20.0,
        normalize_after_nyquist_cut: bool = True,
        normalize_below_nyquist: bool = True,
    ) -> None:
        super().__init__()
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.min_frequency = min_frequency
        self.normalize_after_nyquist_cut = normalize_after_nyquist_cut
        self.normalize_below_nyquist = normalize_below_nyquist

    @property
    def upsampling(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    def get_controls(
        self,
        amplitudes: torch.Tensor,
        harmonic_distribution: torch.Tensor,
        inharm_coef: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inharm_coef = torch.clamp(inharm_coef, min=0.0)
        amplitudes = scale_function(amplitudes)
        harmonic_distribution = scale_function(harmonic_distribution)

        n_harmonics = harmonic_distribution.shape[-1]
        inharmonic_freq = get_inharmonic_frequencies(f0_hz, inharm_coef, n_harmonics)
        harmonic_shifts = safe_divide(inharmonic_freq, torch.clamp(f0_hz, min=1e-4) * torch.arange(
            1, n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype
        ).view(1, 1, -1)) - 1.0

        if not self.normalize_after_nyquist_cut:
            harmonic_distribution = safe_divide(harmonic_distribution, harmonic_distribution.sum(dim=-1, keepdim=True))

        if self.normalize_below_nyquist:
            harmonic_distribution = remove_above_nyquist(harmonic_distribution, inharmonic_freq, self.sample_rate)

        amplitudes = amplitudes * (f0_hz > self.min_frequency).to(amplitudes.dtype)
        if self.normalize_after_nyquist_cut:
            harmonic_distribution = safe_divide(harmonic_distribution, harmonic_distribution.sum(dim=-1, keepdim=True))

        return {
            "amplitudes": amplitudes,
            "harmonic_distribution": harmonic_distribution,
            "harmonic_shifts": harmonic_shifts,
            "f0_hz": f0_hz,
        }

    def get_signal(
        self,
        amplitudes: torch.Tensor,
        harmonic_distribution: torch.Tensor,
        harmonic_shifts: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> torch.Tensor:
        n_harmonics = harmonic_distribution.shape[-1]
        harmonics = torch.arange(1, n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype).view(1, 1, -1)
        harmonic_frequencies = f0_hz * harmonics * (1.0 + harmonic_shifts)
        harmonic_amplitudes = amplitudes * harmonic_distribution
        n_samples = self.upsampling * f0_hz.shape[1]
        frequency_envelopes = upsample(harmonic_frequencies, self.upsampling)[:, :n_samples]
        amplitude_envelopes = upsample(harmonic_amplitudes, self.upsampling)[:, :n_samples]
        phases = torch.cumsum(2.0 * torch.pi * frequency_envelopes / float(self.sample_rate), dim=1)
        audio = (torch.cos(phases) * remove_above_nyquist(amplitude_envelopes, frequency_envelopes, self.sample_rate)).sum(dim=-1)
        return audio

    def forward(
        self,
        amplitudes: torch.Tensor,
        harmonic_distribution: torch.Tensor,
        inharm_coef: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> torch.Tensor:
        controls = self.get_controls(amplitudes, harmonic_distribution, inharm_coef, f0_hz)
        return self.get_signal(**controls)


class MultiInharmonic(InHarmonic):
    """Multi-string inharmonic synthesizer variant."""

    def get_controls(
        self,
        amplitudes: torch.Tensor,
        harmonic_distribution: torch.Tensor,
        inharm_coef: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        controls = super().get_controls(amplitudes, harmonic_distribution, inharm_coef, f0_hz[..., :1])
        controls["f0_hz"] = f0_hz
        controls["amplitudes"] = controls["amplitudes"] / max(1, f0_hz.shape[-1])
        return controls

    def get_signal(
        self,
        amplitudes: torch.Tensor,
        harmonic_distribution: torch.Tensor,
        harmonic_shifts: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> torch.Tensor:
        audio = 0.0
        for idx in range(f0_hz.shape[-1]):
            audio = audio + super().get_signal(amplitudes, harmonic_distribution, harmonic_shifts, f0_hz[..., idx : idx + 1])
        return audio
