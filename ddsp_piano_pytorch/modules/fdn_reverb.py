from __future__ import annotations

import math

import torch
import torch.nn as nn

from ddsp_piano_pytorch.core import fft_convolve


class FeedbackDelayNetwork(nn.Module):
    """Differentiable FDN reverb with Householder mixing."""

    def __init__(
        self,
        sampling_rate: int = 24000,
        delay_lines: int = 8,
        delay_values: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.delay_lines = delay_lines
        if delay_values is None:
            delay_values = [233, 311, 421, 461, 587, 613, 789, 891][:delay_lines]
        self.register_buffer("delay_values", torch.tensor(delay_values, dtype=torch.float32), persistent=False)

        eye = torch.eye(delay_lines, dtype=torch.float32)
        self.register_buffer("mixing_matrix", -eye + (2.0 / delay_lines) * torch.ones_like(eye), persistent=False)

    def _late_ir(
        self,
        input_gain: torch.Tensor,
        output_gain: torch.Tensor,
        gain_allpass: torch.Tensor,
        delays_allpass: torch.Tensor,
        time_rev_0_sec: torch.Tensor,
        alpha_tone: torch.Tensor,
        n_fft: int,
    ) -> torch.Tensor:
        device = input_gain.device
        dtype = input_gain.dtype
        kfreq = n_fft // 2 + 1
        wk = 2 * math.pi * torch.arange(kfreq, device=device, dtype=dtype) / n_fft
        z = torch.exp(-1j * wk[:, None])

        d = self.delay_values.to(device=device, dtype=dtype)
        z_d = torch.exp(-1j * wk[:, None] * torch.floor(d)[None, :])
        frac = d - torch.floor(d)
        eta = (1 - frac) / (1 + frac + 1e-8)
        allpass_interp = (eta[None, :] + z) / (1.0 + eta[None, :] * z + 1e-8)
        diag_delay = torch.diag_embed(z_d * allpass_interp)

        delay_sec = (d + delays_allpass.sum(dim=-1)) / self.sampling_rate
        k = torch.pow(10.0, -3.0 * delay_sec / (time_rev_0_sec + 1e-6))
        kpi = torch.pow(10.0, -3.0 * delay_sec / (alpha_tone * time_rev_0_sec + 1e-6))
        g = 2 * k * kpi / (k + kpi + 1e-8)
        p = (k - kpi) / (k + kpi + 1e-8)
        filter_diag = torch.diag_embed(g[None, :] / (1.0 - p[None, :] * z + 1e-8))

        g_ap = gain_allpass
        d_ap = delays_allpass
        z_ap = torch.exp(1j * wk[:, None, None] * d_ap[None, :, :])
        ap_tf = torch.prod((1 + g_ap[None, :, :] * z_ap) / (g_ap[None, :, :] + z_ap + 1e-8), dim=-1)
        allpass_matrix = torch.diag_embed(ap_tf)

        mixing = self.mixing_matrix.to(device=device, dtype=dtype).to(torch.complex64)
        feedback = filter_diag @ mixing[None, ...] @ allpass_matrix

        eye = torch.eye(self.delay_lines, device=device, dtype=torch.complex64)[None, ...].expand(kfreq, -1, -1)
        lhs = eye - feedback @ diag_delay
        rhs = (diag_delay @ input_gain.view(1, self.delay_lines, 1).to(torch.complex64))
        # Using solve is numerically safer than explicit inverse.
        solved = torch.linalg.solve(lhs, rhs)
        h = output_gain.view(1, 1, self.delay_lines).to(torch.complex64) @ solved
        return torch.fft.irfft(h.squeeze(-1).squeeze(-1), n=n_fft)

    def get_ir(
        self,
        input_gain: torch.Tensor,
        output_gain: torch.Tensor,
        gain_allpass: torch.Tensor,
        delays_allpass: torch.Tensor,
        time_rev_0_sec: torch.Tensor,
        alpha_tone: torch.Tensor,
        early_ir: torch.Tensor,
        n_fft: int = 32768,
    ) -> torch.Tensor:
        late = self._late_ir(input_gain, output_gain, gain_allpass, delays_allpass, time_rev_0_sec, alpha_tone, n_fft=n_fft)
        early = early_ir.reshape(-1)
        if early.shape[0] < late.shape[0]:
            early = torch.nn.functional.pad(early, (0, late.shape[0] - early.shape[0]))
        return early[: late.shape[0]] + late

    def forward(self, audio_dry: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        if ir.dim() == 1:
            ir = ir.unsqueeze(0)
        return fft_convolve(audio_dry, ir)
