"""Core DSP utilities for DDSP-Piano PyTorch.

Most primitives are vendored/adapted from:
https://github.com/acids-ircam/ddsp_pytorch
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7


def safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=EPS))


def multiscale_fft(signal: torch.Tensor, scales: Iterable[int], overlap: float = 0.75) -> list[torch.Tensor]:
    stfts = []
    for size in scales:
        hop = int(size * (1.0 - overlap))
        hop = max(1, hop)
        window = torch.hann_window(size, device=signal.device, dtype=signal.dtype)
        spec = torch.stft(
            signal,
            n_fft=size,
            hop_length=hop,
            win_length=size,
            window=window,
            center=True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(spec)
    return stfts


def resample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Hann-windowed upsampling from ddsp_pytorch."""
    if factor == 1:
        return x
    batch, frames, channels = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channels, 1, frames)
    window = torch.hann_window(factor * 2, dtype=x.dtype, device=x.device).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2], dtype=x.dtype, device=x.device)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = F.pad(y, [factor, factor])
    y = F.conv1d(y, window)[..., :-1]
    return y.reshape(batch, channels, factor * frames).permute(0, 2, 1)


def upsample(signal: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return signal
    signal = signal.permute(0, 2, 1)
    signal = F.interpolate(signal, size=signal.shape[-1] * factor, mode="linear", align_corners=False)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes: torch.Tensor, pitch_hz: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    n_harm = amplitudes.shape[-1]
    harmonics = torch.arange(1, n_harm + 1, device=pitch_hz.device, dtype=pitch_hz.dtype)
    pitches = pitch_hz * harmonics
    aa = (pitches < (sampling_rate / 2.0)).to(amplitudes.dtype) + 1e-4
    return amplitudes * aa


def scale_function(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.sigmoid(x).pow(math.log(10)) + EPS


def mlp(in_size: int, hidden_size: int, n_layers: int) -> nn.Sequential:
    channels = [in_size] + [hidden_size] * n_layers
    layers: list[nn.Module] = []
    for i in range(n_layers):
        layers += [nn.Linear(channels[i], channels[i + 1]), nn.LayerNorm(channels[i + 1]), nn.LeakyReLU()]
    return nn.Sequential(*layers)


def gru(n_input: int, hidden_size: int) -> nn.GRU:
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(pitch: torch.Tensor, amplitudes: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / float(sampling_rate), dim=1)
    harmonics = torch.arange(1, n_harmonic + 1, device=omega.device, dtype=omega.dtype)
    omegas = omega * harmonics
    return (torch.sin(omegas) * amplitudes).sum(dim=-1, keepdim=True)


def amp_to_impulse_response(amp: torch.Tensor, target_size: int) -> torch.Tensor:
    amp = torch.view_as_complex(torch.stack([amp, torch.zeros_like(amp)], dim=-1))
    amp = fft.irfft(amp)
    filter_size = amp.shape[-1]
    amp = torch.roll(amp, filter_size // 2, dims=-1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)
    amp = amp * win
    amp = F.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -(filter_size // 2), dims=-1)
    return amp


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    signal_len = signal.shape[-1]
    kernel_len = kernel.shape[-1]
    if signal_len == kernel_len:
        # Exact acids-ircam behavior for parity.
        s = F.pad(signal, (0, signal_len))
        k = F.pad(kernel, (kernel_len, 0))
        out = fft.irfft(fft.rfft(s) * fft.rfft(k))
        return out[..., out.shape[-1] // 2 :]
    signal = F.pad(signal, (0, kernel_len))
    kernel = F.pad(kernel, (kernel_len, 0))
    n_fft = signal_len + 2 * kernel_len
    out = fft.irfft(fft.rfft(signal, n=n_fft) * fft.rfft(kernel, n=n_fft), n=n_fft)
    return out[..., kernel_len : kernel_len + signal_len]


def midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
    return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)


def hz_to_midi(hz: torch.Tensor) -> torch.Tensor:
    hz = torch.clamp(hz, min=EPS)
    return 69.0 + 12.0 * torch.log2(hz / 440.0)


def safe_divide(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return a / (b + eps)


def get_inharmonic_frequencies(f0_hz: torch.Tensor, inharm_coef: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    multipliers = torch.arange(1, n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype).view(1, 1, -1)
    inharm_factor = torch.sqrt(1.0 + inharm_coef * multipliers.pow(2))
    return f0_hz * multipliers * inharm_factor


def cos_oscillator_bank(
    frequencies: torch.Tensor,
    amplitudes: torch.Tensor,
    sample_rate: int,
    sum_sinusoids: bool = True,
) -> torch.Tensor:
    amplitudes = remove_above_nyquist(amplitudes, frequencies, sample_rate)
    omegas = frequencies * (2.0 * math.pi / float(sample_rate))
    phases = torch.cumsum(omegas, dim=1)
    audio = amplitudes * torch.cos(phases)
    return audio.sum(dim=-1) if sum_sinusoids else audio


def inharmonic_harmonic_synthesis(
    f0_hz: torch.Tensor,
    amplitudes: torch.Tensor,
    harmonic_distribution: torch.Tensor,
    inharm_coef: torch.Tensor,
    sample_rate: int,
    frame_rate: int,
) -> torch.Tensor:
    n_frames = f0_hz.shape[1]
    n_samples = int(round(sample_rate * (n_frames / float(frame_rate))))
    n_harm = harmonic_distribution.shape[-1]
    frequencies = get_inharmonic_frequencies(f0_hz, inharm_coef, n_harm)
    harmonic_amplitudes = amplitudes * harmonic_distribution
    factor = max(1, sample_rate // frame_rate)
    frequency_env = upsample(frequencies, factor)[:, :n_samples]
    amplitude_env = upsample(harmonic_amplitudes, factor)[:, :n_samples]
    return cos_oscillator_bank(frequency_env, amplitude_env, sample_rate, sum_sinusoids=True)


def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    size = int(frame_size + ir_size - 1)
    if not power_of_2:
        return size
    return int(2 ** math.ceil(math.log2(max(2, size))))


def frequency_filter(audio: torch.Tensor, magnitudes: torch.Tensor, window_size: int | None = None) -> torch.Tensor:
    """Applies a framewise STFT-domain filtering approximation.

    Note: TF DDSP builds explicit FIR impulse responses per frame before
    convolution. This implementation uses STFT-domain multiplication for
    efficiency in PyTorch and is validated with numeric tolerances.
    """
    batch, n_samples = audio.shape
    _, n_frames, n_bins = magnitudes.shape
    if window_size is None:
        window_size = max(2, 2 * (n_bins - 1))
    hop = min(max(1, window_size // 2), max(1, n_samples // n_frames))
    window = torch.hann_window(window_size, device=audio.device, dtype=audio.dtype)
    stft = torch.stft(
        audio,
        n_fft=window_size,
        hop_length=hop,
        win_length=window_size,
        window=window,
        return_complex=True,
        center=True,
    )
    if stft.shape[-1] != n_frames:
        magnitudes = F.interpolate(magnitudes.transpose(1, 2), size=stft.shape[-1], mode="linear", align_corners=False)
        magnitudes = magnitudes.transpose(1, 2)
    if stft.shape[1] != n_bins:
        magnitudes = F.interpolate(magnitudes.transpose(1, 2), size=stft.shape[1], mode="linear", align_corners=False)
        magnitudes = magnitudes.transpose(1, 2)
    filt = torch.clamp(magnitudes, min=0.0).transpose(1, 2)
    out = torch.istft(
        stft * filt,
        n_fft=window_size,
        hop_length=hop,
        win_length=window_size,
        window=window,
        length=n_samples,
        center=True,
    )
    return out.reshape(batch, n_samples)
