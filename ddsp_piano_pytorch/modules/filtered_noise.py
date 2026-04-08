from __future__ import annotations

import math

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from ddsp_piano_pytorch.core import frequency_filter, resample, scale_function


def _check_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1) == 0)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 2 ** int(math.ceil(math.log2(x)))


class DynamicSizeFilteredNoise(nn.Module):
    """Filtered noise that supports arbitrary length via frame/sample rate."""

    def __init__(self, frame_rate: int = 250, sample_rate: int = 24000, window_size: int | None = None) -> None:
        super().__init__()
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.window_size = window_size

    @property
    def upsampling(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    def forward(self, magnitudes: torch.Tensor) -> torch.Tensor:
        batch, n_frames, _ = magnitudes.shape
        n_samples = n_frames * self.upsampling
        noise = torch.rand(batch, n_samples, device=magnitudes.device, dtype=magnitudes.dtype) * 2.0 - 1.0
        return frequency_filter(noise, magnitudes, window_size=self.window_size)


class NoiseBandNetSynth(nn.Module):
    """NoiseBandNet-style deterministic filtered noise synthesizer."""

    def __init__(
        self,
        n_band: int,
        upsampling: int = 96,
        sample_rate: int = 24000,
        min_noise_len: int = 2**8,
        linear_min_f: float = 20.0,
        linear_max_f_cutoff_fs: float = 4.0,
        filterbank_attenuation: float = 50.0,
        normalize_noise_bands: bool = True,
    ) -> None:
        super().__init__()
        if not _check_power_of_2(min_noise_len):
            raise ValueError("min_noise_len must be a power of 2.")
        self.n_band = n_band
        self.upsampling = upsampling
        self.sample_rate = sample_rate
        self.min_noise_len = min_noise_len
        self.linear_min_f = linear_min_f
        self.linear_max_f_cutoff_fs = linear_max_f_cutoff_fs
        self.filterbank_attenuation = filterbank_attenuation
        self.normalize_noise_bands = normalize_noise_bands

        filters = self._build_filterbank()
        max_len = max(len(f) for f in filters)
        self.noise_len = max(_next_power_of_2(max_len), self.min_noise_len)
        filters = [np.pad(f, (self.noise_len - len(f), 0)) for f in filters]
        magnitude_filters = np.abs(np.fft.rfft(np.stack(filters, axis=0), axis=-1))
        rng = np.random.default_rng(seed=42)
        phase = rng.uniform(-math.pi, math.pi, size=magnitude_filters.shape)
        phase = np.exp(1j * phase)
        phase[:, 0] = 0.0
        phase[:, -1] = 0.0
        bands = np.fft.irfft(magnitude_filters * phase, axis=-1).T
        if self.normalize_noise_bands:
            bands = bands / (np.max(np.abs(bands)) + 1e-8)
        self.register_buffer("noise_bands", torch.tensor(bands[None, ...], dtype=torch.float32), persistent=False)

    def _get_filter(self, cutoff, pass_zero: bool, transition_bw: float = 0.2, scale: bool = True) -> np.ndarray:
        if isinstance(cutoff, np.ndarray):
            bandwidth = abs(float(cutoff[1] - cutoff[0]))
        elif pass_zero:
            bandwidth = float(cutoff)
        else:
            bandwidth = abs(float((self.sample_rate / 2.0) - cutoff))
        width = max(1e-5, (bandwidth / (self.sample_rate / 2.0)) * transition_bw)
        taps, beta = scipy.signal.kaiserord(ripple=self.filterbank_attenuation, width=width)
        taps = 2 * (taps // 2) + 1
        return scipy.signal.firwin(
            numtaps=taps,
            cutoff=cutoff,
            window=("kaiser", beta),
            scale=scale,
            fs=self.sample_rate,
            pass_zero=pass_zero,
        )

    def _build_filterbank(self) -> list[np.ndarray]:
        n_lin = self.n_band // 2
        n_log = self.n_band - n_lin
        linear_max_f = (self.sample_rate / 2.0) / self.linear_max_f_cutoff_fs
        linear = np.linspace(self.linear_min_f, linear_max_f, n_lin)
        linear = np.vstack([linear[:-1], linear[1:]]).T if len(linear) > 1 else np.zeros((0, 2))
        log = np.geomspace(linear_max_f, self.sample_rate / 2.0, n_log, endpoint=False)
        log = np.vstack([log[:-1], log[1:]]).T if len(log) > 1 else np.zeros((0, 2))
        bands = np.concatenate([linear, log], axis=0) if len(linear) + len(log) > 0 else np.zeros((0, 2))

        filt: list[np.ndarray] = []
        for i in range(max(1, len(bands))):
            if i == 0 and len(bands) > 0:
                filt.append(self._get_filter(float(bands[i, 0]), pass_zero=True))
            cutoff = bands[i] if len(bands) > 0 else np.array([20.0, self.sample_rate / 2.5])
            filt.append(self._get_filter(cutoff, pass_zero=False))
            if i == len(bands) - 1 and len(bands) > 0:
                filt.append(self._get_filter(float(bands[i, -1]), pass_zero=False))
        return filt[: self.n_band]

    def forward(self, magnitudes: torch.Tensor) -> torch.Tensor:
        amplitudes = scale_function(magnitudes)
        frame_len = max(1, self.noise_len // self.upsampling)
        n_frames = int(math.ceil(amplitudes.shape[1] / frame_len))
        n_samples = amplitudes.shape[1] * self.upsampling

        shift = int(torch.randint(0, self.noise_bands.shape[1], (1,), device=amplitudes.device).item())
        noise_bands = torch.roll(self.noise_bands.to(amplitudes.device, amplitudes.dtype), shifts=shift, dims=1)

        chunks = []
        for i in range(n_frames):
            sl = amplitudes[:, i * frame_len : (i + 1) * frame_len, :]
            if sl.numel() == 0:
                continue
            up = resample(sl, self.upsampling)
            if up.shape[1] < self.noise_len:
                up = torch.nn.functional.pad(up, (0, 0, 0, self.noise_len - up.shape[1]))
            chunk = (noise_bands[:, : up.shape[1], : self.n_band] * up[:, :, : self.n_band]).sum(dim=-1)
            chunks.append(chunk)

        signal = torch.cat(chunks, dim=1) if chunks else amplitudes.new_zeros(amplitudes.shape[0], n_samples)
        return signal[:, :n_samples]
