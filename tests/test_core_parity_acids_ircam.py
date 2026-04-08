"""Numerical parity tests against acids-ircam ddsp_pytorch core equations."""

import math

import torch
import torch.nn.functional as F

from ddsp_piano_pytorch import core


# Reference implementations mirrored from:
# https://github.com/acids-ircam/ddsp_pytorch/blob/master/ddsp/core.py
def ref_scale_function(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7


def ref_remove_above_nyquist(amplitudes: torch.Tensor, pitch: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def ref_harmonic_synth(pitch: torch.Tensor, amplitudes: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def ref_fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    signal = F.pad(signal, (0, signal.shape[-1]))
    kernel = F.pad(kernel, (kernel.shape[-1], 0))
    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2 :]
    return output


def test_scale_function_parity() -> None:
    x = torch.randn(3, 5, 7)
    got = core.scale_function(x)
    exp = ref_scale_function(x)
    assert torch.allclose(got, exp, atol=1e-7, rtol=1e-6)


def test_remove_above_nyquist_parity() -> None:
    amps = torch.rand(2, 10, 12)
    pitch = torch.rand(2, 10, 1) * 500
    got = core.remove_above_nyquist(amps, pitch, 24000)
    exp = ref_remove_above_nyquist(amps, pitch, 24000)
    assert torch.allclose(got, exp, atol=1e-7, rtol=1e-6)


def test_harmonic_synth_parity() -> None:
    pitch = torch.rand(2, 128, 1) * 800
    amps = torch.rand(2, 128, 16)
    got = core.harmonic_synth(pitch, amps, 24000)
    exp = ref_harmonic_synth(pitch, amps, 24000)
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-5)


def test_fft_convolve_parity() -> None:
    signal = torch.rand(2, 4096)
    # acids-ircam implementation assumes matched signal/kernel lengths.
    kernel = torch.rand(2, 4096)
    got = core.fft_convolve(signal, kernel)
    exp = ref_fft_convolve(signal, kernel)
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-5)


def test_fft_convolve_mismatched_lengths() -> None:
    signal = torch.rand(2, 24000)
    kernel = torch.rand(2, 8000)
    out = core.fft_convolve(signal, kernel)
    assert out.shape == signal.shape
    assert torch.isfinite(out).all()


def test_midi_hz_roundtrip() -> None:
    midi = torch.linspace(21, 108, 64)
    hz = core.midi_to_hz(midi)
    midi_rt = core.hz_to_midi(hz)
    assert torch.allclose(midi, midi_rt, atol=1e-5, rtol=1e-5)
