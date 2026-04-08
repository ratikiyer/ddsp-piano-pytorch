"""Parity tests against DDSP-Piano TensorFlow logic (formula-level)."""

import torch

from ddsp_piano_pytorch.core import get_inharmonic_frequencies, safe_divide
from ddsp_piano_pytorch.modules.inharm_synth import InHarmonic
from ddsp_piano_pytorch.modules.sub_modules import F0ProcessorCell, NoteRelease, OnsetLinspaceCell


def ref_get_inharmonic_freq(f0_hz: torch.Tensor, inharm_coef: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    # Mirrors ddsp_piano/modules/inharm_synth.py:get_inharmonic_freq
    int_multiplier = torch.linspace(1.0, float(n_harmonics), int(n_harmonics), device=f0_hz.device, dtype=f0_hz.dtype).view(1, 1, -1)
    inharm_factor = torch.sqrt((int_multiplier.pow(2) * inharm_coef) + 1.0)
    return f0_hz * int_multiplier * inharm_factor


def test_get_inharmonic_freq_formula_parity() -> None:
    f0 = torch.rand(2, 20, 1) * 1000
    bcoef = torch.rand(2, 20, 1) * 0.01
    got = get_inharmonic_frequencies(f0, bcoef, 32)
    exp = ref_get_inharmonic_freq(f0, bcoef, 32)
    assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6)


def test_inharmonic_get_controls_normalization() -> None:
    synth = InHarmonic(frame_rate=250, sample_rate=24000)
    amplitudes = torch.randn(2, 30, 1)
    harmonic_distribution = torch.randn(2, 30, 32)
    inharm_coef = torch.rand(2, 30, 1) * 0.01
    f0_hz = torch.rand(2, 30, 1) * 500 + 50
    controls = synth.get_controls(amplitudes, harmonic_distribution, inharm_coef, f0_hz)
    sums = controls["harmonic_distribution"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3)
    assert torch.all(controls["amplitudes"] >= 0)


def test_f0_processor_release_behavior() -> None:
    cell = F0ProcessorCell(frame_rate=250)
    prev_note = torch.tensor([[[60.0]]])
    prev_steps = torch.tensor([[[0.0]]])
    # No new note: should hold pitch through release start.
    out, new_note, new_steps = cell.step(torch.tensor([[[0.0]]]), prev_note, prev_steps)
    assert out.item() == new_note.item()
    assert new_steps.item() >= 0


def test_note_release_sequence_shape() -> None:
    mod = NoteRelease(frame_rate=250)
    cond = torch.zeros(2, 25, 16, 2)
    cond[:, 0, :, 0] = 60.0
    out = mod(cond)
    assert out.shape == (2, 25, 16, 1)


def test_onset_linspace_progress_and_reset() -> None:
    cell = OnsetLinspaceCell()
    onsets = torch.zeros(1, 10, 4, 1)
    onsets[:, 0, 0, 0] = 1.0
    onsets[:, 5, 2, 0] = 1.0
    time = cell(onsets)
    assert time.shape == (1, 10, 1)
    assert time[0, 1, 0] >= time[0, 0, 0]
    assert time[0, 5, 0] <= time[0, 4, 0]


def test_safe_divide_no_nan() -> None:
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[0.0, 0.0]])
    out = safe_divide(a, b)
    assert torch.isfinite(out).all()
