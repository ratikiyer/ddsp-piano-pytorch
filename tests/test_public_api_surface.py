import inspect

import ddsp_piano_pytorch as pkg
from ddsp_piano_pytorch import core
from ddsp_piano_pytorch.modules import (
    DynamicSizeFilteredNoise,
    FeedbackDelayNetwork,
    InHarmonic,
    InharmonicityLoss,
    MultiAdd,
    MultiInharmonic,
    NoiseBandNetSynth,
    PianoModel,
    ProcessorGroup,
    ReverbRegularizer,
    SpectralLoss,
)


def test_root_public_api() -> None:
    assert hasattr(pkg, "PianoModel")


def test_core_public_functions_exist() -> None:
    expected = [
        "harmonic_synth",
        "fft_convolve",
        "amp_to_impulse_response",
        "remove_above_nyquist",
        "scale_function",
        "multiscale_fft",
        "safe_log",
        "resample",
        "upsample",
        "mlp",
        "gru",
        "midi_to_hz",
        "hz_to_midi",
        "safe_divide",
        "cos_oscillator_bank",
        "get_inharmonic_frequencies",
        "inharmonic_harmonic_synthesis",
        "frequency_filter",
        "get_fft_size",
    ]
    for name in expected:
        assert hasattr(core, name), f"Missing public API: core.{name}"
        assert callable(getattr(core, name))


def test_module_classes_constructible() -> None:
    # Constructor-level API parity checks.
    assert inspect.isclass(PianoModel)
    assert inspect.isclass(InHarmonic)
    assert inspect.isclass(MultiInharmonic)
    assert inspect.isclass(DynamicSizeFilteredNoise)
    assert inspect.isclass(NoiseBandNetSynth)
    assert inspect.isclass(FeedbackDelayNetwork)
    assert inspect.isclass(MultiAdd)
    assert inspect.isclass(ProcessorGroup)
    assert inspect.isclass(SpectralLoss)
    assert inspect.isclass(InharmonicityLoss)
    assert inspect.isclass(ReverbRegularizer)
