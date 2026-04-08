from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from ddsp_piano_pytorch.modules.filtered_noise import DynamicSizeFilteredNoise
from ddsp_piano_pytorch.modules.inharm_synth import InHarmonic
from ddsp_piano_pytorch.modules.processor_group import MultiAdd, ProcessorGroup
from ddsp_piano_pytorch.modules.sub_modules import (
    BackgroundNoiseFilter,
    Detuner,
    FiLMContextNetwork,
    InharmonicityNetwork,
    JointParametricInharmTuning,
    MonophonicDeepNetwork,
    MultiInstrumentFeedbackDelayReverb,
    NoteRelease,
    OneHotZEncoder,
    Parallelizer,
)


def exists(x: Any) -> bool:
    return x is not None


class PianoModel(nn.Module):
    """Top-level DDSP-Piano model in PyTorch."""

    def __init__(
        self,
        n_synths: int = 16,
        n_harmonics: int = 96,
        n_noise_bands: int = 65,
        sample_rate: int = 24000,
        frame_rate: int = 250,
        context_dim: int = 128,
        z_encoder: nn.Module | None = None,
        note_release: nn.Module | None = None,
        context_network: nn.Module | None = None,
        parallelizer: Parallelizer | None = None,
        monophonic_network: nn.Module | None = None,
        inharm_model: nn.Module | None = None,
        detuner: nn.Module | None = None,
        background_noise_model: nn.Module | None = None,
        reverb_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.n_synths = n_synths
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.upsampling = int(sample_rate / frame_rate)

        self.z_encoder = z_encoder or OneHotZEncoder(n_instruments=10, z_dim=16, frame_rate=frame_rate)
        self.note_release = note_release or NoteRelease(frame_rate=frame_rate)
        self.context_network = context_network or FiLMContextNetwork(
            n_synths=n_synths, n_instruments=10, layer_dim=512, context_dim=context_dim, z_dim=16
        )
        self.parallelizer = parallelizer or Parallelizer(n_synths=n_synths)
        self.monophonic_network = monophonic_network or MonophonicDeepNetwork(
            rnn_channels=512,
            ch=128,
            context_dim=context_dim,
            output_splits=(("amplitudes", 1), ("harmonic_distribution", n_harmonics), ("noise_magnitudes", n_noise_bands)),
        )
        self.inharm_model = inharm_model or InharmonicityNetwork()
        self.detuner = detuner or Detuner()
        self.background_noise_model = background_noise_model or BackgroundNoiseFilter(
            n_instruments=10, n_filters=n_noise_bands, frame_rate=frame_rate
        )
        self.reverb_model = reverb_model or MultiInstrumentFeedbackDelayReverb(
            n_instruments=10, sample_rate=sample_rate, fdn_n_delays=8
        )

        self.inharmonic = InHarmonic(frame_rate=frame_rate, sample_rate=sample_rate)
        self.noise = DynamicSizeFilteredNoise(frame_rate=frame_rate, sample_rate=sample_rate)
        self.mix = MultiAdd()

        processors = OrderedDict(
            [
                ("harmonic_audio", (self.inharmonic, ["amplitudes", "harmonic_distribution", "inharm_coef", "f0_hz"])),
                ("noise_audio", (self.noise, ["noise_magnitudes"])),
                ("summed_audio", (self.mix, ["harmonic_audio", "noise_audio", "background_audio"])),
            ]
        )
        self.processor_group = ProcessorGroup(processors)

    def alternate_training(self, first_phase: bool = True) -> None:
        for module in [self.inharm_model, self.detuner]:
            if exists(module):
                for p in module.parameters():
                    p.requires_grad_(not first_phase)
        if exists(self.z_encoder) and hasattr(self.z_encoder, "alternate_training"):
            self.z_encoder.alternate_training(first_phase=first_phase)
        for module in [self.note_release, self.context_network, self.background_noise_model, self.monophonic_network, self.reverb_model]:
            if exists(module):
                for p in module.parameters():
                    p.requires_grad_(first_phase)
        if exists(self.detuner) and hasattr(self.detuner, "use_detune"):
            self.detuner.use_detune = not first_phase

    def forward(
        self,
        conditioning: torch.Tensor,
        pedals: torch.Tensor,
        piano_model: torch.Tensor,
        onsets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            conditioning: [B, T, n_synths, 2] (pitch, velocity)
            pedals: [B, T, 4]
            piano_model: [B]
            onsets: [B, T, n_synths] optional
        Returns:
            dict with 'audio' and intermediate controls.
        """
        b, t, _, _ = conditioning.shape
        z, global_inharm, global_detuning = self.z_encoder(piano_model)
        context = self.context_network(conditioning, pedals, piano_model)
        background_noise = self.background_noise_model(piano_model, n_frames=t)
        reverb_ir = self.reverb_model(piano_model)

        features: dict[str, torch.Tensor] = {
            "conditioning": conditioning,
            "context": context,
            "global_inharm": global_inharm,
            "global_detuning": global_detuning,
            "piano_model": piano_model,
        }
        features = self.parallelizer(features, parallelize=True)

        par_cond = features["conditioning"]
        par_context = features["context"]
        par_piano_model = features["piano_model"]
        extended_pitch = self.note_release(par_cond.reshape(b, t, self.n_synths, 2)).reshape(b * self.n_synths, t, 1)
        if isinstance(self.inharm_model, JointParametricInharmTuning):
            f0_hz, inharm_coef = self.inharm_model(extended_pitch, par_piano_model)
        else:
            inharm_coef = self.inharm_model(extended_pitch, features.get("global_inharm"))
            f0_hz = self.detuner(extended_pitch, features.get("global_detuning"))
        mono_out = self.monophonic_network(par_cond, extended_pitch, par_context)

        features.update(mono_out)
        features["inharm_coef"] = inharm_coef
        features["f0_hz"] = f0_hz

        harmonic_audio = self.inharmonic(
            features["amplitudes"],
            features["harmonic_distribution"],
            features["inharm_coef"],
            features["f0_hz"],
        )
        noise_audio = self.noise(features["noise_magnitudes"])
        background_audio = self.noise(background_noise)

        dry_parallel = harmonic_audio + noise_audio
        dry = dry_parallel.reshape(self.n_synths, b, -1).sum(dim=0) + background_audio
        wet = self.reverb_model.reverb_model(dry, reverb_ir)
        return {
            "audio": wet,
            "audio_dry": dry,
            "reverb_ir": reverb_ir,
            "inharm_coef": inharm_coef,
            "z": z,
        }
