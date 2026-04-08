from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddsp_piano_pytorch.core import midi_to_hz, resample
from ddsp_piano_pytorch.modules.fdn_reverb import FeedbackDelayNetwork


class FcStack(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layers: int = 2) -> None:
        super().__init__()
        mods = []
        cur = in_dim
        for _ in range(layers):
            mods += [nn.Linear(cur, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
            cur = out_dim
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SimpleContextNet(nn.Module):
    def __init__(self, n_synths: int = 16, hidden_dim: int = 128, context_dim: int = 64) -> None:
        super().__init__()
        in_dim = n_synths * 2 + 4
        self.fc = FcStack(in_dim=in_dim, out_dim=hidden_dim, layers=2)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, context_dim)

    def forward(self, conditioning: torch.Tensor, pedals: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        b, t, n, f = conditioning.shape
        x = torch.cat([conditioning.reshape(b, t, n * f), pedals], dim=-1)
        x = self.fc(x)
        x, _ = self.gru(x)
        return self.out(x)


class FiLMContextNetwork(nn.Module):
    def __init__(
        self,
        n_synths: int = 16,
        n_instruments: int = 10,
        layer_dim: int = 512,
        context_dim: int = 128,
        z_dim: int = 16,
    ) -> None:
        super().__init__()
        self.midi_norm = 128.0
        self.conditioning_head = FcStack(n_synths * 2, layer_dim // 2, layers=2)
        self.pedal_head = FcStack(4, layer_dim // 2, layers=2)
        self.piano_id_head = nn.Embedding(n_instruments, z_dim)
        self.main_in = nn.Linear(layer_dim, layer_dim)
        self.main_gru = nn.GRU(layer_dim, layer_dim, batch_first=True)
        self.main_out = nn.Sequential(nn.Linear(layer_dim, layer_dim), nn.LayerNorm(layer_dim), nn.LeakyReLU())
        self.film = nn.Linear(z_dim, layer_dim * 2)
        self.output_layer = FcStack(layer_dim, context_dim, layers=2)

    def forward(self, conditioning: torch.Tensor, pedals: torch.Tensor, piano_model: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = conditioning.shape
        cond = conditioning / conditioning.new_tensor([self.midi_norm, 1.0])
        cond = cond.reshape(b, t, n * 2)
        cond_feat = self.conditioning_head(cond)
        pedal_feat = self.pedal_head(pedals)
        x = self.main_in(torch.cat([cond_feat, pedal_feat], dim=-1))
        x, _ = self.main_gru(x)
        x = self.main_out(x)

        piano_feat = self.piano_id_head(piano_model).unsqueeze(1).expand(-1, t, -1)
        film_coef, film_bias = torch.chunk(self.film(piano_feat), chunks=2, dim=-1)
        x = x * film_coef + film_bias
        return self.output_layer(x)


class OneHotZEncoder(nn.Module):
    def __init__(
        self,
        n_instruments: int = 10,
        z_dim: int = 16,
        n_inharm_params: int = 1,
        n_detuning_params: int = 1,
        duration: float | None = None,
        frame_rate: int = 250,
    ) -> None:
        super().__init__()
        self.n_instruments = n_instruments
        self.z_dim = z_dim
        self.duration = duration
        self.frame_rate = frame_rate
        self.embedding = nn.Embedding(n_instruments, z_dim)
        self.inharm_embedding = nn.Embedding(n_instruments, n_inharm_params)
        self.detune_embedding = nn.Embedding(n_instruments, n_detuning_params)

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.frame_rate) if self.duration is not None else 1

    def alternate_training(self, first_phase: bool = True) -> None:
        self.embedding.weight.requires_grad_(first_phase)
        self.inharm_embedding.weight.requires_grad_(not first_phase)
        self.detune_embedding.weight.requires_grad_(not first_phase)

    def forward(self, piano_model: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.n_instruments == 1:
            piano_model = torch.zeros_like(piano_model)
        z = self.embedding(piano_model).unsqueeze(1)
        inharm = self.inharm_embedding(piano_model).unsqueeze(1)
        detune = self.detune_embedding(piano_model).unsqueeze(1)
        if self.n_frames > 1:
            z = resample(z, self.n_frames)
            inharm = resample(inharm, self.n_frames)
            detune = resample(detune, self.n_frames)
        return z, inharm, detune


class BackgroundNoiseFilter(nn.Module):
    def __init__(
        self,
        n_instruments: int = 10,
        n_filters: int = 65,
        duration: float | None = None,
        frame_rate: int = 250,
        denoise: bool = False,
    ) -> None:
        super().__init__()
        self.n_instruments = n_instruments
        self.n_filters = n_filters
        self.duration = duration
        self.frame_rate = frame_rate
        self.denoise = denoise
        self.embedding = nn.Embedding(n_instruments, n_filters)

    def forward(self, piano_model: torch.Tensor, n_frames: int) -> torch.Tensor:
        if self.n_instruments == 1:
            piano_model = torch.zeros_like(piano_model)
        mags = self.embedding(piano_model).unsqueeze(1).expand(-1, n_frames, -1)
        if self.denoise:
            mags = torch.zeros_like(mags)
        return mags


class MultiInstrumentFeedbackDelayReverb(nn.Module):
    def __init__(self, n_instruments: int = 10, sample_rate: int = 24000, fdn_n_delays: int = 8, early_ir_length: int = 200) -> None:
        super().__init__()
        self.n_instruments = n_instruments
        self.fdn_n_delays = fdn_n_delays
        self.input_gain = nn.Embedding(n_instruments, fdn_n_delays)
        self.output_gain = nn.Embedding(n_instruments, fdn_n_delays)
        self.gain_allpass = nn.Embedding(n_instruments, 4 * fdn_n_delays)
        self.delays_allpass = nn.Embedding(n_instruments, 4 * fdn_n_delays)
        self.time_rev_0_sec = nn.Embedding(n_instruments, 1)
        self.alpha_tone = nn.Embedding(n_instruments, 1)
        self.early_ir = nn.Embedding(n_instruments, early_ir_length)
        self.reverb_model = FeedbackDelayNetwork(sampling_rate=sample_rate, delay_lines=fdn_n_delays)

    def _reshape_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.fdn_n_delays, 4)

    def forward(self, piano_model: torch.Tensor) -> torch.Tensor:
        if self.n_instruments == 1:
            piano_model = torch.zeros_like(piano_model)
        in_gain = self.input_gain(piano_model)
        out_gain = self.output_gain(piano_model)
        g_ap = self._reshape_embedding(self.gain_allpass(piano_model))
        d_ap = self._reshape_embedding(self.delays_allpass(piano_model))
        t0 = F.relu(self.time_rev_0_sec(piano_model))
        alpha = torch.sigmoid(self.alpha_tone(piano_model))
        early = self.early_ir(piano_model)
        return self.reverb_model.get_ir(
            input_gain=in_gain,
            output_gain=out_gain,
            gain_allpass=g_ap,
            delays_allpass=d_ap,
            time_rev_0_sec=t0,
            alpha_tone=alpha,
            early_ir=early,
        )


class MonophonicNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_splits: tuple[tuple[str, int], ...] = (("amplitudes", 1), ("harmonic_distribution", 96), ("noise_magnitudes", 65)),
    ) -> None:
        super().__init__()
        self.midi_norm = 128.0
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, dim) for name, dim in output_splits})

    def compute_output(self, conditioning: torch.Tensor, extended_pitch: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([extended_pitch / self.midi_norm, conditioning / conditioning.new_tensor([self.midi_norm, 1.0]), context], dim=-1)
        x, _ = self.rnn(x)
        return x

    def forward(self, conditioning: torch.Tensor, extended_pitch: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.compute_output(conditioning, extended_pitch, context)
        return {name: head(x) for name, head in self.heads.items()}


class MonophonicDeepNetwork(MonophonicNetwork):
    def __init__(
        self,
        rnn_channels: int = 512,
        ch: int = 128,
        output_splits: tuple[tuple[str, int], ...] = (("amplitudes", 1), ("harmonic_distribution", 96), ("noise_magnitudes", 65)),
        context_dim: int = 128,
    ) -> None:
        super().__init__(input_dim=ch * 3, hidden_dim=rnn_channels, output_splits=output_splits)
        self.input_pitch = FcStack(1, ch, layers=2)
        self.input_cond = FcStack(2, ch, layers=2)
        self.input_context = FcStack(context_dim, ch, layers=2)
        self.rnn = nn.GRU(ch * 3, rnn_channels, batch_first=True)
        self.out_stack = FcStack(ch * 3 + rnn_channels, rnn_channels, layers=2)
        self.heads = nn.ModuleDict({name: nn.Linear(rnn_channels, dim) for name, dim in output_splits})

    def compute_output(self, conditioning: torch.Tensor, extended_pitch: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        p = self.input_pitch(extended_pitch / self.midi_norm)
        c = self.input_cond(conditioning / conditioning.new_tensor([self.midi_norm, 1.0]))
        ctx = self.input_context(context)
        x = torch.cat([p, c, ctx], dim=-1)
        x, _ = self.rnn(x)
        x = torch.cat([p, c, ctx, x], dim=-1)
        return self.out_stack(x)


class Parallelizer(nn.Module):
    def __init__(
        self,
        n_synths: int = 16,
        global_keys: tuple[str, ...] = ("conditioning", "context", "global_inharm", "global_detuning", "piano_model"),
        mono_keys: tuple[str, ...] = ("f0_hz", "inharm_coef", "amplitudes", "harmonic_distribution", "noise_magnitudes"),
    ) -> None:
        super().__init__()
        self.n_synths = n_synths
        self.global_keys = global_keys
        self.mono_keys = mono_keys
        self.batch_size: int | None = None

    def _put_polyphony_first(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(self.n_synths, -1)
        elif x.dim() in (2, 3):
            x = x.unsqueeze(0).expand(self.n_synths, *x.shape)
        elif x.dim() == 4:
            x = x.permute(2, 0, 1, 3)
        return x

    def _parallelize_feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_size is None:
            self.batch_size = x.shape[1]
        return x.reshape(self.n_synths * self.batch_size, *x.shape[2:])

    def _unparallelize_feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_size is None:
            raise RuntimeError("Parallelizer batch_size is not initialized.")
        return x.reshape(self.n_synths, self.batch_size, *x.shape[1:])

    def parallelize(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.batch_size = features["conditioning"].shape[0]
        for k in self.global_keys:
            if k in features:
                features[k] = self._parallelize_feature(self._put_polyphony_first(features[k]))
        return features

    def unparallelize(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for k in self.mono_keys:
            if k in features:
                features[k] = self._unparallelize_feature(features[k])
                for i in range(self.n_synths):
                    features[f"{k}_{i}"] = features[k][i]
        return features

    def forward(self, features: dict[str, torch.Tensor], parallelize: bool = True) -> dict[str, torch.Tensor]:
        return self.parallelize(features) if parallelize else self.unparallelize(features)


class InharmonicityNetwork(nn.Module):
    """Parametric inharmonicity model inspired by Rigaud et al."""

    def __init__(self) -> None:
        super().__init__()
        self.midi_norm = 128.0
        self.model_specific_weight = nn.Parameter(torch.zeros(1))
        self.register_buffer("slopes", torch.tensor([9.26e-2 * self.midi_norm, -8.47e-2 * self.midi_norm], dtype=torch.float32))
        self.register_buffer("offsets", torch.tensor([-13.64 / (self.midi_norm * 9.26e-2), -5.82 / (self.midi_norm * -8.47e-2)], dtype=torch.float32))
        self.slopes_modifier = nn.Parameter(torch.zeros(2))
        self.offsets_modifier = nn.Parameter(torch.zeros(2))

    def forward(self, extended_pitch: torch.Tensor, global_inharm: torch.Tensor | None = None) -> torch.Tensor:
        reduced = extended_pitch / self.midi_norm
        slopes = self.slopes + self.slopes_modifier
        offsets = self.offsets + self.offsets_modifier
        asym = slopes * (reduced + offsets)
        if global_inharm is not None:
            g = torch.cat([torch.zeros_like(global_inharm), global_inharm * 10.0], dim=-1)
            asym = asym + self.model_specific_weight * g
        return torch.exp(asym).sum(dim=-1, keepdim=True)


class ParametricTuning(InharmonicityNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.reference_a = torch.tensor(69.0)
        self.pitch_translation = 64.0
        self.decrease_slope = 24.0
        self.low_bass_asymptote = 4.51 - 1.0

    def stretching_model(self, notes: torch.Tensor) -> torch.Tensor:
        rho = 1.0 - torch.tanh((notes - self.pitch_translation) / self.decrease_slope)
        return rho * (self.low_bass_asymptote / 2.0) + 1.0

    def get_deviation_from_et(self, notes: torch.Tensor, global_inharm: torch.Tensor | None = None) -> torch.Tensor:
        ref_note = self.reference_a.to(device=notes.device, dtype=notes.dtype)
        ref_inharm = super().forward(ref_note.view(1, 1, 1), global_inharm)
        ratio = midi_to_hz(notes) / midi_to_hz(ref_note)
        num = 1.0 + ref_inharm * (ratio * self.stretching_model(notes)).pow(2)
        den = 1.0 + super().forward(notes, global_inharm) * self.stretching_model(notes).pow(2)
        return torch.sqrt(num / (den + 1e-8))

    def forward(self, extended_pitch: torch.Tensor, global_inharm: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        inharm = super().forward(extended_pitch, global_inharm)
        detune = self.get_deviation_from_et(extended_pitch, global_inharm)
        return midi_to_hz(extended_pitch) * detune, inharm


class JointParametricInharmTuning(nn.Module):
    """Joint per-instrument inharmonicity+tuning model from TF implementation."""

    def __init__(self, n_instruments: int = 10, pretrained_weights: dict[str, list[list[float]]] | None = None) -> None:
        super().__init__()
        self.n_instruments = n_instruments
        self.alpha_b = nn.Embedding(n_instruments, 1)
        self.beta_b = nn.Embedding(n_instruments, 1)
        self.alpha_t = nn.Embedding(n_instruments, 1)
        self.beta_t = nn.Embedding(n_instruments, 1)
        self.pitch_ref = nn.Embedding(n_instruments, 1)
        self.k_param = nn.Embedding(n_instruments, 1)
        self.alpha = nn.Embedding(n_instruments, 1)

        with torch.no_grad():
            for emb in (self.alpha_b, self.beta_b, self.alpha_t, self.beta_t, self.pitch_ref, self.k_param, self.alpha):
                emb.weight.zero_()

            if pretrained_weights is not None:
                self.alpha_b.weight.copy_(torch.tensor(pretrained_weights["alpha_b"], dtype=self.alpha_b.weight.dtype))
                self.beta_b.weight.copy_(torch.tensor(pretrained_weights["beta_b"], dtype=self.beta_b.weight.dtype))
                self.alpha_t.weight.copy_(torch.tensor(pretrained_weights["alpha_t"], dtype=self.alpha_t.weight.dtype))
                self.beta_t.weight.copy_(torch.tensor(pretrained_weights["beta_t"], dtype=self.beta_t.weight.dtype))
                self.pitch_ref.weight.copy_(torch.tensor(pretrained_weights["pitch_ref"], dtype=self.pitch_ref.weight.dtype))
                self.k_param.weight.copy_(torch.tensor(pretrained_weights["K"], dtype=self.k_param.weight.dtype))
                self.alpha.weight.copy_(torch.tensor(pretrained_weights["alpha"], dtype=self.alpha.weight.dtype))

        if pretrained_weights is not None:
            for emb in (self.alpha_b, self.beta_b, self.alpha_t, self.beta_t, self.pitch_ref, self.k_param, self.alpha):
                emb.weight.requires_grad_(False)

    @staticmethod
    def reverse_scaled_tanh(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - torch.tanh(x)) / 2.0

    @staticmethod
    def _piano_index(piano_model: torch.Tensor) -> torch.Tensor:
        if piano_model.dim() > 1:
            piano_model = piano_model[..., 0]
        return piano_model.long()

    def _embed(self, emb: nn.Embedding, piano_model: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        val = emb(self._piano_index(piano_model))
        return val.unsqueeze(1).to(device=target.device, dtype=target.dtype)

    def get_inharm(self, extended_pitch: torch.Tensor, piano_model: torch.Tensor) -> torch.Tensor:
        bass_asymptote = self._embed(self.alpha_b, piano_model, extended_pitch) * extended_pitch + self._embed(
            self.beta_b, piano_model, extended_pitch
        )
        treble_asymptote = self._embed(self.alpha_t, piano_model, extended_pitch) * extended_pitch + self._embed(
            self.beta_t, piano_model, extended_pitch
        )
        return torch.exp(bass_asymptote) + torch.exp(treble_asymptote)

    def get_deviation_from_et(self, extended_pitch: torch.Tensor, piano_model: torch.Tensor) -> torch.Tensor:
        reference_pitch = self._embed(self.pitch_ref, piano_model, extended_pitch)
        ratio = midi_to_hz(extended_pitch) / torch.clamp(midi_to_hz(reference_pitch), min=1e-7)
        rho = 1.0 + self._embed(self.k_param, piano_model, extended_pitch) * self.reverse_scaled_tanh(
            (extended_pitch - reference_pitch) / torch.clamp(self._embed(self.alpha, piano_model, extended_pitch), min=1e-7)
        )
        num = 1.0 + self.get_inharm(reference_pitch, piano_model) * (ratio * rho).pow(2)
        den = 1.0 + self.get_inharm(extended_pitch, piano_model) * rho.pow(2)
        return torch.sqrt(torch.clamp(num / torch.clamp(den, min=1e-7), min=1e-7))

    def forward(self, extended_pitch: torch.Tensor, piano_model: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inharm_coef = self.get_inharm(extended_pitch, piano_model)
        detuning = self.get_deviation_from_et(extended_pitch, piano_model)
        return midi_to_hz(extended_pitch) * detuning, inharm_coef


class Detuner(nn.Module):
    def __init__(self, use_detune: bool = True) -> None:
        super().__init__()
        self.use_detune = use_detune
        self.linear = nn.Linear(1, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, extended_pitch: torch.Tensor, global_detuning: torch.Tensor | None = None) -> torch.Tensor:
        base_hz = midi_to_hz(extended_pitch)
        if not self.use_detune:
            return base_hz
        shift = self.linear(extended_pitch / 128.0)
        if global_detuning is not None:
            shift = shift + global_detuning
        return base_hz * torch.pow(2.0, shift / 12.0)


class DictDetuner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 1)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, extended_pitch: torch.Tensor, global_detuning: torch.Tensor | None = None) -> torch.Tensor:
        midi_idx = torch.clamp(extended_pitch.round().long(), 0, 127).squeeze(-1)
        shift = self.embedding(midi_idx)
        if global_detuning is not None:
            shift = shift + global_detuning
        return midi_to_hz(extended_pitch) * torch.pow(2.0, shift / 12.0)


class DeepDetuner(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, extended_pitch: torch.Tensor, global_detuning: torch.Tensor | None = None) -> torch.Tensor:
        shift = self.net(extended_pitch / 128.0)
        if global_detuning is not None:
            shift = shift + global_detuning
        return midi_to_hz(extended_pitch) * torch.pow(2.0, shift / 12.0)


class F0ProcessorCell(nn.Module):
    """Custom RNN cell that extends pitch during release."""

    def __init__(self, frame_rate: int = 250) -> None:
        super().__init__()
        self.release_duration = nn.Parameter(torch.tensor(1.1), requires_grad=True)
        self.frame_rate = frame_rate

    @staticmethod
    def saturated_relu(x: torch.Tensor, threshold: torch.Tensor | float = 0.0) -> torch.Tensor:
        return torch.clamp(torch.relu(x - threshold), max=1.0)

    def step(self, midi_note: torch.Tensor, prev_note: torch.Tensor, prev_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        note_activity = self.saturated_relu(midi_note, 0.0)
        threshold = self.release_duration.to(device=prev_steps.device, dtype=prev_steps.dtype) * float(self.frame_rate)
        release_end = self.saturated_relu(prev_steps, threshold)
        out_note = note_activity * midi_note + (1.0 - note_activity) * prev_note * (1.0 - release_end)
        steps = (prev_steps + 1.0) * (1.0 - note_activity) * (1.0 - release_end)
        return out_note, out_note, steps


class NoteRelease(nn.Module):
    def __init__(self, frame_rate: int = 250) -> None:
        super().__init__()
        self.cell = F0ProcessorCell(frame_rate=frame_rate)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        active_pitch = conditioning[..., 0:1]
        b, t, n, _ = active_pitch.shape
        outputs = torch.zeros_like(active_pitch)
        prev_note = torch.zeros(b, n, 1, device=active_pitch.device, dtype=active_pitch.dtype)
        prev_steps = torch.zeros_like(prev_note)
        for i in range(t):
            out, prev_note, prev_steps = self.cell.step(active_pitch[:, i], prev_note, prev_steps)
            outputs[:, i] = out
        return outputs


class OnsetLinspaceCell(nn.Module):
    def forward(self, onsets: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = onsets.shape
        out = torch.zeros(b, t, 1, device=onsets.device, dtype=onsets.dtype)
        state = torch.zeros(b, 1, device=onsets.device, dtype=onsets.dtype)
        onset = (onsets[..., 0].amax(dim=-1, keepdim=True) > 0).to(onsets.dtype)
        for i in range(t):
            state = (state + 1.0) * (1.0 - onset[:, i])
            out[:, i] = state
        return out


class SurrogateModule(nn.Module):
    def __init__(self, n_harmonics: int = 96) -> None:
        super().__init__()
        self.n_harmonics = n_harmonics
        self.amp_model = nn.Embedding(128, n_harmonics)
        nn.init.ones_(self.amp_model.weight)
        self.time_model = OnsetLinspaceCell()

    def forward(self, conditioning: torch.Tensor, extended_pitch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        midi_idx = torch.clamp(extended_pitch[..., 0].round().long(), 0, 127)
        decays = self.amp_model(midi_idx)
        decay_time = self.time_model(conditioning)
        return decays, decay_time


class PartialMasking(nn.Module):
    def __init__(self, n_partials: int) -> None:
        super().__init__()
        self.n_partials = n_partials

    def forward(self, harmonic_distribution: torch.Tensor, n_partials: int | None = None) -> torch.Tensor:
        if n_partials is None:
            n_partials = self.n_partials
        if n_partials is None:
            return harmonic_distribution
        idx = torch.arange(harmonic_distribution.shape[-1], device=harmonic_distribution.device).view(1, 1, -1)
        mask = idx < n_partials
        return torch.where(mask, harmonic_distribution, harmonic_distribution.new_full((), -10.0))
