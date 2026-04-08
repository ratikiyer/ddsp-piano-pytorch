from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi
import torch


@dataclass
class MidiConditioning:
    conditioning: torch.Tensor  # [T, n_synths, 2]
    pedals: torch.Tensor  # [T, 4]
    onsets: torch.Tensor  # [T, n_synths]


def _extract_control(instrument: pretty_midi.Instrument, number: int, times: np.ndarray) -> np.ndarray:
    cc = np.zeros_like(times, dtype=np.float32)
    points = sorted([c for c in instrument.control_changes if c.number == number], key=lambda x: x.time)
    if not points:
        return cc
    idx = 0
    cur = 0.0
    for t_i, t in enumerate(times):
        while idx < len(points) and points[idx].time <= t:
            cur = points[idx].value / 127.0
            idx += 1
        cc[t_i] = cur
    return cc


def piano_roll_to_conditioning(
    piano_roll: np.ndarray,
    frame_rate: int = 250,
    max_polyphony: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert [128, T] pianoroll to [T, n_synths, 2] pitch/velocity."""
    t_frames = piano_roll.shape[1]
    conditioning = np.zeros((t_frames, max_polyphony, 2), dtype=np.float32)
    onsets = np.zeros((t_frames, max_polyphony), dtype=np.float32)
    prev_active: set[int] = set()
    for t in range(t_frames):
        active = np.where(piano_roll[:, t] > 0)[0]
        velocities = piano_roll[active, t]
        order = np.argsort(-velocities)
        active = active[order][:max_polyphony]
        velocities = velocities[order][:max_polyphony]
        for i, (pitch, vel) in enumerate(zip(active, velocities)):
            conditioning[t, i, 0] = float(pitch)
            conditioning[t, i, 1] = float(vel) / 127.0
            if int(pitch) not in prev_active:
                onsets[t, i] = 1.0
        prev_active = {int(p) for p in active}
    return conditioning, onsets


def midi_to_conditioning(
    midi_path: str | Path,
    frame_rate: int = 250,
    sample_rate: int = 24000,
    max_polyphony: int = 16,
) -> MidiConditioning:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    if not midi.instruments:
        raise ValueError("MIDI has no instruments.")
    piano = midi.instruments[0]
    end_t = max((n.end for n in piano.notes), default=0.0)
    n_frames = max(1, int(np.ceil(end_t * frame_rate)))
    times = np.arange(n_frames) / frame_rate
    roll = piano.get_piano_roll(fs=frame_rate)
    if roll.shape[1] < n_frames:
        roll = np.pad(roll, ((0, 0), (0, n_frames - roll.shape[1])))
    roll = roll[:, :n_frames]
    conditioning_np, onsets_np = piano_roll_to_conditioning(roll, frame_rate=frame_rate, max_polyphony=max_polyphony)
    sustain = _extract_control(piano, 64, times)
    sostenuto = _extract_control(piano, 66, times)
    una_corda = _extract_control(piano, 67, times)
    key_count = (roll > 0).sum(axis=0).astype(np.float32) / max(1.0, float(max_polyphony))
    pedals_np = np.stack([sustain, sostenuto, una_corda, key_count], axis=-1)

    return MidiConditioning(
        conditioning=torch.from_numpy(conditioning_np),
        pedals=torch.from_numpy(pedals_np),
        onsets=torch.from_numpy(onsets_np),
    )
