from __future__ import annotations

from pathlib import Path
import gc
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ddsp_piano_pytorch.utils.midi_utils import midi_to_conditioning


def load_audio(path: str | Path, sample_rate: int = 24000, mono: bool = True) -> torch.Tensor:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 1:
        wav = torch.from_numpy(audio).unsqueeze(0).float()
    else:
        wav = torch.from_numpy(audio).T.float()
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0)


def save_audio(path: str | Path, audio: torch.Tensor, sample_rate: int = 24000) -> None:
    audio_np = audio.detach().cpu().numpy()
    sf.write(str(path), audio_np, sample_rate)


def normalize_audio(path: str | Path, target_dbfs: float) -> None:
    """Normalize an audio file to target dBFS in-place."""
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        peak = np.max(np.abs(audio), axis=0).max()
    else:
        peak = np.max(np.abs(audio))
    if peak <= 1e-9:
        return
    target_amp = 10.0 ** (target_dbfs / 20.0)
    gain = target_amp / peak
    sf.write(str(path), audio * gain, sr)


def load_midi_as_conditioning(
    midi_path: str | Path,
    frame_rate: int = 250,
    sample_rate: int = 24000,
    max_polyphony: int = 16,
    duration: float | None = None,
    warm_up_duration: float = 0.0,
) -> dict[str, torch.Tensor | float]:
    midi = midi_to_conditioning(midi_path, frame_rate=frame_rate, sample_rate=sample_rate, max_polyphony=max_polyphony)
    conditioning = midi.conditioning
    pedals = midi.pedals
    onsets = midi.onsets
    if duration is not None:
        n_frames = int(duration * frame_rate)
        conditioning = conditioning[:n_frames]
        pedals = pedals[:n_frames]
        onsets = onsets[:n_frames]
    total_duration = conditioning.shape[0] / float(frame_rate)
    return {
        "conditioning": conditioning.unsqueeze(0),
        "pedals": pedals.unsqueeze(0),
        "onsets": onsets.unsqueeze(0),
        "duration": total_duration + warm_up_duration,
    }


def collect_garbage() -> None:
    gc.collect()
