from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ddsp_piano_pytorch.utils.io_utils import load_audio
from ddsp_piano_pytorch.utils.midi_utils import midi_to_conditioning


@dataclass
class MaestroExample:
    audio_path: str
    midi_path: str
    piano_model: int


def _read_manifest(manifest_path: str | Path) -> list[MaestroExample]:
    manifest_path = Path(manifest_path)
    if manifest_path.suffix == ".json":
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    examples: list[MaestroExample] = []
    for row in rows:
        examples.append(
            MaestroExample(
                audio_path=row["audio_path"],
                midi_path=row["midi_path"],
                piano_model=int(row.get("piano_model", 0)),
            )
        )
    return examples


def build_manifest_from_maestro_csv(
    maestro_dir: str | Path,
    split: str,
    out_manifest: str | Path | None = None,
) -> list[MaestroExample]:
    """Create manifest entries from MAESTRO metadata CSV.

    This mirrors the role of TF preprocessing entrypoints but stores a simple
    CSV/JSON manifest consumed by the PyTorch Dataset.
    """
    maestro_dir = Path(maestro_dir)
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find MAESTRO metadata CSV: {csv_path}")
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    split_rows = [r for r in rows if r.get("split", "").strip().lower() == split.lower()]
    years = sorted({int(r["year"]) for r in split_rows})
    year_to_id = {y: i for i, y in enumerate(years)}

    examples: list[MaestroExample] = []
    for row in split_rows:
        audio_path = maestro_dir / row["audio_filename"]
        midi_path = maestro_dir / row["midi_filename"]
        examples.append(
            MaestroExample(
                audio_path=str(audio_path),
                midi_path=str(midi_path),
                piano_model=year_to_id[int(row["year"])],
            )
        )

    if out_manifest is not None:
        out_manifest = Path(out_manifest)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"audio_path": ex.audio_path, "midi_path": ex.midi_path, "piano_model": ex.piano_model}
            for ex in examples
        ]
        if out_manifest.suffix == ".json":
            out_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            with open(out_manifest, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["audio_path", "midi_path", "piano_model"])
                writer.writeheader()
                writer.writerows(payload)
    return examples


class MaestroDataset(Dataset):
    """PyTorch MAESTRO dataset loader mirroring TF pipeline behavior."""

    def __init__(
        self,
        manifest: str | Path,
        sample_rate: int = 24000,
        frame_rate: int = 250,
        duration: float = 3.0,
        max_polyphony: int = 16,
    ) -> None:
        self.examples = _read_manifest(manifest)
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.duration = duration
        self.max_polyphony = max_polyphony
        self.n_samples = int(sample_rate * duration)
        self.n_frames = int(frame_rate * duration)

    def __len__(self) -> int:
        return len(self.examples)

    def _pad_or_trim(self, x: torch.Tensor, length: int) -> torch.Tensor:
        if x.shape[0] > length:
            return x[:length]
        if x.shape[0] < length:
            pad_t = length - x.shape[0]
            if x.dim() == 3:
                # [T, N, F]
                return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_t))
            if x.dim() == 2:
                # [T, F]
                return torch.nn.functional.pad(x, (0, 0, 0, pad_t))
            return torch.nn.functional.pad(x, (0, pad_t))
        return x

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.examples[idx]
        audio = load_audio(item.audio_path, sample_rate=self.sample_rate, mono=True)
        if audio.shape[0] < self.n_samples:
            audio = torch.nn.functional.pad(audio, (0, self.n_samples - audio.shape[0]))
        start = 0 if audio.shape[0] == self.n_samples else np.random.randint(0, audio.shape[0] - self.n_samples + 1)
        audio = audio[start : start + self.n_samples]

        midi = midi_to_conditioning(item.midi_path, frame_rate=self.frame_rate, sample_rate=self.sample_rate, max_polyphony=self.max_polyphony)
        cond = self._pad_or_trim(midi.conditioning, self.n_frames)
        pedals = self._pad_or_trim(midi.pedals, self.n_frames)
        onsets = self._pad_or_trim(midi.onsets, self.n_frames)

        return {
            "audio": audio.float(),
            "conditioning": cond.float(),
            "pedals": pedals.float(),
            "onsets": onsets.float(),
            "piano_model": torch.tensor(item.piano_model, dtype=torch.long),
            "filename": Path(item.audio_path).stem,
        }


def build_dataloader(
    manifest: str | Path,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    ds = MaestroDataset(manifest=manifest, **dataset_kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_dummy_data(
    batch_size: int = 6,
    duration: float = 3.0,
    sample_rate: int = 24000,
    frame_rate: int = 250,
    n_synths: int = 16,
) -> dict[str, torch.Tensor]:
    n_frames = int(duration * frame_rate)
    n_samples = int(duration * sample_rate)
    pedals = torch.rand(batch_size, n_frames, 4)
    return {
        "conditioning": torch.rand(batch_size, n_frames, n_synths, 2),
        "pedals": pedals,
        "pedal": pedals,
        "audio": torch.rand(batch_size, n_samples),
        "piano_model": torch.randint(0, 10, (batch_size,), dtype=torch.long),
        "onsets": torch.zeros(batch_size, n_frames, n_synths),
        "filename": ["dummy"] * batch_size,
    }


def get_training_dataset(*args, **kwargs) -> DataLoader:
    kwargs.setdefault("shuffle", True)
    return build_dataloader(*args, **kwargs)


def get_validation_dataset(*args, **kwargs) -> DataLoader:
    kwargs.setdefault("shuffle", False)
    return build_dataloader(*args, **kwargs)


def get_test_dataset(*args, **kwargs) -> DataLoader:
    kwargs.setdefault("shuffle", False)
    return build_dataloader(*args, **kwargs)


def preprocess_data_into_manifest(
    filename: str | Path,
    dataset_dir: str | Path,
    split: str = "train",
    **kwargs,
) -> None:
    """Precompute a CSV/JSON manifest from MAESTRO metadata."""
    _ = kwargs
    build_manifest_from_maestro_csv(dataset_dir, split=split, out_manifest=filename)


def preprocess_data_into_tfrecord(
    filename: str | Path,
    dataset_dir: str | Path,
    split: str = "train",
    **kwargs,
) -> None:
    """Backward-compatible alias for TF naming.

    The PyTorch pipeline stores manifest files, not TFRecord datasets. If a
    `.tfrecord` path is provided, we replace the suffix with `.csv`.
    """
    out_path = Path(filename)
    if out_path.suffix == ".tfrecord":
        out_path = out_path.with_suffix(".csv")
    preprocess_data_into_manifest(out_path, dataset_dir=dataset_dir, split=split, **kwargs)
