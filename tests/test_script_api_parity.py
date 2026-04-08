import csv
import sys
from pathlib import Path

import torch

from ddsp_piano_pytorch import data_pipeline
from ddsp_piano_pytorch.evaluate_model import process_args as eval_process_args
from ddsp_piano_pytorch.preprocess_maestro import process_args as preprocess_process_args
from ddsp_piano_pytorch.synthesize import process_args as synth_process_args
from ddsp_piano_pytorch.train import process_args as train_process_args


def test_train_single_phase_cli_compat(monkeypatch) -> None:
    argv = [
        "train_single_phase.py",
        "--n_gpus",
        "1",
        "--batch_size",
        "2",
        "--steps_per_epoch",
        "3",
        "--epochs",
        "1",
        "--lr",
        "0.001",
        "--phase",
        "1",
        "--config",
        "ddsp_piano_pytorch/configs/maestro_v2.yaml",
        "maestro_dir",
        "exp_dir",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = train_process_args()
    assert args.maestro_path == "maestro_dir"
    assert args.exp_dir == "exp_dir"
    assert args.phase == 1


def test_synthesize_cli_compat(monkeypatch) -> None:
    argv = [
        "synthesize_midi_file.py",
        "--config",
        "ddsp_piano_pytorch/configs/maestro_v2.yaml",
        "--ckpt",
        "ckpt.pt",
        "--piano_type",
        "9",
        "--warm_up",
        "0.5",
        "in.mid",
        "out.wav",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = synth_process_args()
    assert args.ckpt == "ckpt.pt"
    assert args.midi_file == "in.mid"
    assert args.out_file == "out.wav"
    assert args.piano_type == 9


def test_evaluate_and_preprocess_cli_compat(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_model.py", "--config", "ddsp_piano_pytorch/configs/maestro_v2.yaml", "--ckpt", "ckpt.pt", "maestro", "out"],
    )
    eargs = eval_process_args()
    assert eargs.maestro_dir == "maestro"
    assert eargs.out_dir == "out"

    monkeypatch.setattr(sys, "argv", ["preprocess_maestro.py", "maestro", "out"])
    pargs = preprocess_process_args()
    assert pargs.maestro_dir == "maestro"
    assert pargs.out_dir == "out"


def test_preprocess_manifest_generation_compat(tmp_path: Path) -> None:
    maestro_dir = tmp_path / "maestro-v3.0.0"
    maestro_dir.mkdir()
    (maestro_dir / "2004").mkdir()
    # Minimal metadata for both train and validation split.
    rows = [
        {"canonical_composer": "a", "canonical_title": "b", "split": "train", "year": "2004", "midi_filename": "2004/x.mid", "audio_filename": "2004/x.wav", "duration": "1.0"},
        {"canonical_composer": "a", "canonical_title": "b", "split": "validation", "year": "2004", "midi_filename": "2004/y.mid", "audio_filename": "2004/y.wav", "duration": "1.0"},
    ]
    with open(maestro_dir / "maestro-v3.0.0.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_manifest = tmp_path / "maestro_train.tfrecord"
    data_pipeline.preprocess_data_into_tfrecord(out_manifest, dataset_dir=maestro_dir, split="train")
    csv_manifest = out_manifest.with_suffix(".csv")
    assert csv_manifest.exists()
    content = csv_manifest.read_text(encoding="utf-8")
    assert "audio_path" in content


def test_dummy_data_contract() -> None:
    dummy = data_pipeline.get_dummy_data(batch_size=2, duration=1.0, sample_rate=8000, frame_rate=100, n_synths=16)
    assert dummy["conditioning"].shape == (2, 100, 16, 2)
    assert dummy["pedals"].shape == (2, 100, 4)
    assert dummy["audio"].shape == (2, 8000)
    assert dummy["piano_model"].dtype == torch.long
