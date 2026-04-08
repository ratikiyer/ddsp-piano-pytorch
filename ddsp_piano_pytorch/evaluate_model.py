from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from ddsp_piano_pytorch.config import load_yaml_config
from ddsp_piano_pytorch.data_pipeline import build_dataloader, build_manifest_from_maestro_csv
from ddsp_piano_pytorch.modules.losses import SpectralLoss, SpectralLossConfig
from ddsp_piano_pytorch.modules.piano_model import PianoModel
from ddsp_piano_pytorch.train import build_model_from_config
from ddsp_piano_pytorch.utils.io_utils import save_audio


def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="ddsp_piano_pytorch/configs/maestro_v2.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--warm_up", "-wu", type=float, default=0.5)
    parser.add_argument("--get_wav", "-w", action="store_true")
    parser.add_argument("maestro_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _build_model(cfg: dict, ckpt_path: str, device: torch.device) -> PianoModel:
    model = build_model_from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main(args: argparse.Namespace) -> None:
    cfg = load_yaml_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"
    if args.get_wav:
        wav_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "maestro_test_manifest.csv"
    build_manifest_from_maestro_csv(args.maestro_dir, split="test", out_manifest=manifest)

    loader = build_dataloader(
        manifest=manifest,
        batch_size=1,
        shuffle=False,
        sample_rate=cfg["model"]["sample_rate"],
        frame_rate=cfg["model"]["frame_rate"],
        duration=10.0,
        max_polyphony=cfg["model"]["n_synths"],
    )
    model = _build_model(cfg, args.ckpt, device)
    spec_loss = SpectralLoss(SpectralLossConfig(tuple(cfg["training"].get("fft_sizes", [2048, 1024, 512, 256])))).to(device)

    records = []
    trim = int(args.warm_up * cfg["model"]["sample_rate"])
    for idx, batch in enumerate(loader):
        conditioning = batch["conditioning"].to(device)
        pedals = batch["pedals"].to(device)
        piano_model = batch["piano_model"].to(device)
        target_audio = batch["audio"].to(device)
        onsets = batch["onsets"].to(device)
        with torch.no_grad():
            out = model(conditioning=conditioning, pedals=pedals, piano_model=piano_model, onsets=onsets)
            loss_val = float(spec_loss(target_audio, out["audio"]).item())
        filename = batch["filename"][0]
        records.append({"filename": filename, "piano_model": int(piano_model[0].item()), "loss_val": loss_val})
        if args.get_wav:
            wav_path = wav_dir / f"{filename}.wav"
            save_audio(wav_path, out["audio"].squeeze(0).detach().cpu()[trim:], sample_rate=cfg["model"]["sample_rate"])
        if idx % 100 == 0:
            pd.DataFrame(records).to_csv(out_dir / "spectral_losses.csv", index=False)
    pd.DataFrame(records).to_csv(out_dir / "spectral_losses.csv", index=False)


if __name__ == "__main__":
    main(process_args())
