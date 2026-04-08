from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ddsp_piano_pytorch.config import load_yaml_config
from ddsp_piano_pytorch.modules.piano_model import PianoModel
from ddsp_piano_pytorch.train import build_model_from_config
from ddsp_piano_pytorch.utils.io_utils import load_midi_as_conditioning, normalize_audio, save_audio


def load_model(checkpoint: str, config: dict, device: torch.device) -> PianoModel:
    model = build_model_from_config(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def synthesize(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    mcfg = config["model"]
    midi = load_midi_as_conditioning(
        args.midi_file,
        frame_rate=mcfg["frame_rate"],
        sample_rate=mcfg["sample_rate"],
        max_polyphony=mcfg["n_synths"],
        duration=args.duration,
        warm_up_duration=args.warm_up,
    )
    conditioning = midi["conditioning"].to(device)
    pedals = midi["pedals"].to(device)
    onsets = midi["onsets"].to(device)
    piano_model = torch.tensor([args.piano_type], device=device, dtype=torch.long)

    model = load_model(args.ckpt, config, device)
    with torch.no_grad():
        outputs = model(conditioning=conditioning, pedals=pedals, piano_model=piano_model, onsets=onsets)
    audio = outputs["audio"].squeeze(0).detach().cpu()
    trim = int(args.warm_up * mcfg["sample_rate"])
    audio = audio[trim:]
    save_audio(args.out_file, audio, sample_rate=mcfg["sample_rate"])
    if args.normalize is not None:
        normalize_audio(args.out_file, args.normalize)

    if args.unreverbed:
        dry_audio = outputs["audio_dry"].squeeze(0).detach().cpu()[trim:]
        dry_path = str(Path(args.out_file).with_suffix("")) + "_unreverbed.wav"
        save_audio(dry_path, dry_audio, sample_rate=mcfg["sample_rate"])
        if args.normalize is not None:
            normalize_audio(dry_path, args.normalize)

    # Differentiability check.
    cond_req = conditioning.clone().detach().requires_grad_(True)
    out2 = model(conditioning=cond_req, pedals=pedals, piano_model=piano_model, onsets=onsets)
    grad = torch.autograd.grad(out2["audio"].sum(), cond_req, retain_graph=False, allow_unused=True)[0]
    if grad is None:
        raise RuntimeError("Differentiability check failed: conditioning gradient is None.")


def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="ddsp_piano_pytorch/configs/maestro_v2.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--piano_type", type=int, default=9)
    parser.add_argument("-wu", "--warm_up", type=float, default=0.5)
    parser.add_argument("-d", "--duration", type=float, default=None)
    parser.add_argument("-n", "--normalize", type=float, default=None)
    parser.add_argument("-u", "--unreverbed", action="store_true")
    parser.add_argument("midi_file", type=str)
    parser.add_argument("out_file", type=str)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = process_args()
    synthesize(args)


if __name__ == "__main__":
    main()
