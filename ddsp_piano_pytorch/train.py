from __future__ import annotations

import argparse
import inspect
import json
import os
from itertools import cycle
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from ddsp_piano_pytorch.config import load_yaml_config
from ddsp_piano_pytorch.data_pipeline import build_dataloader, build_manifest_from_maestro_csv
from ddsp_piano_pytorch.modules.losses import InharmonicityLoss, ReverbRegularizer, SpectralLoss, SpectralLossConfig
from ddsp_piano_pytorch.modules.piano_model import PianoModel
from ddsp_piano_pytorch.modules.sub_modules import (
    BackgroundNoiseFilter,
    DeepDetuner,
    Detuner,
    DictDetuner,
    FiLMContextNetwork,
    InharmonicityNetwork,
    JointParametricInharmTuning,
    MonophonicDeepNetwork,
    MonophonicNetwork,
    MultiInstrumentFeedbackDelayReverb,
    NoteRelease,
    OneHotZEncoder,
    Parallelizer,
    SimpleContextNet,
)


_CLASS_REGISTRY = {
    "OneHotZEncoder": OneHotZEncoder,
    "NoteRelease": NoteRelease,
    "FiLMContextNetwork": FiLMContextNetwork,
    "SimpleContextNet": SimpleContextNet,
    "Parallelizer": Parallelizer,
    "MonophonicNetwork": MonophonicNetwork,
    "MonophonicDeepNetwork": MonophonicDeepNetwork,
    "InharmonicityNetwork": InharmonicityNetwork,
    "JointParametricInharmTuning": JointParametricInharmTuning,
    "Detuner": Detuner,
    "DictDetuner": DictDetuner,
    "DeepDetuner": DeepDetuner,
    "BackgroundNoiseFilter": BackgroundNoiseFilter,
    "MultiInstrumentFeedbackDelayReverb": MultiInstrumentFeedbackDelayReverb,
}


def _build_component(spec: dict | None, fallback: dict | None = None):
    if not spec:
        return None
    class_name = spec.get("class")
    if class_name not in _CLASS_REGISTRY:
        raise ValueError(f"Unknown model component class: {class_name}")
    kwargs = dict(fallback or {})
    kwargs.update({k: v for k, v in spec.items() if k != "class"})
    cls = _CLASS_REGISTRY[class_name]
    valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return cls(**filtered_kwargs)


def build_model_from_config(cfg: dict) -> PianoModel:
    m = cfg["model"]
    mono_output_splits = (
        ("amplitudes", 1),
        ("harmonic_distribution", m["n_harmonics"]),
        ("noise_magnitudes", m["n_noise_bands"]),
    )

    z_encoder = _build_component(m.get("z_encoder"), fallback={"frame_rate": m["frame_rate"]})
    note_release = _build_component(m.get("note_release"), fallback={"frame_rate": m["frame_rate"]})
    context_network = _build_component(
        m.get("context_network"),
        fallback={
            "n_synths": m["n_synths"],
            "context_dim": m.get("context_dim", 128),
            "n_instruments": m.get("n_instruments", 10),
        },
    )
    parallelizer = _build_component(m.get("parallelizer"), fallback={"n_synths": m["n_synths"]})
    mono_spec = m.get("monophonic_network")
    mono_fallback = {"output_splits": mono_output_splits}
    if mono_spec and mono_spec.get("class") == "MonophonicDeepNetwork":
        mono_fallback["context_dim"] = m.get("context_dim", 128)
    elif mono_spec and mono_spec.get("class") == "MonophonicNetwork":
        mono_fallback["input_dim"] = m.get("context_dim", 128) + 3
    monophonic_network = _build_component(mono_spec, fallback=mono_fallback)
    inharm_model = _build_component(m.get("inharm_model"))
    detuner = _build_component(m.get("detuner"))
    background_noise_model = _build_component(
        m.get("background_noise_model"),
        fallback={"n_filters": m["n_noise_bands"], "frame_rate": m["frame_rate"]},
    )
    reverb_model = _build_component(m.get("reverb_model"), fallback={"sample_rate": m["sample_rate"]})

    return PianoModel(
        n_synths=m["n_synths"],
        n_harmonics=m["n_harmonics"],
        n_noise_bands=m["n_noise_bands"],
        sample_rate=m["sample_rate"],
        frame_rate=m["frame_rate"],
        context_dim=m.get("context_dim", 128),
        z_encoder=z_encoder,
        note_release=note_release,
        context_network=context_network,
        parallelizer=parallelizer,
        monophonic_network=monophonic_network,
        inharm_model=inharm_model,
        detuner=detuner,
        background_noise_model=background_noise_model,
        reverb_model=reverb_model,
    )


def _ensure_manifest(maestro_path: str, split: str, out_dir: Path) -> str:
    in_path = Path(maestro_path)
    if in_path.is_file() and in_path.suffix in {".csv", ".json"}:
        return str(in_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"maestro_{split}.csv"
    if not manifest.exists():
        build_manifest_from_maestro_csv(in_path, split=split, out_manifest=manifest)
    return str(manifest)


def train(args: argparse.Namespace) -> None:
    cfg = load_yaml_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = build_model_from_config(cfg).to(device)
    tcfg = cfg["training"]

    exp_dir = Path(args.exp_dir)
    phase_dir = exp_dir / f"phase_{args.phase}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = phase_dir / "manifests"
    train_manifest = _ensure_manifest(args.maestro_path, split="train", out_dir=manifest_dir)
    val_source = args.val_path if args.val_path is not None else args.maestro_path
    val_manifest = _ensure_manifest(val_source, split="validation", out_dir=manifest_dir)

    loader = build_dataloader(
        manifest=train_manifest,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        sample_rate=cfg["model"]["sample_rate"],
        frame_rate=cfg["model"]["frame_rate"],
        duration=tcfg.get("duration", 3.0),
        max_polyphony=cfg["model"]["n_synths"],
    )
    val_loader = build_dataloader(
        manifest=val_manifest,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=tcfg.get("num_workers", 4),
        sample_rate=cfg["model"]["sample_rate"],
        frame_rate=cfg["model"]["frame_rate"],
        duration=tcfg.get("duration", 3.0),
        max_polyphony=cfg["model"]["n_synths"],
    )

    spec_loss = SpectralLoss(SpectralLossConfig(tuple(tcfg.get("fft_sizes", [2048, 1024, 512, 256])), overlap=tcfg.get("fft_overlap", 0.75))).to(device)
    inharm_loss = InharmonicityLoss(weight=tcfg.get("inharmonicity_weight", 10.0)).to(device)
    reverb_reg = ReverbRegularizer(weight=tcfg.get("reverb_weight", 0.01), loss_type=tcfg.get("reverb_loss_type", "L1")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    writer = SummaryWriter(log_dir=str(phase_dir / "logs"))

    ckpt_dir = phase_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    first_phase = ((args.phase % 2) == 1)
    model.alternate_training(first_phase=first_phase)

    if args.restore:
        ckpt = torch.load(args.restore, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0

    train_iterator = cycle(loader)
    for epoch in range(int(args.epochs)):
        model.train()
        running = {"total": 0.0, "spec": 0.0, "inharm": 0.0, "reverb": 0.0}
        for _ in range(int(args.steps_per_epoch)):
            batch = next(train_iterator)
            conditioning = batch["conditioning"].to(device)
            pedals = batch["pedals"].to(device)
            piano_model = batch["piano_model"].to(device)
            target_audio = batch["audio"].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(conditioning=conditioning, pedals=pedals, piano_model=piano_model, onsets=batch["onsets"].to(device))
            loss_spec = spec_loss(target_audio, out["audio"])
            loss_inharm = inharm_loss(out["inharm_coef"])
            loss_reverb = reverb_reg(out["reverb_ir"])
            loss = loss_spec + loss_inharm + loss_reverb
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=float(tcfg.get("grad_clip", 1.0)))
            optimizer.step()
            running["total"] += float(loss.item())
            running["spec"] += float(loss_spec.item())
            running["inharm"] += float(loss_inharm.item())
            running["reverb"] += float(loss_reverb.item())

            if global_step % int(tcfg.get("log_every", 20)) == 0:
                writer.add_scalar("loss/total", float(loss.item()), global_step)
                writer.add_scalar("loss/spectral", float(loss_spec.item()), global_step)
                writer.add_scalar("loss/inharm", float(loss_inharm.item()), global_step)
                writer.add_scalar("loss/reverb", float(loss_reverb.item()), global_step)
            global_step += 1

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_steps = 0
            for val_batch in val_loader:
                val_out = model(
                    conditioning=val_batch["conditioning"].to(device),
                    pedals=val_batch["pedals"].to(device),
                    piano_model=val_batch["piano_model"].to(device),
                    onsets=val_batch["onsets"].to(device),
                )
                val_loss += float(spec_loss(val_batch["audio"].to(device), val_out["audio"]).item())
                val_steps += 1
            val_loss /= max(1, val_steps)
            writer.add_scalar("val/audio_stft_loss", val_loss, global_step)

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch:04d}.pt")
        torch.save(ckpt, phase_dir / "last_iter.pt")

    (phase_dir / "training_meta.json").write_text(
        json.dumps({"steps": global_step, "epochs": int(args.epochs), "phase": int(args.phase)}, indent=2),
        encoding="utf-8",
    )
    writer.close()


def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", "-gpu", type=int, default=1)
    parser.add_argument("--batch_size", "-b", type=int, default=6)
    parser.add_argument("--steps_per_epoch", "-s", type=int, default=5000)
    parser.add_argument("--epochs", "-e", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--config", "-c", type=str, default="ddsp_piano_pytorch/configs/maestro_v2.yaml")
    parser.add_argument("--phase", "-p", type=int, default=1)
    parser.add_argument("--restore", "-r", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("maestro_path", type=str)
    parser.add_argument("exp_dir", type=str)
    return parser.parse_args()


def main() -> None:
    args = process_args()
    # Keep parity with TF CLI while not hard-locking GPUs here.
    _ = args.n_gpus
    train(args)


if __name__ == "__main__":
    main()
