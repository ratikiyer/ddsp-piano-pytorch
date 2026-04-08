# DDSP-Piano-PyTorch

PyTorch port of DDSP-Piano with:

- Maestro-v2 focused model architecture
- Trainable end-to-end MIDI-to-audio synthesis
- YAML-based configuration (replacement for gin)
- TF checkpoint conversion utility
- Inference script with differentiability check

## Quick start

This project uses `uv` as the package manager and environment tool.

Install + sync dependencies:

```bash
uv sync --group dev
```

Run tests (TDD/parity suite):

```bash
uv run pytest
```

Run TF parity extensions (optional, heavier deps):

```bash
uv sync --group dev --group parity
uv run pytest -k parity
```

Train:

```bash
uv run python -m ddsp_piano_pytorch.train_single_phase \
  --config ddsp_piano_pytorch/configs/maestro_v2.yaml \
  <path/to/maestro-v3.0.0/> \
  <experiment-directory/>
```

Synthesize:

```bash
uv run python -m ddsp_piano_pytorch.synthesize_midi_file \
  --config ddsp_piano_pytorch/configs/maestro_v2.yaml \
  --ckpt checkpoints/phase_1/last_iter.pt \
  input.mid output.wav
```

Evaluate:

```bash
uv run python -m ddsp_piano_pytorch.evaluate_model \
  --config ddsp_piano_pytorch/configs/maestro_v2.yaml \
  --ckpt checkpoints/phase_1/last_iter.pt \
  <path/to/maestro-v3.0.0/> \
  <out-dir/>
```

Preprocess manifests:

```bash
uv run python -m ddsp_piano_pytorch.preprocess_maestro \
  <path/to/maestro-v3.0.0/> \
  <out-dir/>
```
