# DDSP-Piano-PyTorch

PyTorch conversion of the TensorFlow project [lrenault/ddsp-piano](https://github.com/lrenault/ddsp-piano), focused on a trainable DDSP-Piano pipeline and easier integration in PyTorch research workflows.

## What this repository provides

- Full PyTorch package: `ddsp_piano_pytorch/`
- Trainable model pipeline (MAESTRO-style workflow)
- TensorFlow-like script entrypoints:
  - `train_single_phase`
  - `synthesize_midi_file`
  - `evaluate_model`
  - `preprocess_maestro`
- YAML configs (replacing Gin)
- TF checkpoint conversion utility
- Parity/API tests to validate key behaviors

---

## Installation (uv)

This project uses `uv` for dependency and environment management.

```bash
uv sync --group dev
```

Optional (only needed for TF checkpoint conversion / TF-side parity checks):

```bash
uv sync --group dev --group parity
```

Run tests:

```bash
uv run pytest
```

---

## Expected dataset layout (MAESTRO)

Point commands to a MAESTRO root directory containing:

- `maestro-v3.0.0.csv`
- relative audio and MIDI files referenced by that CSV

Example:

```text
maestro-v3.0.0/
  maestro-v3.0.0.csv
  2004/...
  2006/...
  ...
```

---

## End-to-end usage

## 1) Preprocess metadata/manifests

This command mirrors the TensorFlow script name, but stores PyTorch-friendly manifest files.

```bash
uv run python -m ddsp_piano_pytorch.preprocess_maestro \
  <path/to/maestro-v3.0.0> \
  <out-dir>
```

## 2) Train (single-phase)

```bash
uv run python -m ddsp_piano_pytorch.train_single_phase \
  --config ddsp_piano_pytorch/configs/maestro_v2.yaml \
  --batch_size 6 \
  --steps_per_epoch 5000 \
  --epochs 128 \
  --lr 1e-3 \
  --phase 1 \
  <path/to/maestro-v3.0.0> \
  <experiment-dir>
```

Key outputs:

- `<experiment-dir>/phase_1/ckpts/epoch_XXXX.pt`
- `<experiment-dir>/phase_1/last_iter.pt`
- `<experiment-dir>/phase_1/logs/`

## 3) Synthesize from MIDI

```bash
uv run python -m ddsp_piano_pytorch.synthesize_midi_file \
  --config ddsp_piano_pytorch/configs/maestro_v2.yaml \
  --ckpt <experiment-dir>/phase_1/last_iter.pt \
  --piano_type 0 \
  input.mid output.wav
```

Useful options:

- `--warm_up` (seconds)
- `--duration` (max synthesized duration)
- `--normalize` (target dBFS)
- `--unreverbed` (also save dry audio)

## 4) Evaluate on test split

```bash
uv run python -m ddsp_piano_pytorch.evaluate_model \
  --config ddsp_piano_pytorch/configs/maestro_v2.yaml \
  --ckpt <experiment-dir>/phase_1/last_iter.pt \
  <path/to/maestro-v3.0.0> \
  <eval-out-dir>
```

Output:

- `<eval-out-dir>/spectral_losses.csv`
- optional wavs if `--get_wav`

---

## TensorFlow to PyTorch migration guide

This section explains what changed from the original TensorFlow repo and how to adapt integrations.

## Script-level mapping

- TF `train_single_phase.py` -> PT `python -m ddsp_piano_pytorch.train_single_phase`
- TF `synthesize_midi_file.py` -> PT `python -m ddsp_piano_pytorch.synthesize_midi_file`
- TF `evaluate_model.py` -> PT `python -m ddsp_piano_pytorch.evaluate_model`
- TF `preprocess_maestro.py` -> PT `python -m ddsp_piano_pytorch.preprocess_maestro`

## Config mapping

- TF `.gin` -> PT `.yaml`
- Main config file: `ddsp_piano_pytorch/configs/maestro_v2.yaml`

If you previously used Gin bindings to override runtime parameters, now pass equivalent settings via:

1. YAML values, and/or
2. CLI args for script-level behavior (e.g. phase, batch size, warm-up).

## Module/function mapping

- TF `ddsp_piano/modules/piano_model.py` -> PT `ddsp_piano_pytorch/modules/piano_model.py`
- TF `sub_modules.py` -> PT `sub_modules.py`
- TF `inharm_synth.py` -> PT `inharm_synth.py`
- TF `filtered_noise_synth.py` -> PT `filtered_noise.py`
- TF `fdn_reverb.py` -> PT `fdn_reverb.py`
- TF `losses.py` -> PT `losses.py`
- TF processor DAG utilities -> PT `processor_group.py`

## Important behavioral differences

- Data preprocessing output format:
  - TF script name uses `tfrecord` wording.
  - PT pipeline stores manifest-style files consumed by `MaestroDataset`.
- Serialization:
  - TF checkpoints (`.ckpt`) vs PT checkpoints (`.pt` with model/optimizer dicts).
- Runtime framework:
  - No `gin` injection or TF strategy wrappers.
  - Standard PyTorch module construction + DataLoader training loop.

---

## TF checkpoint conversion

Use:

```bash
uv run python -m ddsp_piano_pytorch.convert_weights \
  --tf_checkpoint <tf_ckpt_path> \
  --pt_template <pt_template_ckpt_or_state_dict> \
  --name_map <torch_saved_mapping_dict.pt> \
  --output converted.pt
```

Implemented conversions include:

- Dense kernel transpose `[in, out] -> [out, in]`
- GRU gate order swap `TF[z,r,h] -> PT[r,z,h]`

---

## Integrating into your research pipeline

For differentiable forward-model use (MIDI -> audio):

1. Construct `PianoModel` in PyTorch.
2. Feed tensors with shapes:
   - `conditioning`: `[B, T, n_synths, 2]`
   - `pedals`: `[B, T, 4]`
   - `piano_model`: `[B]` (instrument id)
3. Use `outputs["audio"]` as forward-model output.
4. Backprop through the audio objective as usual (`loss.backward()`).

The synthesis path is designed to keep gradients available from output audio back to conditioning and model parameters.

---

## Validation checklist before long training runs

- `uv run pytest` passes
- A short train run writes checkpoint/log files
- MIDI synthesis command produces expected audio
- Evaluation command writes `spectral_losses.csv`

If all four pass, the repo is ready for full MAESTRO-scale experiments.
