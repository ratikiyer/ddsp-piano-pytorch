# TensorFlow-to-PyTorch Feature Matrix

Canonical TF reference: [lrenault/ddsp-piano](https://github.com/lrenault/ddsp-piano?tab=readme-ov-file)

## Modules

| TensorFlow source | PyTorch counterpart | Status | Notes |
|---|---|---|---|
| `ddsp_piano/modules/piano_model.py` | `ddsp_piano_pytorch/modules/piano_model.py` | Implemented | Top-level orchestration and alternating training hooks. |
| `ddsp_piano/modules/sub_modules.py` | `ddsp_piano_pytorch/modules/sub_modules.py` | Implemented | Core context, z-encoder, monophonic, tuning, release, utility cells. |
| `ddsp_piano/modules/inharm_synth.py` | `ddsp_piano_pytorch/modules/inharm_synth.py` | Implemented | Inharmonic and multi-inharmonic additive synths. |
| `ddsp_piano/modules/filtered_noise_synth.py` | `ddsp_piano_pytorch/modules/filtered_noise.py` | Implemented | Dynamic filtered noise + NoiseBandNet-style synth. |
| `ddsp_piano/modules/fdn_reverb.py` | `ddsp_piano_pytorch/modules/fdn_reverb.py` | Implemented | Householder FDN reverb in frequency domain. |
| `ddsp_piano/modules/losses.py` | `ddsp_piano_pytorch/modules/losses.py` | Implemented | Spectral + inharmonic + reverb regularization losses. |
| `ddsp_piano/modules/polyphonic_dag.py` | `ddsp_piano_pytorch/modules/processor_group.py` | Implemented | Ordered processor DAG execution and signal routing. |

## Scripts / Workflows

| TensorFlow script | PyTorch script | Status | Notes |
|---|---|---|---|
| `train_single_phase.py` | `ddsp_piano_pytorch/train.py` | Implemented (aligned CLI) | Supports TF-style phase args and single-phase training behavior. |
| `synthesize_midi_file.py` | `ddsp_piano_pytorch/synthesize.py` | Implemented (aligned CLI) | Supports TF-style options (`--ckpt`, `--piano_type`, warm-up, normalize, unreverbed). |
| `evaluate_model.py` | `ddsp_piano_pytorch/evaluate_model.py` | Implemented | MAESTRO-style evaluation loop and optional wav export. |
| `preprocess_maestro.py` | `ddsp_piano_pytorch/preprocess_maestro.py` | Implemented | Produces serialized train/validation manifests for PyTorch pipeline. |

## Config System

| TensorFlow | PyTorch | Status |
|---|---|---|
| Gin configs (`ddsp_piano/configs/*.gin`) | YAML configs (`ddsp_piano_pytorch/configs/*.yaml`) | Implemented |

## Data Pipeline

| TensorFlow | PyTorch | Status | Notes |
|---|---|---|---|
| `ddsp_piano/data_pipeline.py` | `ddsp_piano_pytorch/data_pipeline.py` | Implemented | Dataset/DataLoader and preprocessing support. |

## Conversion / Equivalence

| Requirement | PyTorch implementation | Status |
|---|---|---|
| TF checkpoint import | `ddsp_piano_pytorch/convert_weights.py` | Implemented |
| Dense kernel transpose | `convert_weights.py` | Implemented |
| GRU gate reorder (TF z/r/h -> PT r/z/h) | `convert_weights.py` | Implemented |
| Public API parity tests | `tests/*` | Implemented and expanded |

## Remaining validation focus

1. End-to-end TF checkpoint golden parity on full model outputs and key intermediates.
2. Multi-step training evidence on real MAESTRO subset with produced checkpoints/audio.
3. Expanded parity assertions for script argument semantics and edge-case behaviors.
