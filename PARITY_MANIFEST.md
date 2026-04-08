# DDSP-Piano Public API Parity Manifest

This file defines the expected parity surface between:

- TensorFlow reference: `/Users/ratikiyer/Programming/research/ddsp-piano`
- PyTorch port: `/Users/ratikiyer/Programming/research/DDSP-Piano-PyTorch`

## 1) Required module export surface

The PyTorch package must provide equivalents for these TensorFlow public symbols:

- `PianoModel`
- `sub_modules`
- `inharm_synth`
- `surrogate_synth` equivalent behaviors exposed by PyTorch submodules
- `filtered_noise_synth` equivalent behaviors (`DynamicSizeFilteredNoise`, `NoiseBandNetSynth`)
- `polyphonic_dag` equivalent behavior through processor wiring
- `losses` (`SpectralLoss`, `ReverbRegularizer`, `InharmonicityLoss`)
- `fdn_reverb` (`FeedbackDelayNetwork`)

## 2) Required CLI parity

The PyTorch scripts must preserve compatible flags/positionals for:

- `train_single_phase.py` parity through `ddsp_piano_pytorch/train.py`
- `synthesize_midi_file.py` parity through `ddsp_piano_pytorch/synthesize_midi_file.py`
- `evaluate_model.py` parity through `ddsp_piano_pytorch/evaluate_model.py`
- `preprocess_maestro.py` parity through `ddsp_piano_pytorch/preprocess_maestro.py`

## 3) Required configuration variants

These TensorFlow gin variants must have runnable YAML equivalents with matching graph semantics:

- `maestro-v2.gin`
- `maestro-v2-regularized.gin`
- `dafx22.gin`
- `dafx22-24kHz.gin`
- `ENSTDkCl-8kHz.gin`
- `ENSTDkCl-32kHz.gin`
- `surrogate.gin`
- `multi_instruments.gin`

## 4) Behavior parity requirements

- Global feature flow in `PianoModel`: `z_encoder`, `context_network`, `background_noise_model`, `reverb_model`.
- Parallelization contract: merge `[B, T, N, ...]` into `[N*B, T, ...]` for monophonic path and unmerge after.
- Monophonic feature stack parity (`note_release`, `inharm_model`, `detuner`, monophonic controls).
- Polyphonic synthesis parity: additive + noise accumulation for each synth, then global reverb.
- Alternate training phase semantics:
  - phase 1 freezes frequency-related modules and enables context/monophonic/reverb learning.
  - phase 2 inverses freezing and enables detuning/inharmonic learning.

## 5) DSP parity requirements

- `FeedbackDelayNetwork` math parity with TF reference:
  - Householder mixing matrix implementation
  - allpass + one-pole tone control path
  - IR construction and convolution shape semantics
- `frequency_filter` path must preserve expected filtered-noise behavior and be covered by tests.

## 6) Test gate requirements

PyTorch repository must include:

- backward smoke tests for full `PianoModel`
- numerical sanity tests (`SpectralLoss(x, x) ~= 0`)
- alternate-training freeze map tests
- convert-weights transform tests (including GRU gate swap)
- FDN IR finite/decay checks
- TF-reference parity tests or generated golden fixtures for critical outputs

