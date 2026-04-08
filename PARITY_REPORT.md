# DDSP-Piano PyTorch Parity Report

## Completed remediation

- Added explicit parity contract document in `PARITY_MANIFEST.md`.
- Refactored `PianoModel.forward` to:
  - synthesize all voices in one batched pass,
  - apply background noise once per batch item,
  - call reverb convolution in batched mode.
- Fixed `F0ProcessorCell.release_duration` gradient path by removing `.detach().item()` usage.
- Reimplemented `JointParametricInharmTuning` with instrument-indexed parameters and TF-style equations.
- Aligned FDN mixing matrix with TF (`-I + 0.5 * ones`) and switched default IR FFT size to `2 * sample_rate`.
- Added DSP tolerance document in `PARITY_TOLERANCES.md`.
- Enabled config-driven component hydration in `train.py` (module classes now instantiated from YAML).
- Added runnable YAML variants for missing TF config families.
- Cleaned preprocessing API:
  - introduced `preprocess_data_into_manifest`,
  - kept `preprocess_data_into_tfrecord` as compatibility alias that writes `.csv`,
  - updated preprocessing CLI outputs to `.csv`.
- Expanded public module exports for parity (`ddsp_piano_pytorch/modules/__init__.py` and package `__init__.py`).

## Test coverage additions

- `tests/test_training_regression_guards.py`
  - end-to-end backward smoke,
  - spectral identity sanity,
  - alternate training freeze map,
  - joint tuning piano-model dependence.
- `tests/test_convert_weights.py` for GRU block swap/transpose guards.
- `tests/test_reverb_and_parallelizer.py` for FDN IR stability/decay and parallelizer round-trip.
- Added mismatched-length convolution test in `tests/test_core_parity_acids_ircam.py`.
- Updated preprocessing compatibility assertions in `tests/test_script_api_parity.py`.
- Added optional TF-runtime parity gates in `tests/test_tf_reference_parity.py`.

## Validation status

- Test suite status: passing (`.venv/bin/python -m pytest -q`).
- Benchmark status (CPU quick run):
  - `steps`: 5
  - `step_time_sec`: ~0.320
  - `steps_per_sec`: ~3.12

## Known non-bit-exact area

- `frequency_filter` remains an STFT-domain approximation in PyTorch. This is documented and bounded by tolerance policy in `PARITY_TOLERANCES.md`.
