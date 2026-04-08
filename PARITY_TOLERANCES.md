# DDSP-Piano Parity Tolerances

This document defines tolerated numerical differences between TensorFlow DDSP-Piano and the PyTorch port.

## Strict (no tolerance)

- Public API symbols and script flags/positionals.
- Tensor shapes and key names returned by stable public entrypoints.

## Numeric tolerances

- DSP modules are compared with `rtol=1e-4`, `atol=1e-5` unless noted otherwise.
- Reverb IR parity allows relaxed comparison (`rtol=5e-3`, `atol=1e-4`) because matrix solves and FFT backends differ across runtimes.

## Known non-bit-exact component

- `frequency_filter` in `ddsp_piano_pytorch/core.py` is implemented with a framewise STFT-domain filtering approximation.
- TensorFlow DDSP builds FIR impulse responses per frame and applies frequency-domain convolution with a different computational path.
- We treat this as numerically equivalent within the tolerance bands above and enforce it through regression tests on deterministic fixtures.

