# Advisor Proof Pack

This folder provides reproducible evidence that the PyTorch DDSP-Piano conversion is trainable and runnable with TensorFlow-DDSP-Piano-like workflows.

## 1) Environment and Tests

- Package manager: `uv`
- Validation command:
  - `uv run pytest`
- Result log:
  - `advisor_evidence/pytest_report.txt`

## 2) Toy MAESTRO-Style Dataset

- Generated tiny MAESTRO-like dataset:
  - `advisor_evidence/toy_maestro/maestro-v3.0.0.csv`
  - `advisor_evidence/toy_maestro/2004/sample.mid`
  - `advisor_evidence/toy_maestro/2004/sample.wav`

## 3) Preprocessing (TF-script-compatible flow)

- Command used:
  - `uv run python -m ddsp_piano_pytorch.preprocess_maestro advisor_evidence/toy_maestro advisor_evidence/preprocessed`
- Outputs:
  - `advisor_evidence/preprocessed/maestro_train.tfrecord` (manifest-format compatibility file)
  - `advisor_evidence/preprocessed/maestro_validation.tfrecord`

## 4) Training (TF `train_single_phase`-style interface)

- Command used:
  - `uv run python -m ddsp_piano_pytorch.train_single_phase --config ddsp_piano_pytorch/configs/maestro_v2_smoke.yaml --batch_size 1 --steps_per_epoch 1 --epochs 1 --lr 1e-3 --phase 1 advisor_evidence/toy_maestro advisor_evidence/experiment_smoke`
- Produced artifacts:
  - `advisor_evidence/experiment_smoke/phase_1/ckpts/epoch_0000.pt`
  - `advisor_evidence/experiment_smoke/phase_1/last_iter.pt`
  - `advisor_evidence/experiment_smoke/phase_1/training_meta.json`
  - `advisor_evidence/experiment_smoke/phase_1/logs/`

## 5) Synthesis (TF `synthesize_midi_file`-style interface)

- Command used:
  - `uv run python -m ddsp_piano_pytorch.synthesize_midi_file --config ddsp_piano_pytorch/configs/maestro_v2_smoke.yaml --ckpt advisor_evidence/experiment_smoke/phase_1/last_iter.pt --piano_type 0 advisor_evidence/toy_maestro/2004/sample.mid advisor_evidence/smoke_output.wav`
- Output:
  - `advisor_evidence/smoke_output.wav`

## 6) Evaluation (TF `evaluate_model`-style interface)

- Command used:
  - `uv run python -m ddsp_piano_pytorch.evaluate_model --config ddsp_piano_pytorch/configs/maestro_v2_smoke.yaml --ckpt advisor_evidence/experiment_smoke/phase_1/last_iter.pt advisor_evidence/toy_maestro advisor_evidence/eval_smoke`
- Output:
  - `advisor_evidence/eval_smoke/spectral_losses.csv`

## 7) Differentiability Evidence

- Gradient check output file:
  - `advisor_evidence/gradient_proof.txt`
- Expected key line:
  - `gradient_is_none=False`

## Notes

- This is a smoke-scale demonstration to prove end-to-end functionality and gradient flow.
- For advisor-facing research results, run the same workflows on full MAESTRO subsets and include longer training/evaluation logs.
