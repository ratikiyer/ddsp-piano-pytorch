import torch

from ddsp_piano_pytorch.modules.fdn_reverb import FeedbackDelayNetwork
from ddsp_piano_pytorch.modules.sub_modules import Parallelizer


def test_fdn_ir_finite_and_decays() -> None:
    torch.manual_seed(0)
    fdn = FeedbackDelayNetwork(sampling_rate=8000, delay_lines=8)
    input_gain = torch.rand(8)
    output_gain = torch.rand(8)
    gain_allpass = torch.rand(8, 4) * 0.5
    delays_allpass = torch.rand(8, 4) * 20.0
    t0 = torch.tensor(2.0)
    alpha = torch.tensor(0.7)
    early = torch.randn(128) * 0.01
    ir = fdn.get_ir(input_gain, output_gain, gain_allpass, delays_allpass, t0, alpha, early)
    assert torch.isfinite(ir).all()
    first = ir[: max(1, ir.shape[0] // 10)]
    last = ir[-max(1, ir.shape[0] // 10) :]
    assert last.pow(2).mean() < first.pow(2).mean()


def test_parallelizer_roundtrip_identity() -> None:
    par = Parallelizer(n_synths=4)
    features = {
        "conditioning": torch.randn(2, 6, 4, 2),
        "context": torch.randn(2, 6, 8),
        "global_inharm": torch.randn(2, 6, 1),
        "global_detuning": torch.randn(2, 6, 1),
        "f0_hz": torch.randn(8, 6, 1),
        "inharm_coef": torch.randn(8, 6, 1),
        "amplitudes": torch.randn(8, 6, 1),
        "harmonic_distribution": torch.randn(8, 6, 12),
        "noise_magnitudes": torch.randn(8, 6, 9),
    }
    p = par.parallelize(dict(features))
    u = par.unparallelize(dict(p))
    assert torch.equal(u["f0_hz"], features["f0_hz"].reshape(4, 2, 6, 1))
