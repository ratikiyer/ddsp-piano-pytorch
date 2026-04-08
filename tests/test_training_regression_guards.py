import torch

from ddsp_piano_pytorch.modules.losses import SpectralLoss, SpectralLossConfig
from ddsp_piano_pytorch.modules.piano_model import PianoModel
from ddsp_piano_pytorch.modules.sub_modules import JointParametricInharmTuning


def test_end_to_end_backward_smoke() -> None:
    torch.manual_seed(0)
    model = PianoModel(
        n_synths=2,
        n_harmonics=8,
        n_noise_bands=9,
        sample_rate=4000,
        frame_rate=50,
        context_dim=16,
    )
    b, t, n = 1, 10, 2
    conditioning = torch.zeros(b, t, n, 2)
    conditioning[:, :4, :, 0] = 60.0
    conditioning[:, :4, :, 1] = 0.8
    pedals = torch.zeros(b, t, 4)
    piano_model = torch.zeros(b, dtype=torch.long)
    target = torch.rand(b, int(4000 * (t / 50.0)))
    out = model(conditioning=conditioning, pedals=pedals, piano_model=piano_model)
    loss = SpectralLoss(SpectralLossConfig(fft_sizes=(256, 128), overlap=0.5))(target, out["audio"])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.monophonic_network.rnn.weight_ih_l0.grad is not None
    assert model.note_release.cell.release_duration.grad is not None
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_spectral_loss_zero_identity() -> None:
    loss_fn = SpectralLoss(SpectralLossConfig(fft_sizes=(512, 256), overlap=0.75))
    x = torch.randn(2, 4096)
    loss = loss_fn(x, x)
    assert float(loss.item()) < 1e-4


def test_alternate_training_freeze_map() -> None:
    model = PianoModel()
    model.alternate_training(first_phase=True)
    assert all(not p.requires_grad for p in model.inharm_model.parameters())
    assert all(not p.requires_grad for p in model.detuner.parameters())
    assert all(p.requires_grad for p in model.note_release.parameters())
    assert all(p.requires_grad for p in model.context_network.parameters())
    model.alternate_training(first_phase=False)
    assert all(p.requires_grad for p in model.inharm_model.parameters())
    assert all(p.requires_grad for p in model.detuner.parameters())
    assert all(not p.requires_grad for p in model.note_release.parameters())
    assert all(not p.requires_grad for p in model.context_network.parameters())


def test_joint_parametric_inharm_tuning_uses_piano_model() -> None:
    model = JointParametricInharmTuning(n_instruments=2)
    with torch.no_grad():
        model.alpha_b.weight[0].fill_(-0.1)
        model.beta_b.weight[0].fill_(-6.8)
        model.alpha_t.weight[0].fill_(0.09)
        model.beta_t.weight[0].fill_(-13.7)
        model.pitch_ref.weight[0].fill_(64.0)
        model.k_param.weight[0].fill_(4.5)
        model.alpha.weight[0].fill_(24.0)

        model.alpha_b.weight[1].fill_(-0.2)
        model.beta_b.weight[1].fill_(-7.2)
        model.alpha_t.weight[1].fill_(0.12)
        model.beta_t.weight[1].fill_(-13.2)
        model.pitch_ref.weight[1].fill_(60.0)
        model.k_param.weight[1].fill_(8.0)
        model.alpha.weight[1].fill_(16.0)

    ext = torch.full((2, 5, 1), 64.0)
    piano_ids = torch.tensor([0, 1], dtype=torch.long)
    f0, inharm = model(ext, piano_ids)
    assert not torch.allclose(f0[0], f0[1])
    assert not torch.allclose(inharm[0], inharm[1])
