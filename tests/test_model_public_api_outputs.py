import torch

from ddsp_piano_pytorch.modules.piano_model import PianoModel


def test_piano_model_public_forward_contract() -> None:
    model = PianoModel(
        n_synths=16,
        n_harmonics=32,
        n_noise_bands=33,
        sample_rate=8000,
        frame_rate=100,
        context_dim=32,
    )
    b, t, n = 1, 8, 16
    conditioning = torch.zeros(b, t, n, 2)
    conditioning[..., 0] = 60.0
    conditioning[..., 1] = 0.5
    pedals = torch.zeros(b, t, 4)
    piano_model = torch.zeros(b, dtype=torch.long)
    out = model(conditioning=conditioning, pedals=pedals, piano_model=piano_model)
    assert isinstance(out, dict)
    assert "audio" in out
    assert "reverb_ir" in out
    assert out["audio"].shape[0] == b
