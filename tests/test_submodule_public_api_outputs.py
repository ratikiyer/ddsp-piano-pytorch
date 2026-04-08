import torch

from ddsp_piano_pytorch.modules.sub_modules import (
    BackgroundNoiseFilter,
    DeepDetuner,
    Detuner,
    DictDetuner,
    FiLMContextNetwork,
    InharmonicityNetwork,
    JointParametricInharmTuning,
    MonophonicDeepNetwork,
    NoteRelease,
    OnsetLinspaceCell,
    OneHotZEncoder,
    Parallelizer,
    ParametricTuning,
    PartialMasking,
    SimpleContextNet,
    SurrogateModule,
)


def test_global_submodules_public_outputs() -> None:
    b, t, n = 2, 6, 16
    conditioning = torch.zeros(b, t, n, 2)
    conditioning[..., 0] = 60.0
    pedals = torch.zeros(b, t, 4)
    piano_model = torch.tensor([0, 1], dtype=torch.long)

    zenc = OneHotZEncoder(n_instruments=10, z_dim=8, n_inharm_params=1, n_detuning_params=1)
    z, gi, gd = zenc(piano_model)
    assert z.shape[-1] == 8 and gi.shape[-1] == 1 and gd.shape[-1] == 1

    film = FiLMContextNetwork(n_synths=n, n_instruments=10, layer_dim=64, context_dim=16, z_dim=8)
    ctx = film(conditioning, pedals, piano_model)
    assert ctx.shape == (b, t, 16)

    simple = SimpleContextNet(n_synths=n, hidden_dim=32, context_dim=8)
    ctx2 = simple(conditioning, pedals, None)
    assert ctx2.shape == (b, t, 8)

    bg = BackgroundNoiseFilter(n_instruments=10, n_filters=33)
    mags = bg(piano_model, n_frames=t)
    assert mags.shape == (b, t, 33)


def test_monophonic_and_tuning_submodule_outputs() -> None:
    b, t, n = 2, 6, 16
    conditioning = torch.zeros(b, t, n, 2)
    conditioning[..., 0] = 60.0
    conditioning[..., 1] = 0.5
    ext = conditioning[..., 0:1]
    piano_model = torch.tensor([0, 1], dtype=torch.long)

    nr = NoteRelease(frame_rate=250)
    ext2 = nr(conditioning)
    assert ext2.shape == ext.shape

    inh = InharmonicityNetwork()
    coef = inh(ext[:, :, 0], None)
    assert coef.shape == (b, t, 1)

    pt = ParametricTuning()
    f0, coef2 = pt(ext[:, :, 0], None)
    assert f0.shape == (b, t, 1)
    assert coef2.shape == (b, t, 1)

    jpt = JointParametricInharmTuning(n_instruments=10)
    f0j, coefj = jpt(ext[:, :, 0], piano_model)
    assert f0j.shape == (b, t, 1)
    assert coefj.shape == (b, t, 1)

    det = Detuner()
    f0d = det(ext[:, :, 0], None)
    assert f0d.shape == (b, t, 1)
    ddet = DictDetuner()
    f0dd = ddet(ext[:, :, 0], None)
    assert f0dd.shape == (b, t, 1)
    deep = DeepDetuner(hidden_dim=16)
    f0deep = deep(ext[:, :, 0], None)
    assert f0deep.shape == (b, t, 1)

    mono = MonophonicDeepNetwork(rnn_channels=32, ch=16, context_dim=8, output_splits=(("amplitudes", 1), ("harmonic_distribution", 12), ("noise_magnitudes", 9)))
    par_cond = conditioning.reshape(b * n, t, 2)
    par_ext = ext.reshape(b * n, t, 1)
    par_ctx = torch.zeros(b * n, t, 8)
    out = mono(par_cond, par_ext, par_ctx)
    assert out["amplitudes"].shape == (b * n, t, 1)
    assert out["harmonic_distribution"].shape == (b * n, t, 12)
    assert out["noise_magnitudes"].shape == (b * n, t, 9)


def test_utility_submodules_public_outputs() -> None:
    b, t, n = 1, 8, 16
    features = {
        "conditioning": torch.zeros(b, t, n, 2),
        "context": torch.zeros(b, t, 8),
        "global_inharm": torch.zeros(b, t, 1),
        "global_detuning": torch.zeros(b, t, 1),
    }
    par = Parallelizer(n_synths=n)
    p = par(features, parallelize=True)
    assert p["conditioning"].shape[0] == b * n

    cell = OnsetLinspaceCell()
    onsets = torch.zeros(b, t, n, 1)
    onsets[:, 0, :, 0] = 1.0
    dt = cell(onsets)
    assert dt.shape == (b, t, 1)

    surr = SurrogateModule(n_harmonics=12)
    cond = torch.zeros(b, t, n, 2)
    ext = torch.zeros(b, t, n, 1) + 60
    decays, decay_time = surr(cond, ext)
    assert decays.shape == (b, t, n, 12)
    assert decay_time.shape == (b, t, 1)

    pm = PartialMasking(n_partials=6)
    hdist = torch.ones(b, t, 12)
    masked = pm(hdist, n_partials=6)
    assert masked.shape == hdist.shape
    assert (masked[..., 6:] <= -9.9).all()
