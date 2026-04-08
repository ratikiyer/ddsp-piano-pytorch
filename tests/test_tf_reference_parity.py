import pytest
import torch

from ddsp_piano_pytorch.modules.fdn_reverb import FeedbackDelayNetwork
from ddsp_piano_pytorch.modules.sub_modules import JointParametricInharmTuning


tf = pytest.importorskip("tensorflow")


def test_fdn_mixing_matrix_matches_tf_reference() -> None:
    n = 8
    pt = FeedbackDelayNetwork(sampling_rate=16000, delay_lines=n).mixing_matrix
    tf_mat = -1.0 * tf.eye(n, dtype=tf.float32) + 0.5 * tf.ones((n, n), dtype=tf.float32)
    assert torch.allclose(pt, torch.from_numpy(tf_mat.numpy()), atol=0.0, rtol=0.0)


def test_joint_parametric_reverse_scaled_tanh_matches_tf() -> None:
    x = torch.linspace(-3.0, 3.0, 101)
    pt = JointParametricInharmTuning.reverse_scaled_tanh(x)
    tf_out = (1.0 - tf.math.tanh(tf.convert_to_tensor(x.numpy()))) / 2.0
    assert torch.allclose(pt, torch.from_numpy(tf_out.numpy()), atol=1e-7, rtol=1e-6)
