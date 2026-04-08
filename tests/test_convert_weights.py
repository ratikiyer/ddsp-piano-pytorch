import torch

from ddsp_piano_pytorch.convert_weights import _swap_gru_zr_blocks


def test_swap_gru_zr_blocks_tensor() -> None:
    hidden = 4
    weight = torch.arange(hidden * 3 * 2, dtype=torch.float32).reshape(hidden * 3, 2)
    swapped = _swap_gru_zr_blocks(weight, hidden)
    assert torch.equal(swapped[:hidden], weight[hidden : 2 * hidden])
    assert torch.equal(swapped[hidden : 2 * hidden], weight[:hidden])
    assert torch.equal(swapped[2 * hidden :], weight[2 * hidden :])


def test_dense_kernel_transpose_rule_shape() -> None:
    tf_kernel = torch.randn(3, 5)
    pt_weight_shape = (5, 3)
    converted = tf_kernel.t()
    assert converted.shape == pt_weight_shape
