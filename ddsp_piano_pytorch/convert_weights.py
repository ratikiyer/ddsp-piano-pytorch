from __future__ import annotations

import argparse
from collections.abc import Callable

import torch


def _swap_gru_zr_blocks(weight: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """TF GRU order [z, r, h] -> PT order [r, z, h]."""
    z = weight[:hidden_size]
    r = weight[hidden_size : 2 * hidden_size]
    h = weight[2 * hidden_size :]
    return torch.cat([r, z, h], dim=0)


def convert_checkpoint(
    tf_checkpoint_path: str,
    state_dict: dict[str, torch.Tensor],
    name_map: dict[str, str],
    transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    import tensorflow as tf  # type: ignore

    transforms = transforms or {}
    reader = tf.train.load_checkpoint(tf_checkpoint_path)
    tf_vars = {k: reader.get_tensor(k) for k, _ in tf.train.list_variables(tf_checkpoint_path)}
    converted = {}

    for tf_name, pt_name in name_map.items():
        if tf_name not in tf_vars or pt_name not in state_dict:
            continue
        tensor = torch.from_numpy(tf_vars[tf_name])
        if tf_name.endswith("kernel") and tensor.ndim == 2 and "gru" not in tf_name.lower():
            tensor = tensor.t()
        if "gru" in tf_name.lower():
            if tensor.ndim == 2 and tensor.shape[0] == state_dict[pt_name].shape[1]:
                tensor = tensor.t()
            hidden = state_dict[pt_name].shape[0] // 3
            if tensor.shape[0] == hidden * 3:
                tensor = _swap_gru_zr_blocks(tensor, hidden)
        if tf_name in transforms:
            tensor = transforms[tf_name](tensor)
        if tensor.shape == state_dict[pt_name].shape:
            converted[pt_name] = tensor.to(dtype=state_dict[pt_name].dtype)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TensorFlow DDSP-Piano checkpoint to PyTorch state_dict.")
    parser.add_argument("--tf_checkpoint", required=True, type=str)
    parser.add_argument("--pt_template", required=True, type=str, help="PyTorch checkpoint containing model/state_dict template.")
    parser.add_argument("--name_map", required=True, type=str, help="Torch-saved dict {tf_name: pt_name}.")
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    template = torch.load(args.pt_template, map_location="cpu")
    state_dict = template["model"] if isinstance(template, dict) and "model" in template else template
    name_map = torch.load(args.name_map, map_location="cpu")
    converted = convert_checkpoint(args.tf_checkpoint, state_dict=state_dict, name_map=name_map)
    state_dict.update(converted)
    torch.save({"model": state_dict}, args.output)
    print(f"Converted {len(converted)} tensors to {args.output}")


if __name__ == "__main__":
    main()
