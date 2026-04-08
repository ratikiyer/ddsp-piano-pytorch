from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


class MultiAdd(nn.Module):
    def forward(self, *signals: torch.Tensor) -> torch.Tensor:
        out = None
        for signal in signals:
            out = signal if out is None else out + signal
        if out is None:
            raise ValueError("MultiAdd requires at least one signal.")
        return out


class ProcessorGroup(nn.Module):
    """Simple ordered DAG execution engine.

    processors: OrderedDict[name] = (module, input_keys)
    """

    def __init__(self, processors: OrderedDict[str, tuple[nn.Module, list[str]]]) -> None:
        super().__init__()
        self.names = list(processors.keys())
        self.modules_dict = nn.ModuleDict({name: proc for name, (proc, _) in processors.items()})
        self.input_keys = {name: keys for name, (_, keys) in processors.items()}

    def forward(self, features: dict[str, Any], return_outputs_dict: bool = False) -> dict[str, Any] | torch.Tensor:
        outputs: dict[str, Any] = dict(features)
        signal = None
        for name in self.names:
            proc = self.modules_dict[name]
            keys = self.input_keys[name]
            args = [outputs[k] for k in keys]
            out = proc(*args)
            outputs[name] = out
            if isinstance(out, torch.Tensor):
                signal = out
        if return_outputs_dict:
            return {"controls": outputs, "signal": signal}
        if signal is None:
            raise ValueError("ProcessorGroup produced no tensor signal.")
        return signal
