import random

import numpy as np
import torch


def pytest_configure() -> None:
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
