import os
import random
import torch
import numpy as np

import os, random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = False) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark     = not deterministic

    torch.use_deterministic_algorithms(deterministic)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def resolve_device(device_str: str | None = None) -> str:
    if device_str is not None:
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"
