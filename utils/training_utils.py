# utils/training_utils.py
import os, random, logging
import numpy as np
import torch

def set_seed( seed: int, deterministic: bool = False, logger: logging.Logger | None = None ) -> None:
    """
    Fija todas las semillas y el modo determinista de cuDNN.

    Args:
        seed (int): valor de la semilla.
        deterministic (bool): True → reproducibilidad completa
                              (más lento en GPU).
        logger (logging.Logger | None): instancia de logger principal;
                                        si es None se usa el del módulo.
    """
    logger = logger or logging.getLogger(__name__)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark     = not deterministic
    torch.use_deterministic_algorithms(deterministic)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.info("cuDNN deterministic mode  ACTIVE (desactivated benchmark mode)")
    else:
        logger.info("cuDNN benchmark mode ACTIVE (desactivated deterministic mode)")


def resolve_device(device_str: str | None = None) -> str:
    if device_str is not None:
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"
