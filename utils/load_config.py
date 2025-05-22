import numpy as np
import yaml
import os, sys
from pathlib import Path
from types import SimpleNamespace   
import logging

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def init_logger(cfg_logging: dict) -> SimpleNamespace:

    if cfg_logging.get("log_to_file", False):
        Path(cfg_logging["log_file"]).parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    if cfg_logging.get("log_to_file", False):
        handlers.append(logging.FileHandler(cfg_logging["log_file"], encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, cfg_logging.get("level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,                        
    )

    return SimpleNamespace(
        main = logging.getLogger("main"),
        train = logging.getLogger("train"),
        utils = logging.getLogger("utils"),
    )