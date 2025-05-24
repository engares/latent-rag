# Project-wide Code Style Guide (English)

> **Scope**  All Python modules, notebooks, shell scripts and configuration files located under the `tfm` repository.  New code **must** comply immediately.  Legacy code should be progressively refactored.

---

## 1  General Principles

| Principle                         | Rationale                                                        |
| --------------------------------- | ---------------------------------------------------------------- |
| **Consistency over cleverness**   | Readability and maintenance outweigh micro-optimisations.        |
| **Explicit > implicit**           | Follow *The Zen of Python*. Avoid hidden state and side effects. |
| **Fail fast, fail loud**          | Raise specific exceptions early; log context-rich messages.      |
| **Pure functions where possible** | Functions that depend only on arguments are easier to test.      |

---

## 2  File & Folder Layout

* Package modules use **snake\_case** filenames: `contrastive_autoencoder.py`.
* Public executables go under `/scripts` with a short shebang and CLI (`argparse`).
* Unit tests mirror the package tree under `/tests`.
* Keep data or model artefacts out of Git; place under `/data` or `/models/checkpoints` and add to `.gitignore`.

---

## 3  Imports

```
# 1. Standard library
import os
import json

# 2. Third-party
import numpy as np
import torch

# 3. First-party (this repo)
from utils.load_config import load_config
```

* Use **absolute imports** inside the package.
* Never use `from module import *`.
* Group imports and separate blocks with one blank line.

---

## 4  Naming Conventions

* **snake\_case** for variables, functions and methods.
* **PascalCase** for classes and `Enum` members.
* **UPPER\_SNAKE\_CASE** for module-level constants.
* Avoid Spanish identifiers; prefer descriptive English (`embedding_dim`, not `dim_emb`).

---

## 5  Docstrings & Comments

* **Every** public module, function, class and method **must** have a docstring in **English** using the **Google style**.
* Keep inline comments short; they explain *why*, not *what*.
* TODO/FIXME tags must include assignee or ticket reference.

```python
def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return Recall@k.

    Args:
        retrieved: Ordered list of retrieved IDs.
        relevant: Set or list of relevant IDs.
        k: Cut-off rank.

    Returns:
        Fraction of relevant items found in the top-k.
    """
```

---

## 6  Type Annotations & Runtime Checks

* Use **PEP 484** type hints everywhere (functions, class attributes).
* **All functions must declare input and output types explicitly.**
* Validate external inputs with `assert` or explicit `if ... raise ValueError`.
* Run **mypy** in *strict* mode as a CI step.

---

## 7  Logging

* Initialise loggers via `utils.load_config.init_logger`.
* Use module-level loggers: `logger = logging.getLogger(__name__)`.
* Logging levels: `debug` (dev insights), `info` (milestones), `warning` (recoverable), `error` (cannot proceed), `critical` (program abort).
* Never hide exceptions; use `logger.exception` to preserve traceback.

---

## 8  Error Handling

* Prefer built-in exceptions (`ValueError`, `TypeError`) unless a custom domain error clarifies intent.
* Enrich messages with variable values.
* Do **not** swallow exceptions silently.

---

## 9  Configuration Management

* All hyper-parameters live in YAML under `/config`.
* Load configs **only** through `utils.load_config.load_config`.
* Functions accept an explicit `config: dict` argument instead of reading files internally, unless the function’s **sole purpose** is configuration loading.

---

## 10  CLI & Scripts

* Use `argparse` with long option names (`--batch_size`).
* Provide `--config` flag pointing to a YAML; CLI flags override YAML.
* Scripts must be import-safe (`if __name__ == "__main__":`).

---

## 11  Testing & Quality Gates

* Write **pytest** unit tests covering critical paths.
* Minimum coverage threshold: **80 %**.
* Run `black`, `ruff`, `isort`, `mypy` and tests in CI before merge.

---

## 12  Formatting Tools

| Tool      | Version | Role                                 |
| --------- | ------- | ------------------------------------ |
| **black** | 24.3+   | Code formatting (line length 88).    |
| **ruff**  | 0.3+    | Linter (select = "ALL", ignore = …). |
| **isort** | 5+      | Import ordering (profile = "black"). |

---

## 13  Dependencies & Virtual Envs

* Pin versions in `requirements.txt` (prod) and `requirements-dev.txt` (lint/test).
* Use **conda** or **venv**; never rely on system Python.
* Document GPU/CPU requirements in `README.md`.

---

## 14  Internationalisation

* **All code, comments and docstrings must be in English.**  Spanish is reserved for external documentation or academic writing outside the repository.

---

## 15  Example Module Skeleton

```python
"""Contrastive Autoencoder model.

Implements an encoder–decoder architecture trained with triplet loss.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

class ContrastiveAutoencoder(nn.Module):
    """Linear contrastive autoencoder with L2-normalised latent vectors."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tensor:  # noqa: D401
        """Return L2-normalised latent representation."""
        return torch.nn.functional.normalize(self.encoder(x), p=2, dim=-1)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:    # type: ignore[override]
        return self.decode(self.encode(x))
```

---

## 16  Migration Plan for Legacy Code

1. **Phase 1**  — New code follows this guide immediately.
2. **Phase 2**  — Touch legacy modules only when editing; refactor headers, identifiers, comments to English.
3. **Phase 3**  — Run automated formatters; fix mypy errors; localised refactors.

---

*End of style guide.*
