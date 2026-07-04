"""Single point of RNG control across random / numpy / torch (and MPS)."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed every RNG we touch. The audit is analytic (exact expvals) so it is
    deterministic regardless; this matters for *training* the models."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # torch optional at import time
        pass
    return seed
