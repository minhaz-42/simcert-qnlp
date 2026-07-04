"""Deterministic run-hash keying + result JSON writers.

Every run is keyed by a sha1 of its canonicalised config (minus volatile paths), so
figures regenerate deterministically and re-auditing a stored circuit never retrains.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def run_hash(cfg: dict, drop_keys=("paths", "run_hash")) -> str:
    filtered = {k: v for k, v in cfg.items() if k not in drop_keys}
    canonical = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def git_sha(default: str = "unknown") -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return default


def lib_versions() -> dict:
    out = {}
    for mod in ("pennylane", "quimb", "qiskit", "qiskit_aer", "numpy", "torch", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def result_path(results_dir: str | Path, dataset: str, model: str, rhash: str) -> Path:
    return Path(results_dir) / "metrics" / f"{dataset}__{model}__{rhash}.json"


def save_result(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # no sort_keys: chi-indexed dicts mix int and "full" keys; insertion order is deterministic
    p.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")
