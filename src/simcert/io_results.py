"""Deterministic run-hash keying + result JSON writers.

Every run is keyed by a sha1 of its canonicalised config (minus volatile paths), so
figures regenerate deterministically and re-auditing a stored circuit never retrains.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def run_hash(cfg: dict, drop_keys=("paths", "run_hash", "mode")) -> str:
    """Content hash of a run configuration.

    ``mode`` is excluded on purpose: train, audit and both describe *when* work happens,
    not *what* is computed, so a checkpoint written by ``mode=train`` has to be findable
    by the ``mode=audit`` pass that consumes it. Including it silently gave the two steps
    different hashes and made the documented two-step workflow miss its own checkpoint.
    """
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


def select_runs(files) -> dict:
    """Group stored runs into one comparable set of seeds per (model, dataset).

    ``results/metrics`` accumulates every run ever stored, which includes the chi*-vs-n
    scaling sweeps (a different audit config at several qubit counts) and runs superseded
    by later code or config revisions. Averaging those into one row or curve reports runs
    at n=4,6,8,10 as though they were seeds of a single configuration.

    Selection keeps only non-scaling audits, pins each group to the qubit count and audit
    config of its most recent run, and keeps one run per seed, the newest. Returns
    ``{(model, dataset): [full_result_dict, ...]}`` so callers can reach both
    ``certificate`` and ``details``.
    """
    import json as _json
    import os as _os
    from collections import defaultdict as _dd

    recs = []
    for f in sorted(files, key=_os.path.getmtime):  # oldest -> newest
        try:
            d = _json.loads(Path(f).read_text())
        except Exception:
            continue
        if "certificate" not in d:
            continue
        cfg = d.get("config", {})
        if cfg.get("audit_name") == "scaling":
            continue  # belongs to the chi*-vs-n figure, not a per-seed aggregate
        recs.append({
            "key": (d["certificate"]["model"], d["certificate"]["dataset"]),
            "variant": ((cfg.get("model") or {}).get("n_qubits"), cfg.get("audit_name")),
            "seed": cfg.get("seed"),
            "run": d,
        })

    groups = _dd(list)
    for r in recs:
        groups[r["key"]].append(r)

    out = {}
    for key, rs in groups.items():
        current = rs[-1]["variant"]  # the newest run defines the current configuration
        rs = [r for r in rs if r["variant"] == current]
        by_seed = {}
        for r in rs:  # ascending mtime, so the last write for a seed wins
            by_seed[r["seed"]] = r["run"]
        out[key] = [by_seed[s] for s in sorted(by_seed, key=lambda x: (x is None, x))]
    return out
