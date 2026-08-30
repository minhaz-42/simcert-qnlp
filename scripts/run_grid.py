"""Re-run the certificate grid so every model carries the same statistics.

The audit gained per-chi bootstrap confidence intervals and an exact paired McNemar
test, and those live in the stored certificate, so a model keeps reporting "--" for
them until it is re-audited. This driver walks the grid, skips any run whose stored
result already has the current statistics, and keeps going when one entry fails so a
single bad model cannot strand the rest.

    conda run -n qnlp python scripts/run_grid.py                # everything still missing
    conda run -n qnlp python scripts/run_grid.py --workers 6    # across cores
    conda run -n qnlp python scripts/run_grid.py --dry-run      # just show the plan

Jobs are independent processes with their own seed, so running them concurrently
changes nothing about the numbers. Each worker is pinned to a single BLAS thread:
the arrays here are small (a 10-qubit statevector is 16 KB) so threaded BLAS buys
nothing, and letting every worker spawn a full thread pool only makes them fight
over the same cores.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from omegaconf import OmegaConf  # noqa: E402

from simcert.io_results import run_hash  # noqa: E402
from simcert.runner import load_config  # noqa: E402

# (model, dataset, audit, seeds). Ordered cheapest-first so the grid produces usable
# coverage early: discocat replays cached circuits in seconds, sst2 costs ~10 min a seed.
GRID = [
    ("discocat", "mc",      "discocat", range(1, 4)),   # cached circuits, seeds 1-3 only
    ("discocat", "mc_real", "discocat", range(1, 4)),
    ("discocat", "rp",      "discocat", range(1, 4)),
    ("qmsan",    "mc",      "default",  range(1, 9)),
    ("qmsan",    "rp",      "default",  range(1, 9)),
    ("qsann",    "mc",      "default",  range(1, 9)),
    ("qsann",    "rp",      "default",  range(1, 9)),
    ("vqc_text", "mc",      "default",  range(1, 9)),
    ("vqc_text", "rp",      "default",  range(1, 9)),
    ("claqs",    "mc",      "claqs",    range(1, 4)),   # 8 qubits, minutes per seed
    ("vqc_text", "sst2",    "default",  range(1, 9)),   # ~10 min a seed
    ("claqs",    "sst2",    "claqs",    range(1, 2)),
]

# chi*-vs-n scaling sweep: a second model so the headline figure is not single-model.
SCALING = [("qsann", "sst2", n) for n in (4, 6, 8, 10)]


def _hash_for(argv):
    cfg = load_config(argv)
    return run_hash(OmegaConf.to_container(cfg, resolve=True))


def _is_done(model, dataset, argv):
    """A run counts as done only if its stored certificate has the current statistics."""
    f = REPO / "results" / "metrics" / f"{dataset}__{model}__{_hash_for(argv)}.json"
    if not f.exists():
        return False
    try:
        cert = json.loads(f.read_text()).get("certificate", {})
    except Exception:
        return False
    return bool(cert.get("mcnemar_by_chi")) and bool(cert.get("accuracy_ci_by_chi"))


def _jobs():
    for model, dataset, audit, seeds in GRID:
        for seed in seeds:
            yield (model, dataset,
                   [f"mode=both", f"model={model}", f"dataset={dataset}",
                    f"audit={audit}", f"seed={seed}"],
                   f"{model}/{dataset}/seed{seed}")
    for model, dataset, n in SCALING:
        yield (model, dataset,
               [f"mode=both", f"model={model}", f"dataset={dataset}",
                "audit=scaling", "seed=1", f"model.n_qubits={n}"],
               f"{model}/{dataset}/scaling-n{n}")


def _run_one(job, index, total, lock):
    _, _, argv, label = job
    env = dict(os.environ)
    # one BLAS thread per worker; see the module docstring
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    start = time.time()
    r = subprocess.run([sys.executable, "-m", "simcert.runner", *argv],
                       cwd=REPO, capture_output=True, text=True, env=env)
    dt = time.time() - start
    with lock:
        stamp = time.strftime("%H:%M:%S")
        if r.returncode != 0:
            print(f"[{stamp}] ({index}/{total}) FAIL {label} ({dt:.0f}s)", flush=True)
            print((r.stdout or "")[-1200:], flush=True)
            print((r.stderr or "")[-1200:], flush=True)
        else:
            for line in (r.stdout or "").splitlines():
                if "VERDICT" in line:
                    print(f"[{stamp}] ({index}/{total}) OK {label} ({dt:.0f}s)  {line.strip()}",
                          flush=True)
                    break
            else:
                print(f"[{stamp}] ({index}/{total}) OK {label} ({dt:.0f}s)", flush=True)
    return label, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent runs; each is an independent process")
    args = ap.parse_args()

    jobs = list(_jobs())
    todo = [j for j in jobs if not _is_done(j[0], j[1], j[2])]
    print(f"[grid] {len(jobs)} jobs, {len(jobs) - len(todo)} already current, {len(todo)} to run, "
          f"{args.workers} worker(s)", flush=True)
    if args.dry_run:
        for _, _, _, label in todo:
            print(f"  TODO {label}", flush=True)
        return

    # Longest first: sst2 seeds and the scaling sweep dominate, so starting them early
    # keeps every worker busy instead of trailing one long job at the end.
    todo.sort(key=lambda j: (0 if ("sst2" in j[1] or "scaling" in j[3]) else 1))

    import threading
    lock = threading.Lock()
    t0 = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(_run_one, j, i, len(todo), lock)
                for i, j in enumerate(todo, 1)]
        for f in futs:
            label, rc = f.result()
            if rc != 0:
                failures.append(label)

    print(f"\n[grid] done in {(time.time()-t0)/60:.1f} min; "
          f"{len(todo)-len(failures)} ok, {len(failures)} failed", flush=True)
    for f in failures:
        print(f"  FAILED {f}", flush=True)


if __name__ == "__main__":
    main()
