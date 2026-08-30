"""Re-run the certificate grid so every model carries the same statistics.

The audit gained per-chi bootstrap confidence intervals and an exact paired McNemar
test, and those live in the stored certificate, so a model keeps reporting "--" for
them until it is re-audited. This driver walks the grid, skips any run whose stored
result already has the current statistics, and keeps going when one entry fails so a
single bad model cannot strand the rest.

    conda run -n qnlp python scripts/run_grid.py            # everything still missing
    conda run -n qnlp python scripts/run_grid.py --dry-run  # just show the plan
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = list(_jobs())
    todo = [j for j in jobs if not _is_done(j[0], j[1], j[2])]
    print(f"[grid] {len(jobs)} jobs, {len(jobs) - len(todo)} already current, {len(todo)} to run",
          flush=True)
    if args.dry_run:
        for _, _, _, label in todo:
            print(f"  TODO {label}", flush=True)
        return

    failures = []
    t0 = time.time()
    for i, (_, _, argv, label) in enumerate(todo, 1):
        print(f"\n[{time.strftime('%H:%M:%S')}] ({i}/{len(todo)}) START {label}", flush=True)
        s = time.time()
        r = subprocess.run([sys.executable, "-m", "simcert.runner", *argv],
                           cwd=REPO, capture_output=True, text=True)
        dt = time.time() - s
        if r.returncode != 0:
            failures.append(label)
            print(f"[{time.strftime('%H:%M:%S')}] FAIL {label} ({dt:.0f}s)", flush=True)
            print((r.stdout or "")[-1500:], flush=True)
            print((r.stderr or "")[-1500:], flush=True)
        else:
            for line in (r.stdout or "").splitlines():
                if "VERDICT" in line or "entanglement-removal" in line or "trained" in line:
                    print("   " + line.strip(), flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] OK {label} ({dt:.0f}s)", flush=True)

    print(f"\n[grid] done in {(time.time()-t0)/60:.1f} min; "
          f"{len(todo)-len(failures)} ok, {len(failures)} failed", flush=True)
    for f in failures:
        print(f"  FAILED {f}", flush=True)


if __name__ == "__main__":
    main()
