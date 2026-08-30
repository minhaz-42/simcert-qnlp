"""Re-run the certificate grid so every model carries the same statistics.

The audit gained per-chi bootstrap confidence intervals and an exact paired McNemar
test, and those live in the stored certificate, so a model keeps reporting "--" for
them until it is re-audited. This driver walks the grid, skips any run whose stored
result already has the current statistics, and keeps going when one entry fails so a
single bad model cannot strand the rest.

    conda run -n qnlp python scripts/run_grid.py                 # memory-safe defaults
    conda run -n qnlp python scripts/run_grid.py --workers 3
    conda run -n qnlp python scripts/run_grid.py --dry-run       # just show the plan

Jobs are independent processes with their own seed, so running them concurrently
changes nothing about the numbers. Each worker is pinned to a single BLAS thread: the
arrays here are small (a 10-qubit statevector is 16 KB) so threaded BLAS buys nothing,
and letting every worker spawn a full thread pool only makes them fight over cores.

Memory, not CPU, is the binding constraint on a 16 GB laptop that is also running an
editor. Six workers once drove the machine into 19.5 GB of swap and took it down, so
scheduling here is admission-controlled rather than fixed-width:

  * a job only starts when the system genuinely has --min-free-gb available, so the
    driver throttles itself when something outside this process tree takes memory;
  * every running job is sampled once a second and killed if its own resident set
    passes --max-rss-gb, which turns a runaway job into one recorded failure instead
    of a system-wide stall;
  * workers default to 2 and jobs run at low priority, so the machine stays usable.

Nothing is lost to a kill or a crash: a run counts as done only once its certificate
is on disk with the current statistics, so re-invoking resumes exactly where it left.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from simcert.io_results import run_hash  # noqa: E402
from simcert.runner import load_config  # noqa: E402

# (model, dataset, audit, seeds). Ordered cheapest-first so the grid produces usable
# coverage early: discocat replays cached circuits in seconds, sst2 costs minutes a seed.
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
    ("vqc_text", "sst2",    "default",  range(1, 9)),
    ("claqs",    "sst2",    "claqs",    range(1, 2)),
]

# chi*-vs-n scaling sweep: a second model so the headline figure is not single-model.
SCALING = [("qsann", "sst2", n) for n in (4, 6, 8, 10)]

# Per-token models backprop through a statevector simulator, so the autograd graph spans
# the whole training set and it is that graph, not the 2^n statevector, that exhausts
# memory: every qsann/sst2 sweep peaked at ~6 GB identically at n=4 and n=10. Chunking
# accumulates the same gradient in bounded memory (tests/test_grad_accumulation.py proves
# the gradients match), so these runs carry a chunk size while the small mc/rp runs keep
# the untouched full-batch path. The chunk sizes differ because the per-example graph
# does: qsann costs a few MB an example, claqs about 83 MB (q=8 with LCU and a degree-5
# QSVT polynomial), which is why every full-batch claqs run was killed: claqs/mc peaked at
# 2.79-3.56 GB on only ~62 training examples. Measured on claqs/sst2, chunk=1 is the
# smallest and the fastest
# (0.53 GB and ~25 min, against 1.17 GB and ~42 min at chunk=4), because the per-example
# graph is large enough that building several at once costs more than it saves.
CHUNKED = {("qsann", "sst2"): 32, ("claqs", "sst2"): 1, ("claqs", "mc"): 1}


def _scaling_chunk(n_qubits: int) -> int:
    """Chunk for a scaling sweep run.

    The per-example graph grows with the statevector, so a chunk that is comfortable at
    n=4 is not at n=10: measured, chunk=32 peaked at 1.18 GB for n=4 and 1.78 GB for n=8
    but blew past 2.5 GB at n=10. Shrinking the chunk with n keeps every point of the
    sweep at roughly the same footprint.
    """
    return 32 if n_qubits <= 8 else 8

GB = 1024 ** 3


def _hash_for(argv):
    return run_hash(OmegaConf.to_container(load_config(argv), resolve=True))


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
            argv = ["mode=both", f"model={model}", f"dataset={dataset}",
                    f"audit={audit}", f"seed={seed}"]
            if (model, dataset) in CHUNKED:
                argv.append(f"model.train_chunk={CHUNKED[(model, dataset)]}")
            yield (model, dataset, argv, f"{model}/{dataset}/seed{seed}")
    for model, dataset, n in SCALING:
        argv = ["mode=both", f"model={model}", f"dataset={dataset}",
                "audit=scaling", "seed=1", f"model.n_qubits={n}",
                f"model.train_chunk={_scaling_chunk(n)}"]
        yield (model, dataset, argv, f"{model}/{dataset}/scaling-n{n}")


def _tree_rss(proc: psutil.Process) -> int:
    """Resident bytes for a job, counting any children it spawned."""
    total = 0
    try:
        total += proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total


def _run_one(job, index, total, lock, max_rss_gb):
    """Run one job under a resident-set watchdog. Returns (label, ok, peak_gb, note)."""
    _, _, argv, label = job
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"

    start = time.time()
    p = subprocess.Popen(
        [sys.executable, "-m", "simcert.runner", *argv],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        preexec_fn=lambda: os.nice(10),  # keep the machine usable while this runs
    )
    try:
        watched = psutil.Process(p.pid)
    except psutil.Error:
        watched = None

    peak = 0
    killed = False
    while p.poll() is None:
        if watched is not None:
            peak = max(peak, _tree_rss(watched))
            if peak > max_rss_gb * GB:
                killed = True
                for c in watched.children(recursive=True):
                    try:
                        c.kill()
                    except psutil.Error:
                        pass
                p.kill()
                break
        time.sleep(1.0)
    out, err = p.communicate()
    dt = time.time() - start
    peak_gb = peak / GB

    ok = (p.returncode == 0) and not killed
    with lock:
        stamp = time.strftime("%H:%M:%S")
        head = f"[{stamp}] ({index}/{total})"
        if killed:
            print(f"{head} KILLED {label} ({dt:.0f}s, peak {peak_gb:.2f} GB > "
                  f"{max_rss_gb} GB cap)", flush=True)
        elif not ok:
            print(f"{head} FAIL {label} ({dt:.0f}s, rc={p.returncode}, "
                  f"peak {peak_gb:.2f} GB)", flush=True)
            tail = (err or out or "").strip().splitlines()
            for line in tail[-12:]:
                print(f"    {line}", flush=True)
            if not tail:
                print("    (no output: the process was terminated by a signal, "
                      "typically the OS reclaiming memory)", flush=True)
        else:
            verdict = next((ln.strip() for ln in (out or "").splitlines()
                            if "VERDICT" in ln), "")
            print(f"{head} OK {label} ({dt:.0f}s, peak {peak_gb:.2f} GB)  {verdict}",
                  flush=True)
    return label, ok, peak_gb


def _await_memory(min_free_gb, lock, label):
    """Block until the system really has room, so we never start a job into swap."""
    warned = False
    while psutil.virtual_memory().available < min_free_gb * GB:
        if not warned:
            with lock:
                print(f"[{time.strftime('%H:%M:%S')}] waiting for memory before {label} "
                      f"({psutil.virtual_memory().available / GB:.1f} GB free, "
                      f"need {min_free_gb} GB)", flush=True)
            warned = True
        time.sleep(5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=2,
                    help="max concurrent runs (default 2; memory, not CPU, is the limit)")
    ap.add_argument("--max-rss-gb", type=float, default=4.0,
                    help="kill a single run whose resident set exceeds this")
    ap.add_argument("--min-free-gb", type=float, default=3.0,
                    help="required free system memory before starting a run")
    args = ap.parse_args()

    jobs = list(_jobs())
    todo = [j for j in jobs if not _is_done(j[0], j[1], j[2])]
    vm = psutil.virtual_memory()
    print(f"[grid] {len(jobs)} jobs, {len(jobs) - len(todo)} already current, "
          f"{len(todo)} to run", flush=True)
    print(f"[grid] {args.workers} worker(s), cap {args.max_rss_gb} GB/run, "
          f"start gate {args.min_free_gb} GB free; system has {vm.available / GB:.1f} GB "
          f"of {vm.total / GB:.1f} GB free now", flush=True)
    if args.dry_run:
        for _, _, _, label in todo:
            print(f"  TODO {label}", flush=True)
        return

    # Longest first so the slow jobs are not left trailing behind everything else.
    todo.sort(key=lambda j: (0 if ("sst2" in j[1] or "scaling" in j[3]) else 1))

    lock = threading.Lock()
    sem = threading.Semaphore(max(1, args.workers))
    results = []
    t0 = time.time()

    def worker(job, idx):
        with sem:
            _await_memory(args.min_free_gb, lock, job[3])
            results.append(_run_one(job, idx, len(todo), lock, args.max_rss_gb))

    threads = []
    for i, j in enumerate(todo, 1):
        t = threading.Thread(target=worker, args=(j, i), daemon=False)
        t.start()
        threads.append(t)
        time.sleep(2.0)  # stagger starts so imports do not spike together
    for t in threads:
        t.join()

    bad = [lab for lab, ok, _ in results if not ok]
    peak = max((p for _, _, p in results), default=0.0)
    print(f"\n[grid] done in {(time.time() - t0) / 60:.1f} min; "
          f"{len(results) - len(bad)} ok, {len(bad)} failed; "
          f"worst single-run peak {peak:.2f} GB", flush=True)
    for b in bad:
        print(f"  FAILED {b}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
