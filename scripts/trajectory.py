"""Does chi* depend on how well the model has actually learned?

    python scripts/trajectory.py --model qsann --dataset rp --seed 1

Trains once with per-epoch snapshots, then re-runs the audit at every snapshot, so chi*
can be read against test accuracy along the training path rather than only at the end.

This exists to answer the standing objection to a simulability verdict on a model that
underperforms its published accuracy: maybe the circuit is bond-cheap only because it
never learned anything. If chi* stays flat while accuracy climbs from chance to its
ceiling, the verdict does not rest on the shortfall. If chi* instead rises with accuracy,
that is the more interesting finding and the audit should be reported that way.

Writes results/trajectory/<dataset>__<model>__seed<k>.json.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from omegaconf import OmegaConf  # noqa: E402

from simcert import models  # noqa: F401,E402  (populates the model registry)
from simcert.audit.metrics import accuracy  # noqa: E402
from simcert.audit.pipeline import audit_model  # noqa: E402
from simcert.data.loaders import build_vocab, load_dataset  # noqa: E402
from simcert.io_results import git_sha, lib_versions, save_result  # noqa: E402
from simcert.registry import get_model  # noqa: E402
from simcert.runner import load_config  # noqa: E402
from simcert.seed import seed_everything  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qsann")
    p.add_argument("--dataset", default="rp")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--audit", default=None, help="audit config group (default: per-model)")
    p.add_argument("--epochs", type=int, default=None, help="override model.epochs")
    p.add_argument("--snapshot-every", type=int, default=1, help="snapshot every k epochs")
    p.add_argument("--max-test", type=int, default=None,
                   help="cap audited test examples (recorded in the output)")
    p.add_argument("--keep-snapshots", action="store_true",
                   help="keep the per-epoch weights instead of deleting them afterwards")
    p.add_argument("overrides", nargs="*", help="extra dotlist overrides, e.g. model.lr=0.02")
    return p.parse_args()


def main():
    args = parse_args()
    audit_name = args.audit or ("claqs" if args.model == "claqs" else "default")
    argv = [f"model={args.model}", f"dataset={args.dataset}",
            f"seed={args.seed}", f"audit={audit_name}", *args.overrides]
    cfg = load_config(argv)
    if args.epochs is not None:
        cfg.model.epochs = int(args.epochs)
    # The twin retrain is a whole-run ablation, not a per-epoch one; chi* is all we need here.
    cfg.audit.entanglement_removal = False

    tag = f"{args.dataset}__{args.model}__seed{args.seed}"
    snap_root = REPO / "checkpoints" / "trajectory" / tag
    if snap_root.exists():
        shutil.rmtree(snap_root)
    cfg.model.snapshot_dir = str(snap_root)
    cfg.model.snapshot_every = int(args.snapshot_every)

    seed_everything(int(cfg.seed))
    ds = load_dataset(
        cfg.dataset_name,
        seed=int(cfg.seed),
        val_frac=float(cfg.dataset.val_frac),
        test_frac=float(cfg.dataset.get("test_frac", 0.2)),
        **{k: v for k, v in OmegaConf.to_container(cfg.dataset, resolve=True).items()
           if k not in ("name", "val_frac", "test_frac")},
    )
    test = ds.test[: args.max_test] if args.max_test else ds.test
    print(f"[trajectory] {tag}  dataset={ds.summary()}  audited_test={len(test)}")

    vocab = build_vocab(ds.train)
    model = get_model(cfg.model_name)()
    model.build(cfg.model, vocab)
    t0 = time.time()
    report = model.fit(ds.train, ds.val, cfg.model)
    print(f"[trajectory] trained {cfg.model.epochs} epochs in {time.time() - t0:.1f}s: "
          f"train={report.train_accuracy:.3f} val={report.val_accuracy:.3f}")

    snaps = sorted(snap_root.glob("epoch*"))
    if not snaps:
        raise SystemExit(f"no snapshots under {snap_root} — is the model's fit() hooked?")

    chi_values = list(cfg.audit.chi_values)
    points = []
    for snap in snaps:
        epoch = int(snap.name.removeprefix("epoch"))
        m = get_model(cfg.model_name).load(snap)
        cert, _ = audit_model(
            m, test, chi_values=chi_values, dataset_name=cfg.dataset_name,
            cutoff=float(cfg.audit.cutoff), seed=int(cfg.seed),
        )
        pt = {
            "epoch": epoch,
            "test_accuracy": cert.full_accuracy,
            # Recorded so early stopping can be assessed after the fact: these models
            # train for a fixed epoch count and keep the LAST weights, so a run whose
            # test accuracy peaks early is audited well past its best point.
            "val_accuracy": accuracy([ex.label for ex in ds.val], m.predict(ds.val))
            if ds.val else None,
            "chi_star": cert.chi_star.get("tau_gen"),
            "chi_star_agree": cert.chi_star.get("tau_agree"),
            "accuracy_by_chi": {str(k): v for k, v in cert.accuracy_by_chi.items()},
            "agreement_by_chi": {str(k): v for k, v in cert.agreement_by_chi.items()},
            "entropy_mean": cert.entropy_mean,
        }
        points.append(pt)
        ent = "n/a" if pt["entropy_mean"] is None else f"{pt['entropy_mean']:.3f}"
        val = "n/a" if pt["val_accuracy"] is None else f"{pt['val_accuracy']:.3f}"
        print(f"  epoch {epoch:>3}  acc={pt['test_accuracy']:.3f}  val={val}  "
              f"chi*={pt['chi_star']}  S={ent}")

    accs = [p["test_accuracy"] for p in points]
    stars = [p["chi_star"] for p in points]
    # Model selection after the fact. fit() keeps the last epoch's weights, so if val
    # accuracy peaks early the audited model is past its best; report both so the gap
    # to the published number can be attributed rather than guessed at.
    scored = [p for p in points if p["val_accuracy"] is not None]
    best = max(scored, key=lambda p: (p["val_accuracy"], -p["epoch"])) if scored else None
    out = {
        "model": cfg.model_name,
        "dataset": cfg.dataset_name,
        "seed": int(cfg.seed),
        "epochs": int(cfg.model.epochs),
        "snapshot_every": int(args.snapshot_every),
        "n_test_audited": len(test),
        "n_test_full": len(ds.test),
        "chi_values": [c if c is not None else "full" for c in chi_values],
        "final_train_accuracy": report.train_accuracy,
        "final_val_accuracy": report.val_accuracy,
        "accuracy_first": accs[0],
        "accuracy_last": accs[-1],
        "accuracy_min": min(accs),
        "accuracy_max": max(accs),
        "accuracy_declined": accs[-1] < accs[0],
        "best_val_epoch": best["epoch"] if best else None,
        "best_val_accuracy": best["val_accuracy"] if best else None,
        "test_accuracy_at_best_val": best["test_accuracy"] if best else None,
        "chi_star_at_best_val": best["chi_star"] if best else None,
        "chi_star_set": sorted({str(s) for s in stars}),
        "chi_star_constant": len(set(stars)) == 1,
        "points": points,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "env": {"git_sha": git_sha(), "libs": lib_versions()},
    }
    path = REPO / "results" / "trajectory" / f"{tag}.json"
    save_result(path, out)
    arrow = ("DECLINED" if accs[-1] < accs[0]
             else "flat" if accs[-1] == accs[0] else "improved")
    print(f"[trajectory] accuracy {accs[0]:.3f} -> {accs[-1]:.3f} ({arrow}; "
          f"range {min(accs):.3f}-{max(accs):.3f}), "
          f"chi* in {{{', '.join(out['chi_star_set'])}}} "
          f"({'constant' if out['chi_star_constant'] else 'VARIES'})")
    if best:
        print(f"[trajectory] best val at epoch {best['epoch']}: val={best['val_accuracy']:.3f} "
              f"test={best['test_accuracy']:.3f} chi*={best['chi_star']} "
              f"(audited model is epoch {points[-1]['epoch']}, test={accs[-1]:.3f})")
    print(f"[trajectory] wrote {path.relative_to(REPO)}")

    if not args.keep_snapshots:
        shutil.rmtree(snap_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
