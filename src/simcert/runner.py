"""Config-driven entry point: ``train`` | ``audit`` | ``both``.

Lightweight, Hydra-compatible config loading (config groups + dotlist overrides via
OmegaConf) without Hydra's working-directory rewriting — so results land at stable,
committed paths keyed by a deterministic run hash.

Usage:
    python -m simcert.runner mode=both model=vqc_text dataset=mc seed=1
    python -m simcert.runner mode=both model=vqc_text audit.chi_values=[1,2,4,8,null]
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from . import models  # noqa: F401  (import populates the model registry)
from .audit.pipeline import audit_model
from .data.loaders import build_vocab, load_dataset
from .io_results import git_sha, lib_versions, result_path, run_hash, save_result
from .registry import get_model
from .seed import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
_SELECTORS = {"mode", "model", "dataset", "audit", "seed"}


def load_config(argv: list[str]):
    overrides = [a for a in argv if "=" in a]
    sel = dict(a.split("=", 1) for a in overrides if a.split("=", 1)[0] in _SELECTORS)
    dotlist = [a for a in overrides if a.split("=", 1)[0] not in _SELECTORS]

    base = OmegaConf.load(CONFIGS / "config.yaml")
    if "mode" in sel:
        base.mode = sel["mode"]
    if "seed" in sel:
        base.seed = int(sel["seed"])
    model_name = sel.get("model", base.model_name)
    dataset_name = sel.get("dataset", base.dataset_name)
    audit_name = sel.get("audit", base.audit_name)
    base.model_name, base.dataset_name, base.audit_name = model_name, dataset_name, audit_name

    cfg = OmegaConf.merge(
        base,
        {
            "model": OmegaConf.load(CONFIGS / "model" / f"{model_name}.yaml"),
            "dataset": OmegaConf.load(CONFIGS / "dataset" / f"{dataset_name}.yaml"),
            "audit": OmegaConf.load(CONFIGS / "audit" / f"{audit_name}.yaml"),
        },
    )
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg.model.seed = cfg.seed  # models read seed from their own namespace
    cfg.model.dataset = cfg.dataset_name  # e.g. DisCoCat consumer locates its artifacts
    return cfg


def _ckpt_dir(cfg, rhash):
    return REPO_ROOT / cfg.paths.checkpoints / f"{cfg.dataset_name}__{cfg.model_name}__{rhash}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config(argv)
    seed_everything(int(cfg.seed))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    rhash = run_hash(cfg_dict)
    print(f"[simcert] mode={cfg.mode} model={cfg.model_name} dataset={cfg.dataset_name} hash={rhash}")

    ds_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(cfg.dataset, resolve=True).items()
        if k not in ("name", "val_frac", "test_frac")
    }
    ds = load_dataset(
        cfg.dataset_name,
        seed=int(cfg.seed),
        val_frac=float(cfg.dataset.val_frac),
        test_frac=float(cfg.dataset.get("test_frac", 0.2)),
        **ds_kwargs,
    )
    print(f"[simcert] dataset: {ds.summary()}")
    vocab = build_vocab(ds.train)

    model = get_model(cfg.model_name)()
    ckpt = _ckpt_dir(cfg, rhash)

    report = None
    if cfg.mode in ("train", "both"):
        model.build(cfg.model, vocab)
        report = model.fit(ds.train, ds.val, cfg.model)
        print(f"[simcert] trained: train={report.train_accuracy:.3f} val={report.val_accuracy:.3f}")
        model.save(ckpt)
    if cfg.mode == "audit":
        model = get_model(cfg.model_name).load(ckpt)

    if cfg.mode in ("audit", "both"):
        # matched classical baseline on the same splits
        baseline_preds = None
        baseline_name = cfg.model.get("baseline")
        if baseline_name:
            baseline = get_model(baseline_name)()
            baseline.fit(ds.train, ds.val, cfg.model)
            baseline_preds = baseline.predict(ds.test)

        # Axis B: entanglement-removal ablation -- retrain a separable (product-state) twin
        from .audit.metrics import accuracy as _acc

        labels_test = [ex.label for ex in ds.test]
        delta_ent = None
        twin_test_acc = None
        if cfg.audit.get("entanglement_removal", True):
            twin_cfg = OmegaConf.merge(cfg.model, {"entangling": False})
            twin = get_model(cfg.model_name)()
            twin.build(twin_cfg, vocab)
            twin.fit(ds.train, ds.val, twin_cfg)
            twin_test_acc = _acc(labels_test, twin.predict(ds.test))
            full_test_acc = _acc(labels_test, model.predict(ds.test))
            delta_ent = full_test_acc - twin_test_acc
            print(f"[simcert] entanglement-removal: full={full_test_acc:.3f} "
                  f"separable={twin_test_acc:.3f} delta_ent={delta_ent:+.3f}")

        cert, details = audit_model(
            model,
            ds.test,
            chi_values=list(cfg.audit.chi_values),
            dataset_name=cfg.dataset_name,
            cutoff=float(cfg.audit.cutoff),
            train_accuracy=report.train_accuracy if report else None,
            baseline_preds=baseline_preds,
            delta_ent=delta_ent,
            seed=int(cfg.seed),
            meta={"run_hash": rhash, "separable_test_accuracy": twin_test_acc},
        )

        # Axis E: data-reuploading Fourier degree (models exposing a scalar_response hook)
        if hasattr(model, "scalar_response"):
            from .audit.fourier import effective_degree

            cert.fourier_degree = effective_degree(model.scalar_response)
        # Axis D: geometric difference g_CQ / sqrt(N) (models exposing kernel_data)
        if hasattr(model, "kernel_data"):
            from .audit.kernels import fidelity_kernel, geometric_difference, rbf_kernel

            kstates, kfeats = model.kernel_data(ds.test)
            gcq = geometric_difference(fidelity_kernel(kstates), rbf_kernel(kfeats))
            cert.gcq_ratio = gcq / (len(ds.test) ** 0.5)

        out = {
            "run_hash": rhash,
            "config": cfg_dict,
            "dataset": ds.summary(),
            "train_report": report.__dict__ if report else None,
            "certificate": cert.to_dict(),
            "details": details,
            "env": {"git_sha": git_sha(), "libs": lib_versions()},
        }
        path = result_path(REPO_ROOT / cfg.paths.results, cfg.dataset_name, cfg.model_name, rhash)
        save_result(path, out)
        cert_path = (
            REPO_ROOT / cfg.paths.results / "certificates"
            / f"{cfg.dataset_name}__{cfg.model_name}__{rhash}.json"
        )
        save_result(cert_path, cert.to_dict())
        print(f"[simcert] VERDICT: {cert.verdict}  chi*={cert.chi_star}  full_acc={cert.full_accuracy:.3f}")
        print(f"[simcert] wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
