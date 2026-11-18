"""Compare deleted-client gradient ascent with retained-logit KL regularization.

The original client-DP model is retrained for every tuple because the earlier
grid did not persist model checkpoints. Post-deletion optimization is explicitly
non-private; it reads deleted examples and retained feature batches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import torch
import torch.nn.functional as F
import yaml

from dp_forgetbench.attacks import loss_membership_probe
from dp_forgetbench.data import apply_client_deletion_requests
from dp_forgetbench.federated import evaluate_loss_accuracy, train_federated
from dp_forgetbench.privacy import make_ledger
from run_grid import _build_feature_federation, _config_sha, _new_head, _noise_for_epsilon, load_config

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))


def _run_unlearning(
    base_model: torch.nn.Module,
    forget_x: torch.Tensor,
    forget_y: torch.Tensor,
    retain_x: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    kl_weight: float,
    batch_size: int,
    seed: int,
) -> tuple[torch.nn.Module, dict]:
    device = next(base_model.parameters()).device
    model = type(base_model)(feature_dim=384, num_classes=10).to(device)
    model.load_state_dict(base_model.state_dict())
    model.train()
    retain_x = retain_x.to(device)
    forget_x, forget_y = forget_x.to(device), forget_y.to(device).long()
    with torch.no_grad():
        reference_probabilities = torch.softmax(base_model(retain_x), dim=1).detach()
    rng = torch.Generator(device=device).manual_seed(int(seed))
    param_list = [p for p in model.parameters() if p.requires_grad]
    initial_forget_loss = float(F.cross_entropy(model(forget_x), forget_y).detach().item())
    start = time.perf_counter()
    kl_values = []
    forget_losses = []
    for _ in range(steps):
        indices = torch.randperm(len(retain_x), generator=rng, device=device)[: min(batch_size, len(retain_x))]
        forget_loss = F.cross_entropy(model(forget_x), forget_y)
        retain_logits = model(retain_x[indices])
        retain_kl = F.kl_div(
            F.log_softmax(retain_logits, dim=1),
            reference_probabilities[indices],
            reduction="batchmean",
        )
        # Minimize -CE on the deletion set (therefore ascend CE) while keeping
        # retained predictions close to the pre-deletion model.
        objective = -forget_loss + kl_weight * retain_kl
        gradients = torch.autograd.grad(objective, param_list)
        with torch.no_grad():
            for parameter, gradient in zip(param_list, gradients):
                parameter.add_(gradient, alpha=-learning_rate)
        kl_values.append(float(retain_kl.detach().item()))
        forget_losses.append(float(forget_loss.detach().item()))
    with torch.no_grad():
        final_forget_loss = float(F.cross_entropy(model(forget_x), forget_y).item())
    return model.eval(), {
        "runtime_seconds": time.perf_counter() - start,
        "unlearning_steps": steps,
        "learning_rate": learning_rate,
        "retain_kl_weight": kl_weight,
        "retain_batch_size": min(batch_size, len(retain_x)),
        "forget_cross_entropy_before": initial_forget_loss,
        "forget_cross_entropy_after": final_forget_loss,
        "retain_kl_mean_during_optimization": sum(kl_values) / max(1, len(kl_values)),
        "forget_loss_mean_during_optimization": sum(forget_losses) / max(1, len(forget_losses)),
        "post_deletion_privacy": "non-private; uses deleted examples and retained features",
    }


def _evaluate(model, test, forget, retain, unseen, attack_seed):
    return {
        "test": evaluate_loss_accuracy(model, *test),
        "forgotten_client": evaluate_loss_accuracy(model, *forget),
        "retained_data": evaluate_loss_accuracy(model, *retain),
        "forgotten_privacy": loss_membership_probe(model, *forget, *unseen, seed=attack_seed),
        "retained_privacy": loss_membership_probe(model, *retain, *unseen, seed=attack_seed + 1),
    }


def _run_cell(base, cfg, alpha, epsilon, seed, cache):
    output_root = ROOT / cfg["output_root"]
    alpha_key = f"alpha{str(alpha).replace('.', 'p')}"
    eps_key = f"eps{int(epsilon)}"
    run_dir = output_root / alpha_key / eps_key / f"seed{seed}"
    metrics_path = run_dir / "metrics.json"
    signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if metrics_path.exists():
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists() and json.loads(meta_path.read_text()).get("method_config_sha256") == signature:
            return metrics_path
        raise FileExistsError(f"Refusing to overwrite existing run {run_dir}")
    stage = run_dir.with_name(f"{run_dir.name}.staging-{os.getpid()}-{uuid4().hex[:8]}")
    stage.mkdir(parents=True, exist_ok=False)

    federation = _build_feature_federation(base, alpha, seed, cache)
    deletion_cfg = {"client_id": int(base["deletion"]["client_id"]), "fraction": float(base["deletion"]["fraction"])}
    deletion = apply_client_deletion_requests(federation, deletion_cfg, seed)
    retained = deletion.retained
    retain_x = torch.cat([c.x for c in retained.values()])
    retain_y = torch.cat([c.y for c in retained.values()])
    fed = dict(base["federated"])
    epsilon_delta = float(base["privacy"]["delta"])
    sigma = _noise_for_epsilon(epsilon, int(fed["rounds"]), float(fed["client_sample_rate"]), epsilon_delta)
    privacy = {
        "enabled": True,
        "population_size": len(federation.clients),
        "delta": epsilon_delta,
        "clip_norm": float(base["privacy"]["clip_norm"]),
        "noise_multiplier": sigma,
        "sampling": str(base["privacy"]["sampling"]),
        "release_count": int(base["privacy"]["release_count"]),
        "secure_aggregation": False,
        "secure_rng": bool(base["privacy"].get("secure_rng", False)),
    }
    ledger = make_ledger(privacy, fed)
    print(f"FIT original DP alpha={alpha:g} eps={epsilon:g} seed={seed}", flush=True)
    dp_model, _dp_cost = train_federated(
        clients=federation.clients,
        n_features=384,
        federated_config=fed,
        privacy_config=privacy,
        seed=seed,
        model=_new_head(seed),
    )
    print(f"UNLEARN GA+retain-KL alpha={alpha:g} eps={epsilon:g} seed={seed}", flush=True)
    candidate, diagnostics = _run_unlearning(
        dp_model,
        deletion.forgotten_x,
        deletion.forgotten_y,
        retain_x,
        steps=int(cfg["unlearning_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        kl_weight=float(cfg["retain_kl_weight"]),
        batch_size=int(cfg["retain_batch_size"]),
        seed=seed + 40_000,
    )
    test = (federation.test_x, federation.test_y)
    forget = (deletion.forgotten_x, deletion.forgotten_y)
    retain = (retain_x, retain_y)
    unseen = (federation.audit_x, federation.audit_y)
    dp_metrics = _evaluate(dp_model, test, forget, retain, unseen, seed + 30_000)
    method_metrics = _evaluate(candidate, test, forget, retain, unseen, seed + 30_000)
    ref_path = ROOT / cfg["reference_root"] / alpha_key / eps_key / f"seed{seed}" / "metrics.json"
    reference = json.loads(ref_path.read_text()) if ref_path.exists() else None
    metrics = {
        "dp_only_replay": dp_metrics,
        "gradient_ascent_retain_kl": {**method_metrics, "method_diagnostics": diagnostics},
        "existing_grid_reference": None if reference is None else {
            key: {subset: reference[key][subset] for subset in ("test", "forgotten_client", "retained_data")}
            for key in (
                "dp_only_no_action", "retained_finetune_baseline",
                "cached_update_reconstruction_baseline", "federated_eraser_baseline", "target_retrain",
            ) if key in reference
        },
    }
    metadata = {
        "experiment_name": cfg["experiment_name"],
        "alpha": float(alpha),
        "epsilon_target": float(epsilon),
        "epsilon_actual": float(ledger.epsilon),
        "noise_multiplier": sigma,
        "seed": int(seed),
        "deletion_requests": [{"client_id": deletion_cfg["client_id"], "fraction": deletion_cfg["fraction"]}],
        "training_config_sha256": _config_sha(base),
        "method_config_sha256": signature,
        "architecture": {
            "backbone": "facebookresearch/dinov2_vits14 frozen feature extractor",
            "feature_dimension": 384,
            "head": "cosine-normalized linear classifier, 3,850 trainable parameters",
        },
        "privacy": {
            "original_training_ledger": ledger.as_dict(),
            "post_deletion_method_is_private": False,
            "explanation": "The gradient-ascent/KL stage reads the deleted client's examples and retained data after deletion. The original DP ledger does not cover this stage or its metrics.",
        },
        "method": {
            "description": "Gradient ascent on deleted-client cross-entropy, regularized by KL distillation to the original model on retained feature batches.",
            "status": "exploratory non-private unlearning baseline; compare against target retrain and existing unlearning methods.",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (stage / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (stage / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    stage.replace(run_dir)
    print(
        f"DONE alpha={alpha:g} eps={epsilon:g} seed={seed} "
        f"DP test={dp_metrics['test']['accuracy']:.4f} forget={dp_metrics['forgotten_client']['accuracy']:.4f}; "
        f"GA-KL test={method_metrics['test']['accuracy']:.4f} forget={method_metrics['forgotten_client']['accuracy']:.4f}",
        flush=True,
    )
    return metrics_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "ga_retain_kl_config.yaml"))
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--only-alpha", type=float, action="append")
    parser.add_argument("--only-epsilon", type=float, action="append")
    parser.add_argument("--only-seed", type=int, action="append")
    args = parser.parse_args()
    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        parser.error("worker-id must be in [0, num-workers)")
    cfg = yaml.safe_load(Path(args.config).resolve().read_text())
    base = load_config((ROOT / cfg["base_config"]).resolve())
    cache_path = (ROOT / base["output"]["feature_cache"]).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    alphas = args.only_alpha or cfg["alpha_values"]
    epsilons = args.only_epsilon or cfg["epsilon_values"]
    seeds = args.only_seed or cfg["seeds"]
    tasks = [(float(a), float(e), int(s)) for a in alphas for e in epsilons for s in seeds]
    failures = []
    for index, (alpha, epsilon, seed) in enumerate(tasks):
        if index % args.num_workers != args.worker_id:
            continue
        try:
            _run_cell(base, cfg, alpha, epsilon, seed, cache)
        except Exception as error:
            failures.append({"alpha": alpha, "epsilon": epsilon, "seed": seed, "error": repr(error), "traceback": traceback.format_exc()})
            print(f"FAILED alpha={alpha:g} eps={epsilon:g} seed={seed}: {error}", flush=True)
    if failures:
        out = ROOT / cfg["output_root"]
        out.mkdir(parents=True, exist_ok=True)
        (out / f"failures_worker{args.worker_id}.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(f"{len(failures)} task(s) failed")
    print(f"Worker {args.worker_id} completed {len(tasks[args.worker_id::args.num_workers])} assigned tuples.", flush=True)


if __name__ == "__main__":
    main()
