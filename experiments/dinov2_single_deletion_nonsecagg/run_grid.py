"""Full-schema single-client deletion grid on frozen resnet-50 features.

Each alpha/epsilon/seed tuple is an independent experiment from the original
100-client federation. Outputs preserve the standard Phase-2 metrics.json
schema. Use two or more processes with distinct --worker-id values to parallelize;
each process should receive its own CUDA_VISIBLE_DEVICES assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dp_forgetbench.data import (
    ClientDataset,
    Federation,
    apply_client_deletion_requests,
    make_cifar10_multiclass_federation,
)
from dp_forgetbench.evaluation import compute_method_mub, evaluate_against_target
from dp_forgetbench.federated import (
    FederatedHistory,
    clone_state,
    finetune_retained,
    federated_eraser,
    get_device,
    reconstruct_from_cached_updates,
    set_seed,
    train_federated,
)
from dp_forgetbench.privacy import make_ledger, poisson_gaussian_epsilon

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


class DINOFeatureLinearHead(nn.Module):
    """The existing DINO Tier-2 cosine linear head, detached from its encoder.

    Embeddings are produced by FrozenBackboneLinear._encode and have unit norm.
    Keeping only this classifier in FL makes local updates and communication
    account for the 384 x 10 weights plus bias, not the frozen DINO checkpoint.
    """

    def __init__(self, feature_dim: int = 384, num_classes: int = 10) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(F.normalize(features, p=2, dim=-1))


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _config_sha(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _noise_for_epsilon(epsilon: float, rounds: int, sample_rate: float, delta: float) -> float:
    lo, hi = 0.01, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        actual = poisson_gaussian_epsilon(
            sample_rate=sample_rate, noise_multiplier=mid, rounds=rounds, delta=delta
        )
        if actual > epsilon:
            lo = mid
        else:
            hi = mid
    actual = poisson_gaussian_epsilon(
        sample_rate=sample_rate, noise_multiplier=hi, rounds=rounds, delta=delta
    )
    if abs(actual - epsilon) > max(1e-6, epsilon * 1e-5):
        raise RuntimeError(f"Could not calibrate requested epsilon={epsilon}: got {actual}")
    return hi


def _build_feature_federation(config: dict, alpha: float, seed: int, cache: dict) -> Federation:
    data_cfg = {
        **config["data"],
        "dataset_root": str((ROOT / config["dataset_root"]).resolve()),
        "download": False,
        "partition_type": "dirichlet",
        "dirichlet_alpha": float(alpha),
    }
    image_federation = make_cifar10_multiclass_federation(data_cfg, seed)
    manifest = image_federation.partition_manifest
    if manifest is None:
        raise RuntimeError("CIFAR-10 partition did not produce a source-index manifest")
    train_features = cache["train_features"]
    test_features = cache["test_features"]
    # Preserve the exact labels/order from the partitioner while replacing only images with cached embeddings.
    clients = {
        int(client_id): ClientDataset(
            int(client_id),
            train_features[torch.tensor(indices, dtype=torch.long)],
            image_federation.clients[int(client_id)].y,
        )
        for client_id, indices in manifest["client_train_indices"].items()
    }
    audit_indices = torch.tensor(manifest["audit_train_indices"], dtype=torch.long)
    test_indices = torch.tensor(manifest["test_indices"], dtype=torch.long)
    return Federation(
        clients=clients,
        test_x=test_features[test_indices],
        test_y=cache["test_labels"][test_indices],
        audit_x=train_features[audit_indices],
        audit_y=cache["train_labels"][audit_indices],
        partition_manifest=manifest,
        validation_x=None,
        validation_y=None,
    )


def _new_head(seed: int) -> DINOFeatureLinearHead:
    set_seed(seed)
    return DINOFeatureLinearHead(feature_dim=384, num_classes=10)


def _round_progress(alpha: float, epsilon: float, seed: int, method: str, total_rounds: int):
    def report(round_number: int, _model: nn.Module) -> None:
        if round_number % 10 == 0 or round_number == total_rounds:
            print(
                f"TRAIN alpha={alpha:g} epsilon={epsilon:g} seed={seed} "
                f"method={method} round={round_number}/{total_rounds}",
                flush=True,
            )
    return report


def _run_one(config: dict, config_path: Path, alpha: float, epsilon: float, seed: int, cache: dict) -> Path:
    output_root = ROOT / config["output"]["root"]
    alpha_key = f"alpha{str(alpha).replace('.', 'p')}"
    eps_key = f"eps{int(epsilon)}"
    run_dir = output_root / alpha_key / eps_key / f"seed{seed}"
    metrics_path = run_dir / "metrics.json"
    config_hash = _config_sha(config)
    if metrics_path.exists():
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text())
            if existing.get("grid_config_sha256") == config_hash:
                return metrics_path
        raise FileExistsError(f"Refusing to overwrite an existing run with a different or missing config fingerprint: {run_dir}")
    staging_dir = run_dir.with_name(f"{run_dir.name}.staging-{os.getpid()}")
    staging_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = staging_dir / "metrics.json"

    federation = _build_feature_federation(config, alpha, seed, cache)
    print(f"START alpha={alpha:g} epsilon={epsilon:g} seed={seed}", flush=True)
    deletion_cfg = {"client_id": int(config["deletion"]["client_id"]), "fraction": float(config["deletion"]["fraction"])}
    deletion = apply_client_deletion_requests(federation, deletion_cfg, seed)
    retained = deletion.retained
    retained_x = torch.cat([client.x for client in retained.values()])
    retained_y = torch.cat([client.y for client in retained.values()])
    n_clients = len(federation.clients)
    sample_rate = float(config["federated"]["client_sample_rate"])
    rounds = int(config["federated"]["rounds"])
    delta = float(config["privacy"]["delta"])
    noise_multiplier = _noise_for_epsilon(epsilon, rounds, sample_rate, delta)
    privacy = {
        "enabled": True,
        "population_size": n_clients,
        "delta": delta,
        "clip_norm": float(config["privacy"]["clip_norm"]),
        "noise_multiplier": noise_multiplier,
        "sampling": str(config["privacy"]["sampling"]),
        "release_count": int(config["privacy"]["release_count"]),
        "secure_aggregation": bool(config["privacy"]["secure_aggregation"]),
        "secure_rng": bool(config["privacy"].get("secure_rng", False)),
    }
    fed = dict(config["federated"])
    ledger = make_ledger(privacy, fed)
    if abs(float(ledger.epsilon) - epsilon) > max(1e-6, epsilon * 1e-5):
        raise RuntimeError(f"Privacy ledger epsilon mismatch: target={epsilon}, actual={ledger.epsilon}")

    # Full DP model: retain history only as the reference implementation does.
    history = FederatedHistory()
    print(f"FIT dp_only alpha={alpha:g} epsilon={epsilon:g} seed={seed}", flush=True)
    full_model, full_cost = train_federated(
        clients=federation.clients,
        n_features=384,
        federated_config=fed,
        privacy_config=privacy,
        seed=seed,
        history=history,
        model=_new_head(seed),
        round_callback=_round_progress(alpha, epsilon, seed, "dp_only", rounds),
    )

    target_privacy = dict(privacy, population_size=len(retained))
    target_ledger = make_ledger(target_privacy, fed)
    ensemble_size = int(config.get("evaluation", {}).get("retrain_ensemble_size", 5))
    print(f"FIT target_retrain 1/{ensemble_size} alpha={alpha:g} epsilon={epsilon:g} seed={seed}", flush=True)
    target_model, target_cost = train_federated(
        clients=retained,
        n_features=384,
        federated_config=fed,
        privacy_config=target_privacy,
        seed=seed + 10_000,
        model=_new_head(seed + 10_000),
        round_callback=_round_progress(alpha, epsilon, seed, "target_retrain_1", rounds),
    )
    ensemble_models, ensemble_costs = [], []
    for index in range(1, ensemble_size):
        init_seed = seed + 10_000 + index
        print(f"FIT target_retrain {index + 1}/{ensemble_size} alpha={alpha:g} epsilon={epsilon:g} seed={seed}", flush=True)
        model, cost = train_federated(
            clients=retained,
            n_features=384,
            federated_config=fed,
            privacy_config=target_privacy,
            seed=init_seed,
            model=_new_head(init_seed),
            round_callback=_round_progress(alpha, epsilon, seed, f"target_retrain_{index + 1}", rounds),
        )
        ensemble_models.append(model)
        ensemble_costs.append(cost)

    print(f"FIT retained_finetune alpha={alpha:g} epsilon={epsilon:g} seed={seed}", flush=True)
    tuned_model, tuned_cost = finetune_retained(
        full_model=full_model,
        clients=retained,
        n_features=384,
        federated_config=fed,
        unlearning_rounds=int(config["unlearning"]["rounds"]),
        seed=seed + 20_000,
    )
    forgotten_client_ids = {int(deletion_cfg["client_id"])}
    print(f"UNLEARN cached_reconstruction alpha={alpha:g} epsilon={epsilon:g} seed={seed}", flush=True)
    cached_model, cached_cost = reconstruct_from_cached_updates(
        history=history,
        forgotten_client_ids=forgotten_client_ids,
        n_features=384,
        federated_config=fed,
        privacy_config=privacy,
        model_template=_new_head(seed),
    )
    print(f"UNLEARN federaser alpha={alpha:g} epsilon={epsilon:g} seed={seed}", flush=True)
    eraser_model, eraser_cost = federated_eraser(
        history=history,
        clients=retained,
        forgotten_client_ids=forgotten_client_ids,
        n_features=384,
        federated_config=fed,
        privacy_config=privacy,
        calibration_ratio=float(config.get("unlearning", {}).get("calibration_ratio", 0.5)),
        model_template=_new_head(seed),
    )
    common = {
        "target": target_model,
        "test_x": federation.test_x,
        "test_y": federation.test_y,
        "forgotten_x": deletion.forgotten_x,
        "forgotten_y": deletion.forgotten_y,
        "retained_x": retained_x,
        "retained_y": retained_y,
        "unseen_x": federation.audit_x,
        "unseen_y": federation.audit_y,
        "pre_deletion_model": full_model,
        "attack_seed": seed + 30_000,
    }
    metrics = {
        "dp_only_no_action": evaluate_against_target(
            candidate=full_model, cost=full_cost, **common,
            assumptions="No post-deletion action; resnet-50-S/14 frozen features with a client-DP-trained cosine linear head.",
            privacy_implications="Head checkpoint is covered by its recorded client-DP ledger under stated assumptions. Feature extraction is fixed and frozen. Data-dependent metrics are not covered by the model ledger.",
        ),
        "retained_finetune_baseline": evaluate_against_target(
            candidate=tuned_model, cost=tuned_cost, **common,
            assumptions="Starts from full head and runs the configured retained-data fine-tuning rounds.",
            privacy_implications="Accesses retained client features after the request and is not DP post-processing; this matches the reference runner's non-private fine-tuning baseline.",
        ),
        "target_retrain": evaluate_against_target(
            candidate=target_model, cost=target_cost, **common,
            assumptions="Gold-standard internal reference trained from scratch on the 99 retained clients.",
            privacy_implications="Internal benchmark reference; independently trained under its own 99-client ledger and not an additional public release under the original-model ledger.",
        ),
        "target_retrain_ensemble": [
            evaluate_against_target(
                candidate=model, cost=cost, **common,
                assumptions="Independent retained-only retrain used to quantify retrain variability.",
                privacy_implications="Internal reference model under its own training ledger; not covered by the original-model ledger.",
            )
            for model, cost in zip(ensemble_models, ensemble_costs)
        ],
        "cached_update_reconstruction_baseline": evaluate_against_target(
            candidate=cached_model, cost=cached_cost, **common,
            assumptions="Directly re-aggregates cached clipped per-client updates from retained participants; no retained-client recalibration.",
            privacy_implications=(
                "Requires the server to retain individual client updates, so this run has no Secure Aggregation protection against the server. "
                "The central-DP training ledger is not a guarantee against a server that observes raw per-client updates. "
                "The reconstruction reuses the original per-round Gaussian noise; releasing both checkpoints can expose correlated information."
            ),
        ),
        "federated_eraser_baseline": evaluate_against_target(
            candidate=eraser_model, cost=eraser_cost, **common,
            assumptions="FedEraser-style recalibration uses retained clients' historical updates and local calibration training at each recorded round; implementation fidelity is unverified.",
            privacy_implications=(
                "Requires the server to retain individual client updates and query retained clients after deletion; this run has no Secure Aggregation protection against the server. "
                "The central-DP training ledger is not a guarantee against a server that observes raw per-client updates. "
                "Reuses original round noise, so joint release of original and erased checkpoints needs separate privacy analysis."
            ),
        ),
    }
    all_targets = [metrics["target_retrain"], *metrics["target_retrain_ensemble"]]
    mub_summary = {}
    evaluated_methods = (
        "dp_only_no_action",
        "retained_finetune_baseline",
        "cached_update_reconstruction_baseline",
        "federated_eraser_baseline",
    )
    for method in evaluated_methods:
        mub = compute_method_mub(metrics[method], metrics["dp_only_no_action"], all_targets)
        metrics[method]["mub"] = mub
        mub_summary[method] = mub
    metrics["mub_summary"] = mub_summary
    metrics["full_model"] = metrics["dp_only_no_action"]

    metadata = {
        "experiment_name": config["experiment_name"],
        "grid_config_sha256": config_hash,
        "config_path": str(config_path),
        "alpha": float(alpha),
        "epsilon_target": float(epsilon),
        "epsilon_actual": float(ledger.epsilon),
        "noise_multiplier": noise_multiplier,
        "seed": seed,
        "deletion_requests": [{"client_id": deletion_cfg["client_id"], "fraction": deletion_cfg["fraction"], "request_index": 0}],
        "partition": federation.partition_manifest,
        "architecture": {
            "backbone": "facebookresearch/resnet-50_vits14",
            "backbone_frozen": True,
            "feature_dimension": 384,
            "head": "cosine-normalized linear classifier, 3,850 trainable parameters",
            "pretrained_checkpoint_sha256": "see cached PyTorch Hub checkpoint; checkpoint file is not copied into this result",
        },
        "effective_privacy_config": privacy,
        "reference_privacy_config": target_privacy,
        "limitations": [
            "Exploratory frozen-feature client-head experiment; it does not unlearn or retrain the resnet-50 backbone.",
            "Secure Aggregation is disabled for this run to make individual client update histories available to the two historical-update unlearning methods; the server can observe per-client updates.",
            "Cached Reconstruction and FedEraser reuse stored per-round noise. Joint release of the original and unlearned checkpoints is not certified by the original training ledger alone.",
            "The metrics file contains data-dependent privacy/utility diagnostics not covered by the model-training ledger.",
            "Frozen features are cached centrally for computation; this cache is not a differentially private release.",
            "Fine-tuning uses retained data and is non-private, matching the existing reference runner semantics.",
            "The tri-population probe is a custom diagnostic, not a validated TC-UMIA reproduction.",
        ],
        "validity_status": "EXPLORATORY_resnet-50_FEATURE_HEAD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp_metrics = metrics_path.with_suffix(".json.tmp")
    tmp_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_metrics.replace(metrics_path)
    (staging_dir / "privacy_ledger.json").write_text(json.dumps(ledger.as_dict(), indent=2, sort_keys=True) + "\n")
    (staging_dir / "reference_privacy_ledger.json").write_text(json.dumps(target_ledger.as_dict(), indent=2, sort_keys=True) + "\n")
    (staging_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    staging_dir.replace(run_dir)
    print(
        f"DONE alpha={alpha:g} epsilon={epsilon:g} seed={seed} "
        f"dp_test={metrics['dp_only_no_action']['test']['accuracy']:.4f} "
        f"cached_test={metrics['cached_update_reconstruction_baseline']['test']['accuracy']:.4f} "
        f"eraser_test={metrics['federated_eraser_baseline']['test']['accuracy']:.4f} -> {run_dir / 'metrics.json'}",
        flush=True,
    )
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--only-alpha", type=float, action="append")
    parser.add_argument("--only-epsilon", type=float, action="append")
    parser.add_argument("--only-seed", type=int, action="append")
    args = parser.parse_args()
    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        parser.error("worker-id must be in [0, num-workers)")
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    cache_path = (ROOT / config["output"]["feature_cache"]).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found. Run prepare_features.py first: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    alphas = args.only_alpha or [float(x) for x in config["alpha_values"]]
    epsilons = args.only_epsilon or [float(x) for x in config["epsilon_values"]]
    seeds = args.only_seed or [int(x) for x in config["seeds"]]
    tasks = [(alpha, epsilon, seed) for alpha in alphas for epsilon in epsilons for seed in seeds]
    failures = []
    for task_index, (alpha, epsilon, seed) in enumerate(tasks):
        if task_index % args.num_workers != args.worker_id:
            continue
        try:
            print(f"QUEUE worker={args.worker_id} task={task_index + 1}/{len(tasks)}", flush=True)
            _run_one(config, config_path, alpha, epsilon, seed, cache)
        except Exception as error:
            failures.append({"alpha": alpha, "epsilon": epsilon, "seed": seed, "error": repr(error), "traceback": traceback.format_exc()})
            print(f"FAILED alpha={alpha:g} epsilon={epsilon:g} seed={seed}: {error}", flush=True)
    out_root = ROOT / config["output"]["root"]
    if failures:
        fail_path = out_root / f"failures_worker{args.worker_id}.json"
        fail_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} grid cells failed; see {fail_path}")
    print(f"Worker {args.worker_id} finished its assigned {len(tasks[args.worker_id::args.num_workers])} cells.", flush=True)


if __name__ == "__main__":
    main()
