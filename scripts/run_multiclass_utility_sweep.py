"""Find a useful finite-epsilon multiclass client-DP training regime.

This is deliberately training-only. It does not run deletion, unlearning,
MUB, or membership inference. The achieved epsilon is recomputed from the
repository accountant for every row and is the authoritative privacy value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dp_forgetbench.config import load_config
from dp_forgetbench.data import make_cifar10_multiclass_federation
from dp_forgetbench.federated import evaluate_loss_accuracy, get_device, set_seed, train_federated
from dp_forgetbench.privacy import make_ledger, poisson_gaussian_epsilon


def calibrate_noise(target_epsilon: float, sample_rate: float, rounds: int, delta: float) -> tuple[float, float]:
    low, high = 1e-4, 1.0
    while poisson_gaussian_epsilon(sample_rate=sample_rate, noise_multiplier=high, rounds=rounds, delta=delta) > target_epsilon:
        high *= 2.0
        if high > 1e5:
            raise RuntimeError(f"Could not bracket target epsilon {target_epsilon}.")
    for _ in range(50):
        middle = (low + high) / 2.0
        epsilon = poisson_gaussian_epsilon(sample_rate=sample_rate, noise_multiplier=middle, rounds=rounds, delta=delta)
        if epsilon > target_epsilon:
            low = middle
        else:
            high = middle
    achieved = poisson_gaussian_epsilon(sample_rate=sample_rate, noise_multiplier=high, rounds=rounds, delta=delta)
    return high, achieved


def run(config_path: Path, output_dir: Path | None = None) -> list[dict]:
    config = load_config(config_path)
    evaluation = config.get("evaluation", {})
    epsilons = [float(value) for value in evaluation.get("epsilons", [8, 12, 16, 20, 24, 32])]
    seeds = [int(value) for value in evaluation.get("seeds", [config["seed"]])]
    rounds_values = [int(value) for value in evaluation.get("rounds_values", [config["federated"]["rounds"]])]
    output_dir = output_dir or Path(config["output"]["root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config = config["data"]
    fed_config = config["federated"]
    base_privacy = config["privacy"]
    device = get_device()
    if bool(config.get("execution", {}).get("require_cuda", False)) and device.type != "cuda":
        raise RuntimeError(
            f"CUDA is required for this sweep, but the active device is {device}. "
            "Ensure an NVIDIA GPU, updated drivers, and a CUDA-enabled PyTorch build are installed."
        )
    records: list[dict] = []

    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()})")
    print(f"Model: {config['model']['name']}; epsilons={epsilons}; rounds={rounds_values}; seeds={seeds}")

    for seed in seeds:
        set_seed(seed)
        federation = make_cifar10_multiclass_federation(data_config, seed)
        for rounds in rounds_values:
            fed_config = {**config["federated"], "rounds": rounds}
            for target_epsilon in epsilons:
                started = time.perf_counter()
                private = not math.isinf(target_epsilon)
                if private:
                    sigma, achieved_epsilon = calibrate_noise(
                        target_epsilon,
                        float(fed_config["client_sample_rate"]),
                        rounds,
                        float(base_privacy["delta"]),
                    )
                else:
                    sigma, achieved_epsilon = 0.0, float("inf")
                privacy_config = {
                    **base_privacy,
                    "enabled": private,
                    "population_size": int(data_config["n_clients"]),
                    "noise_multiplier": sigma,
                }
                ledger = make_ledger(privacy_config, fed_config).as_dict()
                model, cost = train_federated(
                    clients=federation.clients,
                    n_features=None,
                    federated_config=fed_config,
                    privacy_config=privacy_config,
                    seed=seed,
                    model_config=config["model"],
                )
                utility = evaluate_loss_accuracy(model, federation.test_x, federation.test_y)
                invalid_threshold = float(evaluation.get("validity_threshold_invalid", 0.25))
                warning_threshold = float(evaluation.get("validity_threshold_warning", 0.40))
                if utility["accuracy"] < invalid_threshold:
                    status, regime = "INVALID", "COLLAPSE"
                elif utility["accuracy"] < warning_threshold:
                    status, regime = "WARNING", "MARGINAL"
                else:
                    status, regime = "VALID", "HEALTHY"
                record = {
                    "model": config["model"]["name"],
                    "seed": seed,
                    "epsilon_target": target_epsilon,
                    "epsilon_actual": ledger["epsilon"] if private else achieved_epsilon,
                    "delta": ledger["delta"],
                    "noise_multiplier": sigma,
                    "clip_norm": privacy_config["clip_norm"],
                    "sample_rate": fed_config["client_sample_rate"],
                    "population_size": data_config["n_clients"],
                    "samples_per_client": data_config["samples_per_client"],
                    "rounds": rounds,
                    "local_epochs": fed_config["local_epochs"],
                    "aggregation_normalizer": ledger["aggregation_normalizer"],
                    "test_accuracy": utility["accuracy"],
                    "test_loss": utility["loss"],
                    "validity_status": status,
                    "regime": regime,
                    "runtime_seconds": round(time.perf_counter() - started, 3),
                    "train_cost": cost.as_dict(),
                    "device": str(device),
                    "torch_version": torch.__version__,
                }
                records.append(record)
                print(f"epsilon={target_epsilon:g} actual={record['epsilon_actual']:.4f} seed={seed} accuracy={utility['accuracy'] * 100:.2f}% [{status}]")

    (output_dir / "utility_sweep.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    fieldnames = sorted({key for record in records for key in record if key != "train_cost"})
    with (output_dir / "utility_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multiclass_resnet18_utility_sweep.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()