"""Audit the non-SecAgg FU grid and emit per-run and grouped CSV summaries."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/resnet-50_single_deletion_nonsecagg_full_metrics"
METHODS = (
    "dp_only_no_action",
    "retained_finetune_baseline",
    "cached_update_reconstruction_baseline",
    "federated_eraser_baseline",
    "target_retrain",
)
FIELDS = (
    "test_accuracy", "forgotten_accuracy", "retained_accuracy",
    "forgotten_mia_auc", "forgotten_mia_advantage", "retained_mia_auc",
    "retained_mia_advantage", "unseen_forget_vs_unseen_auc",
    "unseen_probe_accuracy", "mub_test_accuracy", "mub_mia_advantage",
)
EXPECTED_TOP = {
    "dp_only_no_action", "retained_finetune_baseline", "target_retrain",
    "target_retrain_ensemble", "cached_update_reconstruction_baseline",
    "federated_eraser_baseline", "mub_summary", "full_model",
}


def finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {number}")
    return number


def main() -> None:
    config = yaml.safe_load((ROOT / "experiments/resnet-50_single_deletion_nonsecagg/config.yaml").read_text())
    if config["federated"].get("secure_aggregation") or config["privacy"].get("secure_aggregation"):
        raise ValueError("non-SecAgg grid config unexpectedly enables secure aggregation")
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    expected_keys = {
        (float(a), float(e), int(s))
        for a in config["alpha_values"] for e in config["epsilon_values"] for s in config["seeds"]
    }
    rows = []
    errors = []
    seen = set()
    for path in sorted(OUT.glob("alpha*/eps*/seed*/metrics.json")):
        try:
            data = json.loads(path.read_text())
            meta = json.loads((path.parent / "run_metadata.json").read_text())
            key = (float(meta["alpha"]), float(meta["epsilon_target"]), int(meta["seed"]))
            if key in seen:
                raise ValueError(f"duplicate run tuple {key}")
            seen.add(key)
            if key not in expected_keys:
                raise ValueError(f"unexpected run tuple {key}")
            if meta.get("grid_config_sha256") != config_sha:
                raise ValueError("run config fingerprint differs")
            if abs(finite(meta["epsilon_actual"], "epsilon_actual") - key[1]) > max(1e-6, key[1] * 1e-5):
                raise ValueError("requested and accounted epsilon differ")
            if meta["effective_privacy_config"].get("secure_aggregation") is not False:
                raise ValueError("metadata does not record SecAgg disabled")
            if set(data) != EXPECTED_TOP:
                raise ValueError(f"top-level metrics schema mismatch: {sorted(data)}")
            if data["full_model"] != data["dp_only_no_action"]:
                raise ValueError("full_model compatibility alias differs from DP-only metrics")
            if len(data["target_retrain_ensemble"]) != 4:
                raise ValueError("expected four additional target-retrain replicates")
            for method in METHODS:
                metric = data[method]
                if not isinstance(metric, dict) or metric.get("incompatible") is True:
                    raise ValueError(f"{method} does not contain evaluated metrics")
                for section, field in (("test", "accuracy"), ("forgotten_client", "accuracy"),
                                       ("retained_data", "accuracy"), ("forgotten_privacy", "auc"),
                                       ("retained_privacy", "auc")):
                    finite(metric[section][field], f"{method}.{section}.{field}")
                # Evaluator records share the required score fields above, while
                # optional cost details and method-specific diagnostics legitimately
                # differ (for example, FedEraser's replay accounting). Do not require
                # byte-for-byte equality of those optional fields across methods.
            if set(data["mub_summary"]) != set(METHODS) - {"target_retrain"}:
                raise ValueError("MUB summary is missing a measured candidate")

            row = {"alpha": key[0], "epsilon": key[1], "seed": key[2],
                   "epsilon_actual": finite(meta["epsilon_actual"], "epsilon_actual"),
                   "noise_multiplier": finite(meta["noise_multiplier"], "noise_multiplier")}
            for method in METHODS:
                metric = data[method]
                row[f"{method}_test_accuracy"] = finite(metric["test"]["accuracy"], method)
                row[f"{method}_forgotten_accuracy"] = finite(metric["forgotten_client"]["accuracy"], method)
                row[f"{method}_retained_accuracy"] = finite(metric["retained_data"]["accuracy"], method)
                row[f"{method}_forgotten_mia_auc"] = finite(metric["forgotten_privacy"]["auc"], method)
                row[f"{method}_forgotten_mia_advantage"] = finite(metric["forgotten_privacy"]["advantage"], method)
                row[f"{method}_retained_mia_auc"] = finite(metric["retained_privacy"]["auc"], method)
                row[f"{method}_retained_mia_advantage"] = finite(metric["retained_privacy"]["advantage"], method)
                row[f"{method}_unseen_forget_vs_unseen_auc"] = finite(metric["unseen_privacy"]["forget_vs_unseen_auc"], method)
                row[f"{method}_unseen_probe_accuracy"] = finite(metric["unseen_privacy"]["probe_accuracy"], method)
                if "mub" in metric:
                    row[f"{method}_mub_test_accuracy"] = finite(metric["mub"]["mub_test_accuracy"], method)
                    row[f"{method}_mub_mia_advantage"] = finite(metric["mub"]["mub_mia_advantage"], method)
            rows.append(row)
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})

    missing = sorted(expected_keys - seen)
    audit = {
        "expected_runs": len(expected_keys),
        "complete_evaluated_runs": len(rows),
        "unique_run_tuples": len(seen),
        "missing_runs": [{"alpha": a, "epsilon": e, "seed": s} for a, e, s in missing],
        "invalid_runs": errors,
        "all_runs_complete": len(rows) == len(expected_keys) and not missing and not errors,
        "cached_reconstruction_and_federaser_evaluated": not errors and len(rows) == len(expected_keys),
        "secure_aggregation": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grid_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if rows:
        fields = list(rows[0])
        with (OUT / "summary_per_seed.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["alpha"], r["epsilon"], r["seed"])))
        groups = defaultdict(list)
        for row in rows:
            groups[(row["alpha"], row["epsilon"])].append(row)
        aggregates = []
        for (alpha, epsilon), group in sorted(groups.items()):
            result = {"alpha": alpha, "epsilon": epsilon, "seed_count": len(group)}
            for method in METHODS:
                for field in FIELDS:
                    col = f"{method}_{field}"
                    values = [float(row[col]) for row in group if col in row]
                    if not values:
                        continue
                    mean = statistics.mean(values)
                    half = 2.776445 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) == 5 else 0.0
                    result[f"{col}_mean"] = mean
                    result[f"{col}_ci95_low"] = mean - half
                    result[f"{col}_ci95_high"] = mean + half
            aggregates.append(result)
        with (OUT / "summary_by_alpha_epsilon.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    print(json.dumps(audit, indent=2))
    if not audit["all_runs_complete"]:
        raise SystemExit("Grid is incomplete or contains invalid outputs; see grid_audit.json")


if __name__ == "__main__":
    main()
