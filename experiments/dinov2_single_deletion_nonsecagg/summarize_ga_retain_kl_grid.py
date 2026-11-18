"""Audit the full GA+retained-KL grid and write per-seed/grouped accuracy tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHOD = "gradient_ascent_retain_kl"
SUBSETS = ("test", "forgotten_client", "retained_data")
REFERENCES = (
    "dp_only_no_action", "retained_finetune_baseline",
    "cached_update_reconstruction_baseline", "federated_eraser_baseline", "target_retrain",
)


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {name}: {number}")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "ga_retain_kl_config_l10_s20.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).resolve().read_text())
    out = ROOT / cfg["output_root"]
    expected = {
        (float(a), float(e), int(s))
        for a in cfg["alpha_values"] for e in cfg["epsilon_values"] for s in cfg["seeds"]
    }
    seen, rows, errors = set(), [], []
    for path in sorted(out.glob("alpha*/eps*/seed*/metrics.json")):
        try:
            data = json.loads(path.read_text())
            meta = json.loads((path.parent / "run_metadata.json").read_text())
            key = (float(meta["alpha"]), float(meta["epsilon_target"]), int(meta["seed"]))
            if key in seen or key not in expected:
                raise ValueError(f"Duplicate or unexpected tuple {key}")
            seen.add(key)
            row = {"alpha": key[0], "epsilon": key[1], "seed": key[2]}
            for method in ("dp_only_replay", METHOD):
                for subset in SUBSETS:
                    row[f"{method}_{subset}_accuracy"] = _finite(data[method][subset]["accuracy"], f"{method}.{subset}")
            refs = data.get("existing_grid_reference") or {}
            for method in REFERENCES:
                if method in refs:
                    for subset in SUBSETS:
                        row[f"reference_{method}_{subset}_accuracy"] = _finite(refs[method][subset]["accuracy"], f"{method}.{subset}")
            rows.append(row)
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
    missing = sorted(expected - seen)
    audit = {
        "expected_cells": len(expected),
        "completed_cells": len(rows),
        "missing_cells": [{"alpha": a, "epsilon": e, "seed": s} for a, e, s in missing],
        "invalid_cells": errors,
        "complete": len(rows) == len(expected) and not missing and not errors,
        "warning": "DP replays use fresh OS-seeded sampling/noise because old checkpoints were not persisted. The old grid is a descriptive reference, not a paired comparison. Accuracy alone does not establish successful unlearning or privacy.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if rows:
        fields = list(rows[0])
        with (out / "summary_per_seed.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["alpha"], r["epsilon"], r["seed"])))
        groups = defaultdict(list)
        for row in rows:
            groups[(row["alpha"], row["epsilon"])].append(row)
        summaries = []
        metric_columns = [field for field in fields if field.endswith("_accuracy")]
        for (alpha, epsilon), group in sorted(groups.items()):
            item = {"alpha": alpha, "epsilon": epsilon, "seed_count": len(group)}
            for column in metric_columns:
                values = [float(row[column]) for row in group if column in row]
                if not values:
                    continue
                mean = statistics.mean(values)
                half = 2.776445 * statistics.stdev(values) / math.sqrt(5) if len(values) == 5 else float("nan")
                item[f"{column}_mean"] = mean
                item[f"{column}_ci95_low"] = mean - half
                item[f"{column}_ci95_high"] = mean + half
            for subset in SUBSETS:
                delta = [
                    row[f"{METHOD}_{subset}_accuracy"] - row[f"dp_only_replay_{subset}_accuracy"]
                    for row in group
                ]
                mean_delta = statistics.mean(delta)
                half_delta = 2.776445 * statistics.stdev(delta) / math.sqrt(5) if len(delta) == 5 else float("nan")
                item[f"paired_delta_{subset}_accuracy_mean"] = mean_delta
                item[f"paired_delta_{subset}_accuracy_ci95_low"] = mean_delta - half_delta
                item[f"paired_delta_{subset}_accuracy_ci95_high"] = mean_delta + half_delta
            summaries.append(item)
        with (out / "summary_by_alpha_epsilon.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    print(json.dumps(audit, indent=2))
    if not audit["complete"]:
        raise SystemExit("Grid incomplete or contains invalid cells")


if __name__ == "__main__":
    main()
