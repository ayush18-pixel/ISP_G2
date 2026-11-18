"""Format actual FU metrics while retaining the prior study's MIA key name.

The SecAgg reference stores statuses at the two FU keys, not metric objects.
This formatter therefore preserves the full non-SecAgg metrics and only maps
the loss-MIA summary key to the reference's ``mia_lira`` spelling.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dinov2_single_deletion_nonsecagg_full_metrics"


def rename_mia(value):
    if isinstance(value, list):
        return [rename_mia(item) for item in value]
    if isinstance(value, dict):
        mapped = {key: rename_mia(item) for key, item in value.items()}
        if "loss_mia" in mapped:
            if "mia_lira" in mapped:
                raise ValueError("both loss_mia and mia_lira already exist")
            loss = mapped.pop("loss_mia")
            mapped["mia_lira"] = {
                "loss_mia_advantage": loss["advantage"],
                "loss_mia_auc": loss["auc"],
                "loss_mia_tpr_0_1pct_fpr": loss["tpr_at_fpr_0_1pct"],
                "loss_mia_tpr_1pct_fpr": loss["tpr_at_fpr_1pct"],
            }
        return mapped
    return value


def main():
    paths = sorted(OUT.glob("alpha*/eps*/seed*/metrics.json"))
    converted = 0
    for path in paths:
        original = json.loads(path.read_text())
        sidecar = path.with_name("metrics_extended.json")
        if not sidecar.exists():
            sidecar.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n")
        path.write_text(json.dumps(rename_mia(original), indent=2, sort_keys=True) + "\n")
        converted += 1
    print(json.dumps({"runs_converted": converted, "all_actual_metrics_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
