"""Configuration loading with intentionally small, explicit dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a mapping.")
    required = {"experiment_name", "seed", "data", "federated", "privacy", "deletion", "unlearning", "output"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    privacy = config["privacy"]
    if privacy.get("enabled"):
        if "population_size" not in privacy:
            raise ValueError("Private experiments must explicitly set privacy.population_size before execution.")
        if int(privacy["population_size"]) != int(config["data"]["n_clients"]):
            raise ValueError("privacy.population_size must equal the original configured data.n_clients.")
    return config
