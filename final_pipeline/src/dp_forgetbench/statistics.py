"""Pre-registered-style summary utilities for repeated benchmark runs."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(values: list[float], seed: int = 0, draws: int = 10_000) -> dict[str, float]:
    """Non-parametric percentile CI over independent training seeds."""
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        raise ValueError("At least two independent seeds are required for a confidence interval.")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n_seeds": int(len(array)),
    }


def redundancy_screen(
    dp_only_distances: list[float],
    unlearning_distances: list[float],
    retrain_variability: list[float],
) -> dict:
    """A conservative screen, not a substitute for the final TOST analysis.

    Distance is JS divergence to an independently trained retraining target, so
    lower is better. The practical tolerance is the 90th percentile of observed
    retrain-to-retrain distance. We call this a *screen* because formal endpoint
    tolerances and all attack/utility endpoints must be frozen before Phase 2.
    """
    if not (len(dp_only_distances) == len(unlearning_distances) == len(retrain_variability)):
        raise ValueError("Paired seed lists must have equal length.")
    improvements = (np.asarray(dp_only_distances) - np.asarray(unlearning_distances)).tolist()
    threshold = float(np.quantile(np.asarray(retrain_variability), 0.90))
    result = {
        "metric": "test_js_divergence_to_retrain (lower is better)",
        "practical_tolerance_from_retrain_variability_q90": threshold,
        "marginal_unlearning_benefit": bootstrap_mean_ci(improvements),
        "dp_only_distance": bootstrap_mean_ci(dp_only_distances),
        "unlearning_distance": bootstrap_mean_ci(unlearning_distances),
        "retrain_variability": bootstrap_mean_ci(retrain_variability),
        "decision": "insufficient_seeds_for_confirmatory_equivalence" if len(improvements) < 5 else "exploratory_screen_only",
        "warning": (
            "This is not a claim that DP is equivalent to unlearning. Final redundancy decisions require "
            "pre-registered TOST/non-inferiority bounds across utility, forgetting, retained privacy, and cost."
        ),
    }
    if len(improvements) >= 5:
        upper = result["marginal_unlearning_benefit"]["ci95_high"]
        result["screen_interpretation"] = (
            "no practically material JS improvement detected by this screen"
            if upper <= threshold
            else "possible material JS improvement; confirm with the frozen multi-endpoint protocol"
        )
    return result

