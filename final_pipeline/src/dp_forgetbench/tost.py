"""Seed-level paired TOST helpers for planned equivalence analyses.

The independent units are paired federation seeds. ``tost_paired`` performs
the classic two one-sided t tests on those per-seed differences; it does not
resample images or clients as if they were independent seeds. A valid
equivalence claim still requires a prespecified margin, matched runs, and
enough independent seeds for useful power.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import integrate, stats
from typing import Sequence


# ---------------------------------------------------------------------------
# Core paired-seed TOST
# ---------------------------------------------------------------------------

def tost_paired(
    d: Sequence[float],
    delta: float,
    alpha: float = 0.05,
    preregistration_ref: str | None = None,
) -> dict:
    """Paired-seed TOST for a mean difference.

    Args:
        d: Per-seed paired differences, one value per independent federation
           seed. Orient the difference consistently for the endpoint.
        delta: Prespecified positive equivalence margin in the endpoint's units.
        alpha: One-sided significance level; alpha=0.05 corresponds to a 90%
               two-sided t interval for the equivalence decision.
        preregistration_ref: Stable reference to the frozen endpoint and margin
            specification. Without it the calculation is exploratory and cannot
            support an intersection-union ``REDUNDANT`` verdict.

    Returns:
        TOST p-values and paired-seed t intervals. Equivalence requires both
        one-sided tests to reject at ``alpha`` (equivalently, the 90% interval
        is strictly inside the prespecified margins when alpha=0.05).
    """
    arr = np.asarray(d, dtype=float)
    if arr.ndim != 1:
        raise ValueError("d must be a one-dimensional sequence of seed-level differences.")
    n = len(arr)
    if n < 2:
        raise ValueError("At least 2 independent seeds required for TOST.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("d must contain only finite seed-level differences.")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a finite, positive equivalence margin.")
    if not np.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5.")

    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    critical90 = float(stats.t.ppf(1 - alpha, df=n - 1))
    critical95 = float(stats.t.ppf(0.975, df=n - 1))
    ci90_lo, ci90_hi = mean - critical90 * se, mean + critical90 * se
    ci95_lo, ci95_hi = mean - critical95 * se, mean + critical95 * se

    # H01: mean <= -delta, H11: mean > -delta.
    # H02: mean >= +delta, H12: mean < +delta.
    if se == 0:
        p_lower = 0.0 if mean > -delta else (0.5 if mean == -delta else 1.0)
        p_upper = 0.0 if mean < delta else (0.5 if mean == delta else 1.0)
    else:
        lower_t = (mean + delta) / se
        upper_t = (mean - delta) / se
        p_lower = float(stats.t.sf(lower_t, df=n - 1))
        p_upper = float(stats.t.cdf(upper_t, df=n - 1))
    equivalent = p_lower < alpha and p_upper < alpha

    if equivalent:
        interp = f"EQUIVALENT: 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] inside (+/-{delta:.4f}). Effect is negligible."
    elif ci90_lo > 0:
        interp = f"BENEFICIAL: 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] entirely above 0. Unlearning improves over DP-only."
    elif ci90_hi < 0:
        interp = f"HARMFUL: 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] entirely below 0. Unlearning degrades relative to DP-only."
    else:
        interp = f"INCONCLUSIVE: 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] overlaps 0 and/or margin +/-{delta:.4f}. More seeds needed."

    return {
        "mean": mean,
        "sd": sd,
        "ci90_low": float(ci90_lo),
        "ci90_high": float(ci90_hi),
        "ci95_low": float(ci95_lo),
        "ci95_high": float(ci95_hi),
        "delta": float(delta),
        "alpha": float(alpha),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "method": "paired_seed_level_tost",
        "preregistration_ref": preregistration_ref,
        "equivalent": equivalent,
        "n_seeds": n,
        "interpretation": interp,
    }


def tost_nested(
    seed_example_diffs: list[list[float]],
    delta: float,
    n_outer: int = 5_000,
    n_inner: int = 5_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Exploratory nested bootstrap interval for per-example differences.

    This does not recompute AUC from client-clustered scores/labels and is not a
    confirmatory client-level TOST. Do not use it as an input to a redundancy
    decision; use seed-level per-client attack summaries and a validated
    client-bootstrap procedure instead.

    Args:
        seed_example_diffs: List of per-seed lists of per-example AUC scores.
        delta: Equivalence margin (typically 0.05 AUC).
        n_outer: Outer seed resamples.
        n_inner: Inner example resamples per seed.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_seeds = len(seed_example_diffs)
    if n_seeds < 2:
        raise ValueError("At least 2 seeds required for nested bootstrap.")

    outer_means = []
    for _ in range(n_outer):
        # Resample seeds
        sampled_seed_indices = rng.choice(n_seeds, size=n_seeds, replace=True)
        inner_seed_means = []
        for si in sampled_seed_indices:
            examples = np.asarray(seed_example_diffs[si])
            # Resample examples within that seed
            inner_mean = rng.choice(examples, size=len(examples), replace=True).mean()
            inner_seed_means.append(inner_mean)
        outer_means.append(float(np.mean(inner_seed_means)))

    outer_means_arr = np.asarray(outer_means)
    ci90_lo = float(np.percentile(outer_means_arr, alpha * 100))
    ci90_hi = float(np.percentile(outer_means_arr, (1 - alpha) * 100))
    ci95_lo = float(np.percentile(outer_means_arr, 2.5))
    ci95_hi = float(np.percentile(outer_means_arr, 97.5))

    screen_within_margin = (-delta < ci90_lo) and (ci90_hi < delta)
    grand_mean = float(outer_means_arr.mean())

    return {
        "grand_mean": grand_mean,
        "ci90_low": ci90_lo,
        "ci90_high": ci90_hi,
        "ci95_low": ci95_lo,
        "ci95_high": ci95_hi,
        "delta": delta,
        "equivalent": None,
        "exploratory_screen_within_margin": screen_within_margin,
        "method": "nested_percentile_bootstrap_exploratory",
        "preregistration_ref": None,
        "confirmatory_use": False,
        "n_seeds": n_seeds,
        "interpretation": (
            f"EXPLORATORY SCREEN WITHIN MARGIN: 90% interval [{ci90_lo:.4f}, {ci90_hi:.4f}]"
            if screen_within_margin
            else f"EXPLORATORY SCREEN OUTSIDE MARGIN: 90% interval [{ci90_lo:.4f}, {ci90_hi:.4f}]"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-endpoint intersection-union decision
# ---------------------------------------------------------------------------

def intersection_union_decision(
    e1_result: dict,  # Utility TOST
    e2_result: dict,  # JS alignment TOST
    e3_result: dict,  # Deleted-client attackability TOST
    e4_result: dict,  # Retained-client attackability TOST
) -> dict:
    """Intersection-union test over four preregistered paired-seed endpoints.

    REDUNDANT requires utility, functional alignment, deleted-client privacy,
    and retained-client privacy all to pass equivalence. Cost remains a
    separate reported trade-off and cannot replace a missing privacy endpoint.
    No multiplicity correction needed across endpoints for IUT (Berger & Hsu).
    Consumers should apply Holm or B-Y correction across cells/methods.

    Args:
        e1_result: TOST on per-seed ``utility(method) - utility(DP)``.
        e2_result: TOST on per-seed ``JS(DP, retrain) - JS(method, retrain)``.
        e3_result: TOST on per-seed oriented deleted-client attackability
            reduction, e.g.
            ``attackability(DP) - attackability(method)``. The attackability
            definition and any direction correction must be frozen first. The
            result must come from ``tost_paired`` and include its
            preregistration reference.
        e4_result: TOST on per-seed change in retained-client attackability,
            oriented as ``attackability(method) - attackability(DP)``.

    Returns:
        dict with:
          - verdict: REDUNDANT | UNLEARNING-BENEFICIAL | UNLEARNING-HARMFUL | INCONCLUSIVE
          - endpoint_results: dict of all four endpoint verdicts
          - rationale: human-readable explanation
    """
    def has_preregistered_paired_tost(result: dict) -> bool:
        return bool(
            result.get("method") == "paired_seed_level_tost"
            and isinstance(result.get("preregistration_ref"), str)
            and result.get("preregistration_ref").strip()
        )

    def is_equivalent(result: dict) -> bool:
        return has_preregistered_paired_tost(result) and bool(result.get("equivalent", False))

    e1_eq = is_equivalent(e1_result)
    e2_eq = is_equivalent(e2_result)
    e3_eq = is_equivalent(e3_result)
    e4_eq = is_equivalent(e4_result)

    # Positive endpoint differences are oriented to mean benefit for E2/E3.
    e2_beneficial = (
        has_preregistered_paired_tost(e2_result)
        and e2_result.get("ci90_low", 0.0) > e2_result.get("delta", float("inf"))
    )
    e3_beneficial = (
        has_preregistered_paired_tost(e3_result)
        and e3_result.get("ci90_low", 0.0) > e3_result.get("delta", float("inf"))
    )

    # Require no material utility loss and retained-privacy non-inferiority for
    # the beneficial verdict. E4 is attackability(method) - attackability(DP).
    e1_non_harmful = (
        has_preregistered_paired_tost(e1_result)
        and e1_result.get("ci90_low", -float("inf")) > -e1_result.get("delta", 0.0)
    )
    e4_non_harmful = (
        has_preregistered_paired_tost(e4_result)
        and e4_result.get("ci90_high", float("inf")) < e4_result.get("delta", 0.0)
    )

    # Material harms use the frozen margins, not merely a CI excluding zero.
    e1_harmful = (
        has_preregistered_paired_tost(e1_result)
        and e1_result.get("ci90_high", 0.0) < -e1_result.get("delta", float("inf"))
    )
    e4_harmful = (
        has_preregistered_paired_tost(e4_result)
        and e4_result.get("ci90_low", 0.0) > e4_result.get("delta", float("inf"))
    )

    if e1_eq and e2_eq and e3_eq and e4_eq:
        verdict = "REDUNDANT"
        rationale = "All four endpoints (utility, functional alignment, deleted-client attackability, retained-client attackability) are equivalent within their frozen margins."
    elif e2_beneficial and e3_beneficial and e1_non_harmful and e4_non_harmful:
        verdict = "UNLEARNING-BENEFICIAL"
        rationale = "Unlearning improves functional alignment and deleted-client privacy while preserving retained-client privacy and without significant utility harm."
    elif e1_harmful or e4_harmful:
        verdict = "UNLEARNING-HARMFUL"
        rationale = "The method causes utility loss or retained-client attackability increase beyond its frozen practical margin."
    else:
        verdict = "INCONCLUSIVE"
        failing = [k for k, v in [("E1-utility", e1_eq), ("E2-JS", e2_eq), ("E3-deleted-privacy", e3_eq), ("E4-retained-privacy", e4_eq)] if not v]
        rationale = f"Cannot declare equivalence. Failing endpoints: {', '.join(failing)}. Consider more seeds or tighter delta."

    return {
        "verdict": verdict,
        "rationale": rationale,
        "endpoint_results": {
            "e1_utility": {
                "equivalent": e1_eq,
                "delta": e1_result.get("delta"),
                "ci90": [e1_result.get("ci90_low"), e1_result.get("ci90_high")],
                "mean_diff": e1_result.get("mean"),
                "preregistration_ref": e1_result.get("preregistration_ref"),
            },
            "e2_js_alignment": {
                "equivalent": e2_eq,
                "delta": e2_result.get("delta"),
                "ci90": [e2_result.get("ci90_low"), e2_result.get("ci90_high")],
                "mean_mub_js": e2_result.get("mean"),
                "preregistration_ref": e2_result.get("preregistration_ref"),
            },
            "e3_privacy_auc": {
                "equivalent": e3_eq,
                "delta": e3_result.get("delta"),
                "ci90": [e3_result.get("ci90_low"), e3_result.get("ci90_high")],
                "mean_auc_excess": e3_result.get("mean"),
                "preregistration_ref": e3_result.get("preregistration_ref"),
            },
            "e4_retained_client_attackability": {
                "equivalent": e4_eq,
                "delta": e4_result.get("delta"),
                "ci90": [e4_result.get("ci90_low"), e4_result.get("ci90_high")],
                "mean_attackability_change": e4_result.get("mean"),
                "preregistration_ref": e4_result.get("preregistration_ref"),
            },
        },
    }


# ---------------------------------------------------------------------------
# Power estimation helper
# ---------------------------------------------------------------------------

def estimate_seeds_needed(
    effect_size: float,
    delta: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int | float:
    """Approximate seed count for TOST to achieve target power.

    Based on the paired t-test approximation: n ≈ (z_α + z_β)² * σ² / (δ - |μ|)².
    Safe for planning only — use simulation for final power analysis.
    """
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    margin = delta - abs(effect_size)
    if margin <= 0:
        return float("inf")
    n = ((z_alpha + z_beta) ** 2 * sigma ** 2) / (margin ** 2)
    return max(2, int(np.ceil(n)))


def tost_power_exact(
    n_seeds: int,
    effect_size: float,
    delta: float,
    sigma: float,
    alpha: float = 0.05,
) -> float:
    """Exact paired-seed TOST power under normally distributed differences.

    The two one-sided tests share the same sample mean and standard error, so
    their powers must not be multiplied as if they were independent. This
    integrates over the chi-square sampling distribution of the sample
    variance and the conditional normal distribution of the sample mean.
    ``sigma`` and ``effect_size`` describe the population of paired-seed
    differences; the independent unit is still a federation seed.
    """
    if isinstance(n_seeds, bool) or int(n_seeds) != n_seeds or n_seeds < 2:
        raise ValueError("n_seeds must be an integer of at least 2.")
    if not math.isfinite(effect_size):
        raise ValueError("effect_size must be finite.")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive.")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    if not math.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5.")

    df = int(n_seeds) - 1
    critical = float(stats.t.ppf(1.0 - alpha, df=df))
    mean_sd = sigma / math.sqrt(n_seeds)

    def conditional_power(chi_square: float) -> float:
        sample_sd = sigma * math.sqrt(chi_square / df)
        standard_error = sample_sd / math.sqrt(n_seeds)
        lower = -delta + critical * standard_error
        upper = delta - critical * standard_error
        if lower >= upper:
            return 0.0
        probability = stats.norm.cdf((upper - effect_size) / mean_sd) - stats.norm.cdf(
            (lower - effect_size) / mean_sd
        )
        return float(probability * stats.chi2.pdf(chi_square, df=df))

    power, _ = integrate.quad(
        conditional_power, 0.0, math.inf, epsabs=1e-10, epsrel=1e-9, limit=200
    )
    return float(min(1.0, max(0.0, power)))


def estimate_seeds_needed_exact(
    effect_size: float,
    delta: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
    max_seeds: int = 500,
) -> int | float:
    """Find the first seed count meeting exact TOST power under normality.

    Returns ``math.inf`` when the assumed true effect is on or outside the
    equivalence boundary, or when the requested power is not reached by
    ``max_seeds``. This is a planning calculation, not a guarantee that the
    normal-difference model or variance estimate is correct.
    """
    if not math.isfinite(power) or not 0 < power < 1:
        raise ValueError("power must be between 0 and 1.")
    if isinstance(max_seeds, bool) or int(max_seeds) != max_seeds or max_seeds < 2:
        raise ValueError("max_seeds must be an integer of at least 2.")
    if not math.isfinite(effect_size):
        raise ValueError("effect_size must be finite.")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive.")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    if not math.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5.")
    # At or beyond a practical-equivalence boundary, increasing n cannot make
    # a valid TOST conclusion likely under the stated effect model.
    if abs(effect_size) >= delta:
        return math.inf
    for n in range(2, int(max_seeds) + 1):
        if tost_power_exact(n, effect_size, delta, sigma, alpha) >= power:
            return n
    return math.inf
