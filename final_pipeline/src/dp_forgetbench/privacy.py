"""Client-level central-DP primitives and an auditable accountant ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import dp_accounting
from dp_accounting import rdp


def _positive_integer(value: Any, field_name: str) -> int:
    """Validate an integer-valued count without silently truncating it."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be a positive integer.") from error
    if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(numeric)


@dataclass(frozen=True)
class PrivacyLedger:
    enabled: bool
    adjacency: str
    accountant: str
    sampling: str
    client_sample_rate: float
    public_population_size: int | None
    aggregation_normalizer: float | None
    mechanism_version: str
    rounds: int
    release_count: int
    clip_norm: float | None
    noise_multiplier: float | None
    epsilon: float | None
    delta: float | None
    sampling_rng: str
    noise_rng: str
    floating_point_noise_mitigation: bool
    threat_model_note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def poisson_gaussian_epsilon(
    *, sample_rate: float, noise_multiplier: float, rounds: int, delta: float
) -> float:
    """Compute RDP epsilon for the simulated Poisson-sampled Gaussian mechanism.

    This applies to add/remove-one client adjacency. Callers must not reuse it for
    fixed-size sampling, per-example clipping, or independently trained model
    releases. The accountant composes every training round, bounding the sequence
    of noised aggregate outputs and model checkpoints derived from that run. This
    assumes sampled-client identities and plaintext individual updates are not
    released.
    """
    if not math.isfinite(sample_rate) or not (0.0 < sample_rate <= 1.0):
        raise ValueError("sample_rate must be in (0, 1].")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be finite and positive.")
    rounds = _positive_integer(rounds, "rounds")
    if not math.isfinite(delta) or not (0.0 < delta < 1.0):
        raise ValueError("Invalid Gaussian-DP accounting parameters.")
    # Integer and selected half-integer orders avoid numerical convergence warnings
    # from the default dense fractional grid at high sampling probabilities.
    orders = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 48, 64, 96, 128, 256]
    accountant = rdp.RdpAccountant(orders=orders)
    event = dp_accounting.PoissonSampledDpEvent(
        sample_rate, dp_accounting.GaussianDpEvent(noise_multiplier)
    )
    accountant.compose(dp_accounting.SelfComposedDpEvent(event, rounds))
    epsilon, _ = accountant.get_epsilon_and_optimal_order(delta)
    epsilon = float(epsilon)
    # For valid finite inputs and a nontrivial sampled Gaussian mechanism,
    # epsilon is strictly positive. dp-accounting can round tiny RDP values
    # slightly negative and report zero for extreme finite noise; fail closed
    # rather than serialize a false perfect-privacy claim.
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ArithmeticError(
            "The RDP accountant returned a non-positive or non-finite epsilon; "
            "the requested parameters are outside its reliable numeric range."
        )
    return epsilon


def make_ledger(privacy_config: dict, federated_config: dict) -> PrivacyLedger:
    enabled = bool(privacy_config["enabled"])
    sample_rate = float(federated_config["client_sample_rate"])
    raw_rounds = federated_config["rounds"]
    if not math.isfinite(sample_rate) or not (0.0 < sample_rate <= 1.0):
        raise ValueError("federated.client_sample_rate must be finite and in (0, 1].")
    rounds = _positive_integer(raw_rounds, "federated.rounds")
    # `release_count` counts independently trained model trajectories, not
    # intermediate checkpoints from one trajectory; the accountant below
    # composes all of that trajectory's rounds.
    release_count = _positive_integer(privacy_config.get("release_count", 1), "privacy.release_count")
    if release_count != 1:
        raise ValueError(
            "This ledger supports exactly one independently trained model trajectory. "
            "Add explicit release composition before setting release_count > 1."
        )
    if not enabled:
        return PrivacyLedger(
            enabled=False,
            adjacency="none (non-private control)",
            accountant="none",
            sampling="not applicable",
            client_sample_rate=sample_rate,
            public_population_size=None,
            aggregation_normalizer=None,
            mechanism_version="non_private_control",
            rounds=rounds,
            release_count=0,
            clip_norm=None,
            noise_multiplier=None,
            epsilon=None,
            delta=None,
            sampling_rng="not applicable",
            noise_rng="not applicable",
            floating_point_noise_mitigation=False,
            threat_model_note="Non-private control; no DP claim is made.",
        )
    if privacy_config.get("sampling") != "poisson":
        raise ValueError("The Phase 0 accountant supports only Poisson client sampling.")
    public_population_size = _positive_integer(
        privacy_config["population_size"], "privacy.population_size"
    )
    clip_norm = float(privacy_config["clip_norm"])
    noise_multiplier = float(privacy_config["noise_multiplier"])
    delta = float(privacy_config["delta"])
    if not math.isfinite(clip_norm) or clip_norm <= 0.0:
        raise ValueError("privacy.clip_norm must be finite and positive.")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("privacy.noise_multiplier must be finite and positive.")
    if not math.isfinite(delta) or not (0.0 < delta < 1.0):
        raise ValueError("privacy.delta must be finite and in (0, 1).")
    aggregation_normalizer = public_population_size * sample_rate
    epsilon = poisson_gaussian_epsilon(
        sample_rate=sample_rate,
        noise_multiplier=noise_multiplier,
        rounds=rounds,
        delta=delta,
    )
    secure_rng = bool(privacy_config.get("secure_rng", False))
    noise_rng = (
        "torchcsprng random-device generator (/dev/urandom)"
        if secure_rng
        else "Torch PRNG seeded from OS entropy (not a CSPRNG)"
    )
    sampling_rng = "Python secrets.SystemRandom (OS-backed CSPRNG)"
    mitigation = secure_rng
    randomness_note = (
        "Noise uses torchcsprng OS random-device CSPRNG and the four-draw floating-point "
        "mitigation. "
        if secure_rng
        else "Noise uses an OS-entropy-seeded Torch PRNG, not a CSPRNG; this is exploratory "
        "and is not secure-RNG-audited evidence. "
    )
    return PrivacyLedger(
        enabled=True,
        adjacency="add/remove one complete client dataset",
        accountant="dp-accounting RDP accountant; Poisson-sampled Gaussian mechanism",
        sampling="independent Poisson client participation",
        client_sample_rate=sample_rate,
        public_population_size=public_population_size,
        aggregation_normalizer=aggregation_normalizer,
        mechanism_version="poisson_sum_fixed_public_normalizer_v2",
        rounds=rounds,
        release_count=release_count,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        epsilon=epsilon,
        delta=delta,
        sampling_rng=sampling_rng,
        noise_rng=noise_rng,
        floating_point_noise_mitigation=mitigation,
        threat_model_note=(
            "Simulator assumes a server-side Gaussian mechanism and protects a final model release. "
            "Each round sums clipped client updates, adds Gaussian noise to that sum, and divides by the fixed "
            "public normalizer q*N; this matches the Poisson-sampled Gaussian accountant under add/remove-one "
            "client adjacency. Client sampling uses an OS-backed CSPRNG. "
            f"{randomness_note} "
            "No mechanism randomness seed is serialized. "
            "It is not a production secure-aggregation implementation. The reported epsilon composes all rounds "
            "of this training run and bounds model checkpoints derived from its noised aggregates, provided "
            "sampled-client identities and plaintext individual updates are not released. Independently trained "
            "models or raw-data unlearning computations need separate privacy analysis."
        ),
    )
