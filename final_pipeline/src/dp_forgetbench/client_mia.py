"""Client-level black-box membership inference utilities.

These utilities separate feature extraction from attack training. Callers must
provide shadow-model observations for attack fitting/calibration and a distinct
target model's client observations for evaluation. Example-level membership
results are not a substitute for this client-level protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import ClientDataset
from .federated import flatten_logits, get_device


def match_clients_by_label_profile(
    member_ids: Sequence[int],
    nonmember_ids: Sequence[int],
    clients: Mapping[int, ClientDataset],
    *,
    n_pairs: int,
    max_total_variation: float,
    n_classes: int,
) -> list[tuple[int, int, float]]:
    """Match member/nonmember clients on label histograms with an explicit caliper.

    Returns deterministic ``(member_id, nonmember_id, TV distance)`` pairs,
    ordered by distance and then client IDs. Each client is used at most once.
    The caller must record the retained-pair count because matching changes the
    target population to the common-support cohort.
    """
    members = sorted(map(int, member_ids))
    nonmembers = sorted(map(int, nonmember_ids))
    if n_pairs < 1 or n_pairs > min(len(members), len(nonmembers)):
        raise ValueError("n_pairs must be positive and no larger than either cohort")
    if not np.isfinite(max_total_variation) or not 0.0 <= max_total_variation <= 1.0:
        raise ValueError("max_total_variation must lie in [0, 1]")
    if n_classes < 2:
        raise ValueError("n_classes must be at least two")
    if set(members) & set(nonmembers):
        raise ValueError("Member and nonmember client IDs must be disjoint")
    missing = (set(members) | set(nonmembers)) - set(map(int, clients))
    if missing:
        raise ValueError(f"Missing client records for IDs: {sorted(missing)[:5]}")

    def profile(client_id: int) -> np.ndarray:
        labels = clients[client_id].y.detach().cpu().long().reshape(-1).numpy()
        if labels.size == 0 or np.any(labels < 0) or np.any(labels >= n_classes):
            raise ValueError(f"Invalid labels for client {client_id}")
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        return counts / counts.sum()

    member_profiles = np.stack([profile(client_id) for client_id in members])
    nonmember_profiles = np.stack([profile(client_id) for client_id in nonmembers])
    distances = 0.5 * np.abs(member_profiles[:, None, :] - nonmember_profiles[None, :, :]).sum(axis=2)
    row_indices, column_indices = linear_sum_assignment(distances)
    matched = sorted(
        (
            (members[int(row)], nonmembers[int(column)], float(distances[row, column]))
            for row, column in zip(row_indices, column_indices, strict=True)
        ),
        key=lambda item: (item[2], item[0], item[1]),
    )
    eligible = [pair for pair in matched if pair[2] <= max_total_variation + 1e-12]
    if len(eligible) < n_pairs:
        raise ValueError(
            f"Only {len(eligible)} pairs meet TV caliper {max_total_variation}; "
            f"{n_pairs} required"
        )
    return eligible[:n_pairs]


def split_shadow_and_target_nonmember_clients(
    auxiliary_client_ids: Sequence[int], *, n_target_clients: int, seed: int
) -> tuple[list[int], list[int]]:
    """Create disjoint shadow and target-nonmember client pools.

    The shadow pool is sized to provide ``n_target_clients`` members and an
    equal number of nonmembers for every shadow model. A separate, equally
    large pool is reserved for evaluating the target attack.
    """
    ids = [int(value) for value in auxiliary_client_ids]
    if n_target_clients < 1:
        raise ValueError("n_target_clients must be positive")
    if len(ids) != len(set(ids)):
        raise ValueError("auxiliary_client_ids must be unique")
    if len(ids) < 4 * n_target_clients:
        raise ValueError("Need at least 4*n_target_clients auxiliary clients")
    order = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
    split = 2 * n_target_clients
    shadow_pool = [int(value) for value in order[:split]]
    target_nonmembers = [int(value) for value in order[split:]]
    if set(shadow_pool) & set(target_nonmembers):
        raise RuntimeError("Shadow and target-nonmember client pools overlap")
    return shadow_pool, target_nonmembers


def split_shadow_fit_calibration_and_target_nonmembers(
    auxiliary_client_ids: Sequence[int], *, n_target_clients: int, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Reserve disjoint clients for shadow fitting, calibration, and target audit.

    Each shadow stage needs one target-sized member cohort and one equally sized
    nonmember cohort. The returned shadow-fit and shadow-calibration pools each
    contain ``2*n_target_clients`` clients; a further ``2*n_target_clients`` are
    held out for target evaluation. Clients therefore cannot occur on both sides
    of the shadow attack's fit/calibration split.
    """
    ids = [int(value) for value in auxiliary_client_ids]
    if n_target_clients < 1:
        raise ValueError("n_target_clients must be positive")
    if len(ids) != len(set(ids)):
        raise ValueError("auxiliary_client_ids must be unique")
    required = 6 * n_target_clients
    if len(ids) < required:
        raise ValueError(f"Need at least 6*n_target_clients auxiliary clients ({required})")
    order = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
    width = 2 * n_target_clients
    fit_pool = [int(value) for value in order[:width]]
    calibration_pool = [int(value) for value in order[width : 2 * width]]
    target_nonmembers = [int(value) for value in order[2 * width :]]
    pools = [set(fit_pool), set(calibration_pool), set(target_nonmembers)]
    if any(left & right for i, left in enumerate(pools) for right in pools[i + 1 :]):
        raise RuntimeError("Shadow fit, calibration, and target nonmember pools must be disjoint")
    return fit_pool, calibration_pool, target_nonmembers


def client_output_features(
    model: nn.Module,
    clients: Mapping[int, ClientDataset],
    *,
    examples_per_client: int = 128,
    seed: int = 0,
    class_labels: Sequence[int] | None = None,
    examples_per_class: int = 16,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one size-controlled, class-agnostic output vector per client.

    By default, exactly ``examples_per_client`` examples are sampled per client;
    clients smaller than that are excluded instead of exposing their size through
    the feature vector. Features contain macro-averages of loss, correctness,
    true-label confidence, entropy, margin, and per-example loss quantiles.

    If ``class_labels`` is supplied, the same number of examples is sampled from
    every listed class within each client, then class-level features are averaged
    equally. Clients lacking that common support are omitted. This optional mode
    controls class-mixture confounding, but can substantially restrict the target
    population under strongly non-IID partitions. Returned IDs identify retained
    clients; callers must align membership labels to these IDs explicitly.
    """
    if examples_per_client < 2 or examples_per_class < 2 or batch_size < 1:
        raise ValueError("Feature sample sizes must be >=2 and batch_size must be positive")
    labels = tuple(int(value) for value in class_labels) if class_labels is not None else None
    if labels is not None and (not labels or len(set(labels)) != len(labels)):
        raise ValueError("class_labels must be a non-empty sequence of unique labels")

    device = get_device()
    model = model.to(device)
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    client_ids: list[int] = []
    rows: list[np.ndarray] = []
    try:
        sample_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        segments: list[tuple[int, int, int]] = []
        offset = 0
        # Gather all fixed-size client/class samples first. Batched inference is
        # materially faster than launching one model forward for every tiny
        # class group, especially when auditing hundreds of clients.
        for client_id in sorted(clients):
            client = clients[client_id]
            y = client.y.detach().cpu().long().reshape(-1)
            x = client.x.detach().cpu()
            if len(x) != len(y):
                raise ValueError(f"Client {client_id} has mismatched x/y lengths")

            if labels is None:
                if len(y) < examples_per_client:
                    continue
                idx = rng.choice(len(y), size=examples_per_client, replace=False)
                class_groups = [np.asarray(idx, dtype=np.int64)]
            else:
                groups = []
                for label in labels:
                    available = torch.where(y == label)[0].numpy()
                    if len(available) < examples_per_class:
                        groups = []
                        break
                    groups.append(
                        rng.choice(available, size=examples_per_class, replace=False)
                    )
                if not groups:
                    continue
                class_groups = groups

            for indices in class_groups:
                indices_tensor = torch.as_tensor(indices, dtype=torch.long)
                group_x = x[indices_tensor]
                group_y = y[indices_tensor]
                sample_batches.append(group_x)
                label_batches.append(group_y)
                segments.append((int(client_id), offset, offset + len(group_y)))
                offset += len(group_y)

        if not segments:
            raise ValueError("No clients met the requested fixed-size/class-support feature protocol")
        all_x = torch.cat(sample_batches, dim=0)
        all_y = torch.cat(label_batches, dim=0)
        logits_parts = []
        with torch.no_grad():
            for start in range(0, len(all_y), batch_size):
                logits_parts.append(
                    flatten_logits(model, all_x[start : start + batch_size].to(device))
                    .detach()
                    .cpu()
                )
        all_logits = torch.cat(logits_parts)

        if all_logits.ndim == 1 or all_logits.shape[-1] == 1:
            binary_logits = all_logits.reshape(-1)
            positive_probability = torch.sigmoid(binary_logits)
            probs = torch.stack([1.0 - positive_probability, positive_probability], dim=1)
            losses = nn.functional.binary_cross_entropy_with_logits(
                binary_logits, all_y.float(), reduction="none"
            )
            pred = (binary_logits >= 0).long()
        else:
            probs = torch.softmax(all_logits, dim=1)
            losses = nn.functional.cross_entropy(all_logits, all_y, reduction="none")
            pred = all_logits.argmax(dim=1)
        true_confidence = probs.gather(1, all_y.reshape(-1, 1)).squeeze(1)
        ordered = probs.sort(dim=1, descending=True).values
        margins = ordered[:, 0] - ordered[:, 1]
        entropies = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        # Convert the per-example outputs once and perform group summaries in
        # NumPy. This avoids thousands of tiny tensor reductions/thread-pool
        # launches in the class-matched protocol.
        losses_np = losses.numpy()
        correct_np = (pred == all_y).numpy().astype(np.float64)
        confidence_np = true_confidence.numpy()
        entropy_np = entropies.numpy()
        margin_np = margins.numpy()
        segments_by_client: dict[int, list[tuple[int, int]]] = {}
        for client_id, start, stop in segments:
            segments_by_client.setdefault(client_id, []).append((start, stop))
        for client_id, client_segments in segments_by_client.items():
            class_features = []
            for start, stop in client_segments:
                group_losses = losses_np[start:stop]
                group_conf = confidence_np[start:stop]
                group_entropy = entropy_np[start:stop]
                group_margin = margin_np[start:stop]
                class_features.append(
                    np.asarray(
                        [
                            group_losses.mean(),
                            np.median(group_losses),
                            np.quantile(group_losses, 0.1),
                            np.quantile(group_losses, 0.9),
                            correct_np[start:stop].mean(),
                            group_conf.mean(),
                            group_conf.std(),
                            group_entropy.mean(),
                            group_entropy.std(),
                            group_margin.mean(),
                            group_margin.std(),
                        ],
                        dtype=np.float64,
                    )
                )
            rows.append(np.mean(class_features, axis=0))
            client_ids.append(int(client_id))
    finally:
        model.train(was_training)

    if not rows:
        raise ValueError("No clients met the requested fixed-size/class-support feature protocol")
    return np.stack(rows), np.asarray(client_ids, dtype=np.int64)


def _tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = tpr[fpr <= target_fpr]
    return float(valid.max()) if len(valid) else 0.0


def _stratified_client_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    n_bootstraps: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    by_label = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    collected: dict[str, list[float]] = {
        "auc": [],
        "advantage": [],
        "tpr_at_fpr_0_1pct": [],
        "tpr_at_fpr_1pct": [],
        "tpr_at_fpr_5pct": [],
    }
    if any(len(indices) == 0 for indices in by_label.values()):
        return {key: [float("nan"), float("nan")] for key in collected}
    for _ in range(n_bootstraps):
        sampled = np.concatenate(
            [rng.choice(by_label[label], size=len(by_label[label]), replace=True) for label in (0, 1)]
        )
        boot_y, boot_s = labels[sampled], scores[sampled]
        fpr, tpr, _ = roc_curve(boot_y, boot_s)
        collected["auc"].append(float(roc_auc_score(boot_y, boot_s)))
        collected["advantage"].append(float(np.max(tpr - fpr)))
        collected["tpr_at_fpr_0_1pct"].append(_tpr_at_fpr(boot_y, boot_s, 0.001))
        collected["tpr_at_fpr_1pct"].append(_tpr_at_fpr(boot_y, boot_s, 0.01))
        collected["tpr_at_fpr_5pct"].append(_tpr_at_fpr(boot_y, boot_s, 0.05))
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in collected.items()
    }


def evaluate_client_membership_attack(
    shadow_features: np.ndarray,
    shadow_membership: np.ndarray,
    shadow_model_ids: np.ndarray,
    target_features: np.ndarray,
    target_membership: np.ndarray,
    *,
    seed: int = 0,
    n_bootstraps: int = 1000,
    fit_shadow_model_ids: Sequence[int] | None = None,
    calibration_shadow_model_ids: Sequence[int] | None = None,
) -> dict:
    """Train/calibrate on shadow models, evaluate once on disjoint target clients.

    Entire shadow model IDs are partitioned between attack-model fitting and
    threshold calibration, preventing a shadow model's client rows from leaking
    across those stages. Target-model features and labels are only used for final
    evaluation. Bootstrap intervals resample clients within membership class.
    """
    sx = np.asarray(shadow_features, dtype=np.float64)
    sy = np.asarray(shadow_membership, dtype=np.int64).reshape(-1)
    groups = np.asarray(shadow_model_ids).reshape(-1)
    tx = np.asarray(target_features, dtype=np.float64)
    ty = np.asarray(target_membership, dtype=np.int64).reshape(-1)
    if sx.ndim != 2 or tx.ndim != 2 or sx.shape[1] != tx.shape[1]:
        raise ValueError("Shadow and target features must be 2-D with matching columns")
    if len(sx) != len(sy) or len(sx) != len(groups) or len(tx) != len(ty):
        raise ValueError("Features, labels, and shadow model IDs must have matching row counts")
    if set(np.unique(sy)) != {0, 1} or set(np.unique(ty)) != {0, 1}:
        raise ValueError("Both shadow and target memberships must contain labels 0 and 1")
    unique_models = np.unique(groups)
    if len(unique_models) < 4:
        raise ValueError("At least four independent shadow model IDs are required")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive")

    if (fit_shadow_model_ids is None) != (calibration_shadow_model_ids is None):
        raise ValueError("Provide both fit and calibration shadow-model IDs, or neither")
    if fit_shadow_model_ids is None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        fit_idx, calibration_idx = next(splitter.split(sx, sy, groups))
        split_method = "group_shuffle_split_by_shadow_model"
    else:
        fit_groups = set(map(int, fit_shadow_model_ids))
        calibration_groups = set(map(int, calibration_shadow_model_ids or ()))
        if not fit_groups or not calibration_groups or fit_groups & calibration_groups:
            raise ValueError("Fit and calibration shadow-model groups must be non-empty and disjoint")
        observed_groups = set(map(int, unique_models))
        if fit_groups | calibration_groups != observed_groups:
            raise ValueError("Fit/calibration groups must partition all observed shadow-model IDs")
        fit_idx = np.flatnonzero(np.isin(groups, list(fit_groups)))
        calibration_idx = np.flatnonzero(np.isin(groups, list(calibration_groups)))
        split_method = "explicit_disjoint_client_pools_by_shadow_model"
    if set(np.unique(sy[fit_idx])) != {0, 1} or set(np.unique(sy[calibration_idx])) != {0, 1}:
        raise ValueError("Both shadow fit and calibration partitions need member and nonmember clients")

    attack = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    attack.fit(sx[fit_idx], sy[fit_idx])
    calibration_scores = attack.predict_proba(sx[calibration_idx])[:, 1]
    cal_fpr, cal_tpr, cal_thresholds = roc_curve(sy[calibration_idx], calibration_scores)
    threshold_idx = int(np.argmax(cal_tpr - cal_fpr))
    threshold = float(cal_thresholds[threshold_idx])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(np.max(calibration_scores), np.inf))

    scores = attack.predict_proba(tx)[:, 1]
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(ty, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(ty, scores)
    bootstrap_ci = _stratified_client_bootstrap(
        ty, scores, n_bootstraps=n_bootstraps, seed=seed
    )
    member_prevalence = float(np.mean(ty == 1))
    tpr_at_0_1 = _tpr_at_fpr(ty, scores, 0.001)
    tpr_at_1 = _tpr_at_fpr(ty, scores, 0.01)
    tpr_at_5 = _tpr_at_fpr(ty, scores, 0.05)
    ppv_by_prior = {}
    for prior in (0.01, 0.05, 0.10):
        ppv_by_prior[f"{prior:.0%}"] = float(
            (tpr_at_1 * prior)
            / max(tpr_at_1 * prior + (1.0 - 0.01) * (1.0 - prior), 1e-12)
        )

    return {
        "membership_unit": "whole client",
        "attack": "shadow-calibrated standardized logistic regression",
        "shadow_fit_calibration_split": split_method,
        "target_auc": float(roc_auc_score(ty, scores)),
        "target_auc_ci95_client_bootstrap": bootstrap_ci["auc"],
        "target_average_precision": float(average_precision_score(ty, scores)),
        "target_balanced_accuracy": float(balanced_accuracy_score(ty, predictions)),
        "target_accuracy": float(accuracy_score(ty, predictions)),
        "target_precision": float(precision_score(ty, predictions, zero_division=0)),
        "target_recall": float(recall_score(ty, predictions, zero_division=0)),
        "target_f1": float(f1_score(ty, predictions, zero_division=0)),
        "target_advantage": float(np.max(tpr - fpr)),
        "target_advantage_ci95_client_bootstrap": bootstrap_ci["advantage"],
        "target_tpr_at_fpr_0_1pct": tpr_at_0_1,
        "target_tpr_at_fpr_0_1pct_ci95_client_bootstrap": bootstrap_ci["tpr_at_fpr_0_1pct"],
        "target_tpr_at_fpr_1pct": tpr_at_1,
        "target_tpr_at_fpr_1pct_ci95_client_bootstrap": bootstrap_ci["tpr_at_fpr_1pct"],
        "target_tpr_at_fpr_5pct": tpr_at_5,
        "target_tpr_at_fpr_5pct_ci95_client_bootstrap": bootstrap_ci["tpr_at_fpr_5pct"],
        "target_ppv_at_member_priors_using_1pct_fpr": ppv_by_prior,
        "target_members": int(tp + fn),
        "target_nonmembers": int(tn + fp),
        "target_member_prevalence": member_prevalence,
        "low_fpr_support": {
            f"{fpr:.1%}": {
                "target_nonmembers": int(tn + fp),
                "expected_false_positives_at_operating_point": float((tn + fp) * fpr),
                "at_least_5_expected_false_positives": bool((tn + fp) * fpr >= 5),
            }
            for fpr in (0.001, 0.01, 0.05)
        },
        "target_threshold": threshold,
        "threshold_selection": "maximum TPR-FPR on shadow models held out by model ID",
        "shadow_model_count": int(len(unique_models)),
        "shadow_fit_model_count": int(len(np.unique(groups[fit_idx]))),
        "shadow_calibration_model_count": int(len(np.unique(groups[calibration_idx]))),
        "shadow_fit_model_ids": np.unique(groups[fit_idx]).tolist(),
        "shadow_calibration_model_ids": np.unique(groups[calibration_idx]).tolist(),
        "shadow_fit_clients": int(len(fit_idx)),
        "shadow_calibration_clients": int(len(calibration_idx)),
        "uncertainty_unit": "client, stratified bootstrap within membership class",
        "notes": [
            "Target clients are evaluated once and are not used to fit or calibrate the attack.",
            "Use fixed example counts and, when feasible, shared class support to reduce size and class-mixture confounding.",
            "This function evaluates supplied shadow observations; it does not train shadow federations.",
            "Low-FPR estimates are limited by the number of target nonmember clients.",
        ],
    }
