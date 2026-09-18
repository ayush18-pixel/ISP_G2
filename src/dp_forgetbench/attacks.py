"""Attack diagnostics with fixed, held-out populations.

These diagnostics are intentionally labelled probes. They are valuable Phase 0
sanity checks, but do not substitute for a full LiRA/A-LiRA reproduction on
public datasets.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from torch import nn

from .federated import flatten_logits, get_device


def per_sample_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """Per-example loss for either a binary (single-logit) or multi-class model."""
    device = get_device()
    model = model.to(device)
    # ``x`` stays on CPU; ``flatten_logits`` streams it to the device in batches.
    logits = flatten_logits(model, x)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), y.cpu().float(), reduction="none")
    else:
        loss = torch.nn.functional.cross_entropy(logits, y.cpu().long(), reduction="none")
    return loss.numpy()


def _confidence(logits: torch.Tensor) -> torch.Tensor:
    """Scalar confidence signal for attack features: P(true-ish class), model-family agnostic."""
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.softmax(logits, dim=-1).max(dim=-1).values


def _balanced_binary_groups(
    member_x: torch.Tensor,
    member_y: torch.Tensor,
    unseen_x: torch.Tensor,
    unseen_y: torch.Tensor,
    limit_per_class: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match membership populations by label before calculating attack scores.

    Iterates over every label actually present in either population, so this
    works for binary (0/1) tasks and multi-class (e.g. 10-way CIFAR-10) tasks
    alike -- a fixed ``(0.0, 1.0)`` label set would silently drop all non-binary
    data instead of raising, which is why this is derived from the data.
    """
    labels_present = torch.unique(torch.cat([member_y, unseen_y])).tolist()
    member_indices, unseen_indices = [], []
    for label in labels_present:
        m = torch.where(member_y == label)[0]
        u = torch.where(unseen_y == label)[0]
        take = min(len(m), len(u), limit_per_class)
        if take == 0:
            continue
        member_indices.append(m[:take])
        unseen_indices.append(u[:take])
    if not member_indices:
        raise ValueError("Attack populations do not have overlapping label support.")
    member_idx = torch.cat(member_indices)
    unseen_idx = torch.cat(unseen_indices)
    return member_x[member_idx], member_y[member_idx], unseen_x[unseen_idx], unseen_y[unseen_idx]


def _bootstrap_roc_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = 100,
    seed: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    auc_list, adv_list, tpr1_list, tpr01_list = [], [], [], []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = labels[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        boot_scores = scores[idx]
        try:
            boot_auc = float(roc_auc_score(boot_labels, boot_scores))
            fpr, tpr, _ = roc_curve(boot_labels, boot_scores)
            adv = float(np.max(tpr - fpr))
            def _tpr_at(target_fpr: float) -> float:
                valid = tpr[fpr <= target_fpr]
                return float(valid.max()) if len(valid) else 0.0
            auc_list.append(boot_auc)
            adv_list.append(adv)
            tpr1_list.append(_tpr_at(0.01))
            tpr01_list.append(_tpr_at(0.001))
        except Exception:
            continue
    def _ci(vals: list[float]) -> list[float]:
        if not vals:
            return [0.0, 0.0]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return {
        "auc_ci_95": _ci(auc_list),
        "advantage_ci_95": _ci(adv_list),
        "tpr_at_fpr_1pct_ci_95": _ci(tpr1_list),
        "tpr_at_fpr_0_1pct_ci_95": _ci(tpr01_list),
    }


def loss_membership_probe(
    model: nn.Module,
    member_x: torch.Tensor,
    member_y: torch.Tensor,
    unseen_x: torch.Tensor,
    unseen_y: torch.Tensor,
    n_bootstraps: int = 100,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    """Black-box loss-threshold MIA with label-matched member/non-member groups and bootstrap CIs."""
    mx, my, ux, uy = _balanced_binary_groups(member_x, member_y, unseen_x, unseen_y)
    # Lower loss means stronger membership evidence.
    scores = np.concatenate([-per_sample_loss(model, mx, my), -per_sample_loss(model, ux, uy)])
    labels = np.concatenate([np.ones(len(mx), dtype=int), np.zeros(len(ux), dtype=int)])
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    advantage = float(np.max(tpr - fpr))
    def tpr_at(target_fpr: float) -> float:
        valid = tpr[fpr <= target_fpr]
        return float(valid.max()) if len(valid) else 0.0
    tpr1 = tpr_at(0.01)
    tpr01 = tpr_at(0.001)
    ci_dict = _bootstrap_roc_metrics(labels, scores, n_bootstraps=n_bootstraps, seed=seed)
    return {
        "auc": auc,
        "auc_ci_95": ci_dict["auc_ci_95"],
        "advantage": advantage,
        "advantage_ci_95": ci_dict["advantage_ci_95"],
        "tpr_at_fpr_1pct": tpr1,
        "tpr_at_fpr_1pct_ci_95": ci_dict["tpr_at_fpr_1pct_ci_95"],
        "tpr_at_fpr_0_1pct": tpr01,
        "tpr_at_fpr_0_1pct_ci_95": ci_dict["tpr_at_fpr_0_1pct_ci_95"],
        "members": int(len(mx)),
        "nonmembers": int(len(ux)),
    }


def _features(pre_model: nn.Module, post_model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    pre_loss = per_sample_loss(pre_model, x, y)
    post_loss = per_sample_loss(post_model, x, y)
    pre_confidence = _confidence(flatten_logits(pre_model, x)).numpy()
    post_confidence = _confidence(flatten_logits(post_model, x)).numpy()
    return np.stack([pre_loss, post_loss, pre_confidence, post_confidence], axis=1)


def tri_population_probe(
    pre_model: nn.Module,
    post_model: nn.Module,
    populations: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> dict:
    """Held-out three-class probe for forget, retain, and unseen populations.

    It uses a disjoint stratified train/evaluation split and exposes no raw
    examples. This is a baseline probe, not a claim of reproducing TC-UMIA.
    """
    names = ("forget", "retain", "unseen")
    if set(populations) != set(names):
        raise ValueError(f"Expected exactly {names} populations.")
    limit = min(len(populations[name][1]) for name in names)
    if limit < 12:
        raise ValueError("Need at least 12 examples in each tri-class population.")
    features, labels = [], []
    for class_id, name in enumerate(names):
        x, y = populations[name]
        features.append(_features(pre_model, post_model, x[:limit], y[:limit]))
        labels.append(np.full(limit, class_id, dtype=int))
    x_all = np.concatenate(features)
    y_all = np.concatenate(labels)
    x_train, x_eval, y_train, y_eval = train_test_split(x_all, y_all, test_size=0.5, random_state=seed, stratify=y_all)
    probe = LogisticRegression(max_iter=500, random_state=seed)
    probe.fit(x_train, y_train)
    predicted = probe.predict(x_eval)
    probabilities = probe.predict_proba(x_eval)
    return {
        "probe_accuracy": float((predicted == y_eval).mean()),
        "macro_ovr_auc": float(roc_auc_score(y_eval, probabilities, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(y_eval, predicted, labels=[0, 1, 2]).tolist(),
        "examples_per_population": int(limit),
        "warning": "Baseline pre/post loss-confidence probe, not a reproduction of TC-UMIA.",
    }
