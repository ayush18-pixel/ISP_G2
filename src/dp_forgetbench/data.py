"""Deterministic synthetic federation and versioned deletion manifests.

The synthetic source is deliberately used only for Phase 0. It makes every
privacy/deletion invariant testable before adapters for public datasets are added.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ClientDataset:
    client_id: int
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class Federation:
    clients: dict[int, ClientDataset]
    test_x: torch.Tensor
    test_y: torch.Tensor
    audit_x: torch.Tensor
    audit_y: torch.Tensor
    partition_manifest: dict | None = None
    validation_x: torch.Tensor | None = None
    validation_y: torch.Tensor | None = None
    auxiliary_clients: dict[int, ClientDataset] | None = None


@dataclass(frozen=True)
class DeletionResult:
    retained: dict[int, ClientDataset]
    forgotten_x: torch.Tensor
    forgotten_y: torch.Tensor
    rows: list[dict[str, int | float | str]]


def _allocate_auxiliary_clients(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    class_pools: dict[int, list[int]],
    *,
    n_clients: int,
    samples_per_client: int,
    partition_type: str,
    dirichlet_alpha: float,
    rng: np.random.Generator,
    first_client_id: int,
) -> tuple[dict[int, ClientDataset], dict[str, list[int]]]:
    """Allocate disjoint auxiliary clients from rows left after target partitioning."""
    if n_clients <= 0:
        return {}, {}
    if samples_per_client <= 0:
        raise ValueError("data.auxiliary_samples_per_client must be positive when auxiliary clients are requested")
    total = n_clients * samples_per_client
    capacities = np.asarray([len(class_pools[label]) for label in sorted(class_pools)], dtype=int)
    if int(capacities.sum()) < total:
        raise ValueError(
            f"Requested {total} auxiliary-client rows but only {int(capacities.sum())} remain after target/audit/validation splits."
        )

    labels = sorted(class_pools)
    counts = np.zeros((n_clients, len(labels)), dtype=int)
    proportions = rng.dirichlet([dirichlet_alpha] * len(labels), size=n_clients) if partition_type == "dirichlet" else None
    for client_id in range(n_clients):
        if partition_type == "iid":
            base, rem = divmod(samples_per_client, len(labels))
            counts[client_id] = base
            counts[client_id, :rem] += 1
        else:
            assert proportions is not None
            raw = proportions[client_id] * samples_per_client
            row = np.floor(raw).astype(int)
            remainder = samples_per_client - int(row.sum())
            if remainder:
                row[np.argsort(-(raw - row))[:remainder]] += 1
            counts[client_id] = row

    overflow = np.maximum(counts.sum(axis=0) - capacities, 0)
    deficits = np.zeros(n_clients, dtype=int)
    if partition_type == "iid":
        # Balance fixed-size client assignments against any per-class remainder.
        for class_index in np.flatnonzero(overflow):
            for client_id in rng.permutation(n_clients):
                move = min(int(counts[client_id, class_index]), int(overflow[class_index]))
                counts[client_id, class_index] -= move
                deficits[client_id] += move
                overflow[class_index] -= move
                if overflow[class_index] == 0:
                    break
    else:
        assert proportions is not None
        for class_index in np.flatnonzero(overflow):
            for client_id in np.argsort(proportions[:, class_index]):
                move = min(int(counts[client_id, class_index]), int(overflow[class_index]))
                counts[client_id, class_index] -= move
                deficits[client_id] += move
                overflow[class_index] -= move
                if overflow[class_index] == 0:
                    break

    capacity_left = capacities - counts.sum(axis=0)
    for client_id in range(n_clients):
        candidates = rng.permutation(len(labels)) if partition_type == "iid" else np.argsort(-proportions[client_id])
        for class_index in candidates:
            move = min(int(deficits[client_id]), int(capacity_left[class_index]))
            counts[client_id, class_index] += move
            deficits[client_id] -= move
            capacity_left[class_index] -= move
            if deficits[client_id] == 0:
                break
        if deficits[client_id] != 0:
            raise RuntimeError("Could not allocate requested auxiliary client sizes from remaining class rows")

    pools = {label: list(class_pools[label]) for label in labels}
    clients: dict[int, ClientDataset] = {}
    source_indices: dict[str, list[int]] = {}
    for offset in range(n_clients):
        client_id = first_client_id + offset
        indices: list[int] = []
        for class_index, label in enumerate(labels):
            take = int(counts[offset, class_index])
            indices.extend(pools[label][:take])
            pools[label] = pools[label][take:]
        if partition_type == "iid":
            indices = rng.permutation(indices).tolist()
        idx = torch.tensor(indices, dtype=torch.long)
        clients[client_id] = ClientDataset(client_id, train_images[idx], train_labels[idx])
        source_indices[str(client_id)] = [int(value) for value in indices]
    return clients, source_indices


def make_synthetic_federation(data_config: dict, seed: int) -> Federation:
    """Build a binary task with client-specific class imbalance and feature shift."""
    rng = np.random.default_rng(seed)
    n_clients = int(data_config["n_clients"])
    samples_per_client = int(data_config["samples_per_client"])
    n_features = int(data_config["n_features"])
    heterogeneity = float(data_config["heterogeneity"])
    if not 0.0 <= heterogeneity <= 1.0:
        raise ValueError("heterogeneity must lie in [0, 1].")

    direction = rng.normal(size=n_features)
    direction = direction / np.linalg.norm(direction)
    clients: dict[int, ClientDataset] = {}
    for client_id in range(n_clients):
        class_probability = 0.5 + heterogeneity * (0.42 if client_id % 2 else -0.42)
        labels = rng.binomial(1, class_probability, size=samples_per_client).astype(np.float32)
        client_shift = rng.normal(0.0, 0.45 * heterogeneity, size=n_features)
        features = rng.normal(0.0, 1.0, size=(samples_per_client, n_features))
        features += ((2.0 * labels - 1.0)[:, None] * direction * 1.25) + client_shift
        clients[client_id] = ClientDataset(
            client_id=client_id,
            x=torch.tensor(features, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.float32),
        )

    test_samples = int(data_config["test_samples"])
    test_labels = rng.binomial(1, 0.5, size=test_samples).astype(np.float32)
    test_features = rng.normal(0.0, 1.0, size=(test_samples, n_features))
    test_features += (2.0 * test_labels - 1.0)[:, None] * direction * 1.25
    # This is a held-out population generated from the same global process. It is
    # reserved exclusively for attack calibration/evaluation, never training.
    audit_labels = rng.binomial(1, 0.5, size=test_samples).astype(np.float32)
    audit_features = rng.normal(0.0, 1.0, size=(test_samples, n_features))
    audit_features += (2.0 * audit_labels - 1.0)[:, None] * direction * 1.25
    return Federation(
        clients=clients,
        test_x=torch.tensor(test_features, dtype=torch.float32),
        test_y=torch.tensor(test_labels, dtype=torch.float32),
        audit_x=torch.tensor(audit_features, dtype=torch.float32),
        audit_y=torch.tensor(audit_labels, dtype=torch.float32),
    )


def _load_cifar10_raw(root: str, download: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 data as (train_images, train_labels, test_images, test_labels).

    Tries torchvision if installed; falls back to loading directly from the
    standard CIFAR-10 python batch files (e.g. root/cifar-10-batches-py).
    """
    try:
        from torchvision.datasets import CIFAR10
        try:
            train = CIFAR10(root=root, train=True, download=False)
            test = CIFAR10(root=root, train=False, download=False)
        except RuntimeError:
            if not download:
                raise
            train = CIFAR10(root=root, train=True, download=True)
            test = CIFAR10(root=root, train=False, download=True)
        return np.asarray(train.data), np.asarray(train.targets), np.asarray(test.data), np.asarray(test.targets)
    except Exception:
        import os
        import pickle
        cifar_dir = Path(root) / "cifar-10-batches-py"
        if not cifar_dir.exists():
            if download:
                import tarfile
                import urllib.request
                Path(root).mkdir(parents=True, exist_ok=True)
                url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
                tar_path = Path(root) / "cifar-10-python.tar.gz"
                urllib.request.urlretrieve(url, tar_path)
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=root)
            else:
                raise RuntimeError(
                    f"CIFAR-10 dataset not found in {cifar_dir} and download=False. "
                    "Ensure batches are extracted in data/cifar-10-batches-py or set download: true."
                )
        train_batches = []
        train_targets = []
        for i in range(1, 6):
            batch_path = cifar_dir / f"data_batch_{i}"
            with open(batch_path, "rb") as f:
                entry = pickle.load(f, encoding="bytes")
                train_batches.append(entry[b"data"])
                train_targets.extend(entry[b"labels"])
        train_data = np.vstack(train_batches).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        test_path = cifar_dir / "test_batch"
        with open(test_path, "rb") as f:
            test_entry = pickle.load(f, encoding="bytes")
            test_data = test_entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            test_targets = test_entry[b"labels"]
        return train_data, np.array(train_targets), test_data, np.array(test_targets)


def make_cifar10_binary_federation(data_config: dict, seed: int) -> Federation:
    """Create a small, deterministic real-data Phase 1 federation from CIFAR-10.

    This is a CPU validation configuration using classes 0 and 1 with a linear
    model. It verifies public-data loading, deletion, DP accounting, and attacks;
    it is not the final CIFAR-10/CIFAR-100 multi-class benchmark.
    """
    root = str(data_config.get("dataset_root", "data"))
    download = bool(data_config.get("download", False))
    raw_tr_data, raw_tr_targets, raw_te_data, raw_te_targets = _load_cifar10_raw(root=root, download=download)
    rng = np.random.default_rng(seed)
    train_images = torch.tensor(raw_tr_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    train_labels = torch.tensor(raw_tr_targets, dtype=torch.long)
    test_images = torch.tensor(raw_te_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    test_labels = torch.tensor(raw_te_targets, dtype=torch.long)
    train_mask = (train_labels == 0) | (train_labels == 1)
    test_mask = (test_labels == 0) | (test_labels == 1)
    pool_kernel = int(data_config.get("feature_pool_kernel", 1))
    if pool_kernel < 1 or 32 % pool_kernel:
        raise ValueError("feature_pool_kernel must be a positive divisor of 32.")
    if pool_kernel > 1:
        # Fixed public preprocessing: no learned parameters and no private-data
        # fitting. It reduces DP noise exposure for this CPU feasibility pilot.
        train_images = functional.avg_pool2d(train_images, kernel_size=pool_kernel)
        test_images = functional.avg_pool2d(test_images, kernel_size=pool_kernel)
    x_pool, y_pool = train_images[train_mask].flatten(1), train_labels[train_mask].float()
    x_test, y_test = test_images[test_mask].flatten(1), test_labels[test_mask].float()
    x_pool = (x_pool - 0.5) / 0.5
    x_test = (x_test - 0.5) / 0.5
    n_clients = int(data_config["n_clients"])
    samples_per_client = int(data_config["samples_per_client"])
    audit_samples = int(data_config.get("audit_samples", 400))
    test_samples = int(data_config.get("test_samples", len(y_test)))
    required = n_clients * samples_per_client + audit_samples
    if required > len(y_pool):
        raise ValueError(f"Requested {required} train/audit examples but only {len(y_pool)} selected CIFAR-10 examples exist.")
    # Shuffle label-specific pools then assign a skewed, but auditable client sample.
    pools = {label: rng.permutation(torch.where(y_pool == label)[0].numpy()).tolist() for label in (0, 1)}
    clients: dict[int, ClientDataset] = {}
    heterogeneity = float(data_config.get("heterogeneity", 0.75))
    for client_id in range(n_clients):
        p_one = 0.5 + heterogeneity * (0.40 if client_id % 2 else -0.40)
        labels = rng.binomial(1, p_one, size=samples_per_client)
        indices = []
        for label in labels:
            if not pools[int(label)]:
                raise ValueError("Class pool exhausted; reduce samples_per_client or heterogeneity.")
            indices.append(pools[int(label)].pop())
        idx = torch.tensor(indices, dtype=torch.long)
        clients[client_id] = ClientDataset(client_id=client_id, x=x_pool[idx], y=y_pool[idx])
    remaining = [index for label in (0, 1) for index in pools[label]]
    audit_idx = torch.tensor(rng.choice(remaining, size=audit_samples, replace=False), dtype=torch.long)
    test_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC1FA10]))
    test_idx = torch.tensor(test_rng.choice(len(y_test), size=min(test_samples, len(y_test)), replace=False), dtype=torch.long)
    return Federation(
        clients=clients,
        test_x=x_test[test_idx],
        test_y=y_test[test_idx],
        audit_x=x_pool[audit_idx],
        audit_y=y_pool[audit_idx],
    )


def make_cifar10_multiclass_federation(data_config: dict, seed: int) -> Federation:
    """Create a 10-class, image-tensor CIFAR-10 federation for the Phase 2 CNN model.

    Unlike ``make_cifar10_binary_federation``, this keeps images as ``(N, 3, 32, 32)``
    tensors (no flattening/pooling into fixed public features) so a convolutional
    model can be trained. Client heterogeneity is controlled with a symmetric
    Dirichlet split over the 10 classes: ``alpha -> 0`` yields near-single-class
    clients (highly non-IID) and ``alpha`` large yields near-uniform label mixes,
    matching the standard non-IID-FL convention used by FedAvg-family papers.
    This is a Phase 2 backend: it is CPU-usable for smoke tests but is intended
    to be trained on GPU for the full benchmark grid.
    """
    root = str(data_config.get("dataset_root", "data"))
    download = bool(data_config.get("download", False))
    raw_tr_data, raw_tr_targets, raw_te_data, raw_te_targets = _load_cifar10_raw(root=root, download=download)
    rng = np.random.default_rng(seed)
    train_images = torch.tensor(raw_tr_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    train_labels = torch.tensor(raw_tr_targets, dtype=torch.long)
    test_images = torch.tensor(raw_te_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    test_labels = torch.tensor(raw_te_targets, dtype=torch.long)
    # Fixed public per-channel normalization (CIFAR-10 train-set statistics).
    # No learned parameters and no private-data fitting.
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    n_clients = int(data_config["n_clients"])
    samples_per_client = int(data_config["samples_per_client"])
    auxiliary_n_clients = int(data_config.get("auxiliary_n_clients", 0))
    auxiliary_samples_per_client = int(
        data_config.get("auxiliary_samples_per_client", samples_per_client)
    )
    audit_samples = int(data_config.get("audit_samples", 400))
    validation_samples = int(data_config.get("validation_samples", 0))
    test_samples = int(data_config.get("test_samples", len(test_labels)))
    if audit_samples < 0 or validation_samples < 0 or auxiliary_n_clients < 0:
        raise ValueError("audit/validation sizes and auxiliary_n_clients cannot be negative.")
    if auxiliary_n_clients and auxiliary_samples_per_client <= 0:
        raise ValueError("data.auxiliary_samples_per_client must be positive when auxiliary clients are requested.")
    partition_type = str(data_config.get("partition_type", "dirichlet")).lower()
    if partition_type not in {"iid", "dirichlet"}:
        raise ValueError("data.partition_type must be 'iid' or 'dirichlet'.")
    alpha = float(data_config.get("dirichlet_alpha", data_config.get("heterogeneity_alpha", 100.0)))
    if partition_type == "dirichlet" and alpha <= 0.0:
        raise ValueError("data.dirichlet_alpha must be positive (small values are more non-IID).")
    n_classes = 10
    required = (
        n_clients * samples_per_client
        + audit_samples
        + validation_samples
        + auxiliary_n_clients * auxiliary_samples_per_client
    )
    if required > len(train_labels):
        raise ValueError(f"Requested {required} train/audit/validation examples but only {len(train_labels)} CIFAR-10 train examples exist.")

    class_pools = {label: rng.permutation(torch.where(train_labels == label)[0].numpy()).tolist() for label in range(n_classes)}
    # Reserve exact-size, class-stratified audit and validation rows before
    # assigning any client data. All indices refer to official train rows.
    def reserve_stratified(sample_count: int, split_name: str) -> list[int]:
        base, remainder = divmod(sample_count, n_classes)
        reserved: list[int] = []
        for label in range(n_classes):
            take = base + int(label < remainder)
            if take > len(class_pools[label]):
                raise ValueError(f"Not enough class-{label} CIFAR-10 rows to reserve {split_name}.")
            reserved.extend(class_pools[label][:take])
            class_pools[label] = class_pools[label][take:]
        return reserved

    audit_indices = reserve_stratified(audit_samples, "audit set")
    validation_indices = reserve_stratified(validation_samples, "validation set")

    clients: dict[int, ClientDataset] = {}
    client_source_indices: dict[int, list[int]] = {}
    if partition_type == "iid":
        remaining = [index for label in range(n_classes) for index in class_pools[label]]
        remaining = rng.permutation(remaining).tolist()
        needed = n_clients * samples_per_client
        if needed > len(remaining):
            raise ValueError("Not enough unreserved CIFAR-10 rows for the requested IID client partition.")
        for client_id in range(n_clients):
            indices = remaining[client_id * samples_per_client : (client_id + 1) * samples_per_client]
            idx = torch.tensor(indices, dtype=torch.long)
            clients[client_id] = ClientDataset(client_id=client_id, x=train_images[idx], y=train_labels[idx])
            client_source_indices[client_id] = [int(value) for value in indices]

    else:
        class_proportions = rng.dirichlet(alpha=[alpha] * n_classes, size=n_clients)
        client_counts = np.zeros((n_clients, n_classes), dtype=int)
        for client_id in range(n_clients):
            counts = np.floor(class_proportions[client_id] * samples_per_client).astype(int)
            remainder = samples_per_client - counts.sum()
            if remainder > 0:
                fractional = class_proportions[client_id] * samples_per_client - counts
                counts[np.argsort(-fractional)[:remainder]] += 1
            client_counts[client_id] = counts

        class_capacities = np.array([len(class_pools[label]) for label in range(n_classes)])
        class_overflow = np.maximum(client_counts.sum(axis=0) - class_capacities, 0)
        client_deficits = np.zeros(n_clients, dtype=int)
        for label in np.flatnonzero(class_overflow):
            for client_id in np.argsort(class_proportions[:, label]):
                take = min(client_counts[client_id, label], class_overflow[label])
                client_counts[client_id, label] -= take
                client_deficits[client_id] += take
                class_overflow[label] -= take
                if class_overflow[label] == 0:
                    break

        class_capacity_left = class_capacities - client_counts.sum(axis=0)
        for client_id in range(n_clients):
            for label in np.argsort(-class_proportions[client_id]):
                take = min(client_deficits[client_id], class_capacity_left[label])
                client_counts[client_id, label] += take
                client_deficits[client_id] -= take
                class_capacity_left[label] -= take
                if client_deficits[client_id] == 0:
                    break
            if client_deficits[client_id] != 0:
                raise RuntimeError("Unable to rebalance Dirichlet class counts within available CIFAR-10 pools.")

        for client_id in range(n_clients):
            indices: list[int] = []
            for label in range(n_classes):
                take = int(client_counts[client_id, label])
                indices.extend(class_pools[label][:take])
                class_pools[label] = class_pools[label][take:]
            idx = torch.tensor(indices, dtype=torch.long)
            clients[client_id] = ClientDataset(client_id=client_id, x=train_images[idx], y=train_labels[idx])
            client_source_indices[client_id] = [int(value) for value in indices]

    if partition_type == "iid":
        used_client_indices = {index for indices in client_source_indices.values() for index in indices}
        for label in range(n_classes):
            class_pools[label] = [index for index in class_pools[label] if index not in used_client_indices]
    auxiliary_clients, auxiliary_source_indices = _allocate_auxiliary_clients(
        train_images,
        train_labels,
        class_pools,
        n_clients=auxiliary_n_clients,
        samples_per_client=auxiliary_samples_per_client,
        partition_type=partition_type,
        dirichlet_alpha=alpha,
        rng=rng,
        first_client_id=n_clients,
    )

    audit_idx = torch.tensor(audit_indices, dtype=torch.long)
    # Keep the evaluation rows independent of the client partition RNG so IID
    # and Dirichlet runs with the same seed are measured on the same examples.
    test_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC1FA10]))
    test_idx = torch.tensor(test_rng.choice(len(test_labels), size=min(test_samples, len(test_labels)), replace=False), dtype=torch.long)
    manifest = {
        "dataset": "cifar10",
        "train_source": "official CIFAR-10 train split",
        "test_source": "official CIFAR-10 test split",
        "seed": int(seed),
        "partition_type": partition_type,
        "dirichlet_alpha": alpha if partition_type == "dirichlet" else None,
        "n_clients": n_clients,
        "samples_per_client": samples_per_client,
        "auxiliary_n_clients": len(auxiliary_clients),
        "auxiliary_samples_per_client": auxiliary_samples_per_client if auxiliary_clients else 0,
        "auxiliary_train_indices": auxiliary_source_indices,
        "audit_samples": len(audit_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test_idx),
        "client_train_indices": {str(cid): ids for cid, ids in client_source_indices.items()},
        "audit_train_indices": [int(value) for value in audit_indices],
        "validation_train_indices": [int(value) for value in validation_indices],
        "test_indices": [int(value) for value in test_idx.tolist()],
        "client_class_counts": {
            str(cid): {str(label): int((client.y == label).sum()) for label in range(n_classes)}
            for cid, client in clients.items()
        },
    }
    return Federation(
        clients=clients,
        test_x=test_images[test_idx],
        test_y=test_labels[test_idx],
        audit_x=train_images[audit_idx],
        audit_y=train_labels[audit_idx],
        partition_manifest=manifest,
        validation_x=train_images[torch.tensor(validation_indices, dtype=torch.long)],
        validation_y=train_labels[torch.tensor(validation_indices, dtype=torch.long)],
        auxiliary_clients=auxiliary_clients,
    )


GENERIC_DATASETS = {
    "cifar10": {"class": "CIFAR10", "num_classes": 10, "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"class": "CIFAR100", "num_classes": 100, "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "svhn": {"class": "SVHN", "num_classes": 10, "mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970)},
}


def _partition_image_federation(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    data_config: dict,
    seed: int,
    dataset_name: str = "generic_image",
) -> Federation:
    rng = np.random.default_rng(seed)
    n_clients = int(data_config["n_clients"])
    samples_per_client = int(data_config["samples_per_client"])
    audit_samples = int(data_config.get("audit_samples", 400))
    validation_samples = int(data_config.get("validation_samples", 0))
    test_samples = int(data_config.get("test_samples", len(test_labels)))
    num_classes = int(data_config["num_classes"])
    partition_type = str(data_config.get("partition_type", "dirichlet")).lower()
    if partition_type not in {"iid", "dirichlet"}:
        raise ValueError("data.partition_type must be 'iid' or 'dirichlet'.")
    alpha = float(data_config.get("dirichlet_alpha", data_config.get("heterogeneity_alpha", 100.0)))
    if partition_type == "dirichlet" and alpha <= 0:
        raise ValueError("data.dirichlet_alpha must be positive.")
    auxiliary_n_clients = int(data_config.get("auxiliary_n_clients", 0))
    auxiliary_samples_per_client = int(
        data_config.get("auxiliary_samples_per_client", samples_per_client)
    )
    if audit_samples < 0 or validation_samples < 0 or auxiliary_n_clients < 0:
        raise ValueError("data.audit_samples, data.validation_samples, and data.auxiliary_n_clients cannot be negative.")
    if auxiliary_n_clients and auxiliary_samples_per_client <= 0:
        raise ValueError("data.auxiliary_samples_per_client must be positive when auxiliary clients are requested.")

    required = (
        n_clients * samples_per_client
        + audit_samples
        + validation_samples
        + auxiliary_n_clients * auxiliary_samples_per_client
    )
    if required > len(train_labels):
        raise ValueError(f"Requested {required} train/audit/validation examples, but dataset contains {len(train_labels)}.")

    class_pools = {
        label: rng.permutation(torch.where(train_labels == label)[0].cpu().numpy().tolist())
        for label in range(num_classes)
    }
    def reserve_stratified(sample_count: int) -> list[int]:
        base, remainder = divmod(sample_count, num_classes)
        indices: list[int] = []
        for label in range(num_classes):
            take = base + int(label < remainder)
            if take > len(class_pools[label]):
                raise ValueError(f"Not enough class-{label} rows to reserve a split.")
            indices.extend(int(value) for value in class_pools[label][:take])
            class_pools[label] = class_pools[label][take:]
        return indices

    audit_indices = reserve_stratified(audit_samples)
    validation_indices = reserve_stratified(validation_samples)
    clients: dict[int, ClientDataset] = {}
    client_source_indices: dict[int, list[int]] = {}
    if partition_type == "iid":
        remaining = [int(index) for pool in class_pools.values() for index in pool]
        remaining = rng.permutation(remaining).tolist()
        needed = n_clients * samples_per_client
        if needed > len(remaining):
            raise ValueError("Not enough unreserved examples for the requested IID client partition.")
        for client_id in range(n_clients):
            indices = remaining[client_id * samples_per_client : (client_id + 1) * samples_per_client]
            idx = torch.tensor(indices, dtype=torch.long)
            clients[client_id] = ClientDataset(client_id, train_images[idx], train_labels[idx])
            client_source_indices[client_id] = indices
    else:
        proportions = rng.dirichlet([alpha] * num_classes, size=n_clients)
        client_counts = np.zeros((n_clients, num_classes), dtype=int)
        for client_id in range(n_clients):
            counts = np.floor(proportions[client_id] * samples_per_client).astype(int)
            remainder = samples_per_client - int(counts.sum())
            if remainder:
                fractional = proportions[client_id] * samples_per_client - counts
                counts[np.argsort(-fractional)[:remainder]] += 1
            client_counts[client_id] = counts

        capacities = np.array([len(class_pools[label]) for label in range(num_classes)])
        overflow = np.maximum(client_counts.sum(axis=0) - capacities, 0)
        deficits = np.zeros(n_clients, dtype=int)
        for label in np.flatnonzero(overflow):
            for client_id in np.argsort(proportions[:, label]):
                moved = min(int(client_counts[client_id, label]), int(overflow[label]))
                client_counts[client_id, label] -= moved
                deficits[client_id] += moved
                overflow[label] -= moved
                if overflow[label] == 0:
                    break

        remaining_capacity = capacities - client_counts.sum(axis=0)
        for client_id in range(n_clients):
            for label in np.argsort(-proportions[client_id]):
                moved = min(int(deficits[client_id]), int(remaining_capacity[label]))
                client_counts[client_id, label] += moved
                deficits[client_id] -= moved
                remaining_capacity[label] -= moved
                if deficits[client_id] == 0:
                    break
            if deficits[client_id] != 0:
                raise RuntimeError("Unable to rebalance Dirichlet client counts within class capacities.")

        for client_id in range(n_clients):
            indices: list[int] = []
            for label in range(num_classes):
                take = int(client_counts[client_id, label])
                indices.extend(int(value) for value in class_pools[label][:take])
                class_pools[label] = class_pools[label][take:]
            idx = torch.tensor(indices, dtype=torch.long)
            clients[client_id] = ClientDataset(client_id, train_images[idx], train_labels[idx])
            client_source_indices[client_id] = indices

    if partition_type == "iid":
        used_client_indices = {index for indices in client_source_indices.values() for index in indices}
        for label in range(num_classes):
            class_pools[label] = [int(index) for index in class_pools[label] if int(index) not in used_client_indices]
    auxiliary_clients, auxiliary_source_indices = _allocate_auxiliary_clients(
        train_images,
        train_labels,
        {label: [int(index) for index in pool] for label, pool in class_pools.items()},
        n_clients=auxiliary_n_clients,
        samples_per_client=auxiliary_samples_per_client,
        partition_type=partition_type,
        dirichlet_alpha=alpha,
        rng=rng,
        first_client_id=n_clients,
    )

    test_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC1FA10]))
    excluded_test_indices = [int(index) for index in data_config.get("exclude_test_indices", [])]
    if len(excluded_test_indices) != len(set(excluded_test_indices)):
        raise ValueError("data.exclude_test_indices cannot contain duplicates.")
    if any(index < 0 or index >= len(test_labels) for index in excluded_test_indices):
        raise ValueError("data.exclude_test_indices contains an out-of-range test index.")
    sampled_test_indices = test_rng.choice(
        len(test_labels), size=min(test_samples, len(test_labels)), replace=False
    ).tolist()
    excluded_set = set(excluded_test_indices)
    test_indices = [int(index) for index in sampled_test_indices if int(index) not in excluded_set]
    manifest = {
        "dataset": dataset_name,
        "train_source": f"official {dataset_name} train split",
        "test_source": f"official {dataset_name} test split",
        "seed": int(seed),
        "partition_type": partition_type,
        "dirichlet_alpha": alpha if partition_type == "dirichlet" else None,
        "n_clients": n_clients,
        "samples_per_client": samples_per_client,
        "auxiliary_n_clients": len(auxiliary_clients),
        "auxiliary_samples_per_client": auxiliary_samples_per_client if auxiliary_clients else 0,
        "auxiliary_train_indices": auxiliary_source_indices,
        "audit_samples": len(audit_indices),
        "validation_samples": len(validation_indices),
        "requested_test_samples": min(test_samples, len(test_labels)),
        "test_samples": len(test_indices),
        "client_train_indices": {str(cid): indices for cid, indices in client_source_indices.items()},
        "audit_train_indices": audit_indices,
        "validation_train_indices": validation_indices,
        "test_indices": [int(index) for index in test_indices],
        "test_excluded_indices": [index for index in excluded_test_indices if index in set(sampled_test_indices)],
        "client_class_counts": {
            str(cid): {str(label): int((client.y == label).sum()) for label in range(num_classes)}
            for cid, client in clients.items()
        },
    }
    audit_idx = torch.tensor(audit_indices, dtype=torch.long)
    validation_idx = torch.tensor(validation_indices, dtype=torch.long)
    test_idx = torch.tensor(test_indices, dtype=torch.long)
    return Federation(
        clients=clients,
        test_x=test_images[test_idx],
        test_y=test_labels[test_idx],
        audit_x=train_images[audit_idx],
        audit_y=train_labels[audit_idx],
        partition_manifest=manifest,
        validation_x=train_images[validation_idx],
        validation_y=train_labels[validation_idx],
        auxiliary_clients=auxiliary_clients,
    )


def make_generic_image_federation(data_config: dict, seed: int) -> Federation:
    from torchvision.datasets import CIFAR10, CIFAR100, SVHN

    dataset_name = str(data_config["dataset"]).lower()
    if dataset_name not in GENERIC_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from {sorted(GENERIC_DATASETS)}")
    spec = GENERIC_DATASETS[dataset_name]
    root = str(data_config.get("dataset_root", "data"))
    download = bool(data_config.get("download", True))
    dataset_class = {"CIFAR10": CIFAR10, "CIFAR100": CIFAR100, "SVHN": SVHN}[spec["class"]]

    if dataset_name == "svhn":
        train_set = dataset_class(root=root, split="train", download=download)
        test_set = dataset_class(root=root, split="test", download=download)
        train_labels = torch.tensor(train_set.labels, dtype=torch.long)
        test_labels = torch.tensor(test_set.labels, dtype=torch.long)
    else:
        train_set = dataset_class(root=root, train=True, download=download)
        test_set = dataset_class(root=root, train=False, download=download)
        train_labels = torch.tensor(train_set.targets, dtype=torch.long)
        test_labels = torch.tensor(test_set.targets, dtype=torch.long)

    def to_rgb_tensor(images) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(images), dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1).expand(-1, 3, -1, -1)
        elif tensor.ndim == 4 and tensor.shape[1] == 3:
            pass
        elif tensor.ndim == 4 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected grayscale or RGB images, got shape {tuple(tensor.shape)}")
        return tensor / 255.0

    train_images = to_rgb_tensor(train_set.data)
    test_images = to_rgb_tensor(test_set.data)
    mean = torch.tensor(spec["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(spec["std"]).view(1, 3, 1, 1)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std
    config = {**data_config, "num_classes": spec["num_classes"]}
    return _partition_image_federation(
        train_images, train_labels, test_images, test_labels, config, seed, dataset_name=dataset_name
    )


def make_federation(data_config: dict, seed: int) -> Federation:
    backend = data_config.get("backend", "synthetic")
    if backend == "synthetic":
        return make_synthetic_federation(data_config, seed)
    if backend == "cifar10_binary":
        return make_cifar10_binary_federation(data_config, seed)
    if backend == "cifar10_multiclass":
        return make_cifar10_multiclass_federation(data_config, seed)
    if backend == "vision_classification":
        return make_generic_image_federation(data_config, seed)
    raise ValueError(f"Unsupported dataset backend: {backend}")


def write_deletion_manifest(rows: list[dict[str, int | float | str]], directory: str | Path) -> tuple[Path, str]:
    """Write an auditable request manifest and return its content checksum."""
    if not rows:
        raise ValueError("Deletion manifest cannot be empty.")
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    with buffer:
        handle = buffer
        writer = csv.DictWriter(handle, fieldnames=["request_index", "client_id", "sample_index", "fraction", "action"])
        writer.writeheader()
        writer.writerows(rows)
        content = handle.getvalue().encode("utf-8")
    checksum = hashlib.sha256(content).hexdigest()
    path = output_dir / f"deletion_{checksum[:16]}.csv"
    path.write_bytes(content)
    return path, checksum


def normalized_deletion_requests(deletion_config: dict) -> list[dict]:
    """Normalize one or multiple client requests into a frozen cumulative request list."""
    requests = deletion_config.get("requests")
    if requests is None:
        requests = [{"client_id": deletion_config["client_id"], "fraction": deletion_config.get("fraction", 1.0)}]
    if not isinstance(requests, list) or not requests:
        raise ValueError("deletion.requests must be a non-empty list.")
    normalized = []
    seen = set()
    for index, request in enumerate(requests):
        client_id = int(request["client_id"])
        fraction = float(request.get("fraction", 1.0))
        if not 0 < fraction <= 1:
            raise ValueError("Deletion fraction must be in (0, 1].")
        if client_id in seen:
            raise ValueError("A client may appear at most once in a cumulative request list.")
        normalized.append({"request_index": index, "client_id": client_id, "fraction": fraction})
        seen.add(client_id)
    return normalized


def apply_client_deletion_requests(federation: Federation, deletion_config: dict, seed: int) -> DeletionResult:
    """Create retained and forgotten populations without mutating the source federation.

    Partial deletion is deterministic stratified index sampling within a client.
    A request list is cumulative: targets are retrained without all requested data.
    """
    requests = normalized_deletion_requests(deletion_config)
    retained = dict(federation.clients)
    forgotten_x, forgotten_y, rows = [], [], []
    rng = np.random.default_rng(seed + 40_000)
    for request in requests:
        client_id, fraction = request["client_id"], request["fraction"]
        if client_id not in retained:
            raise KeyError(f"Unknown or already removed client: {client_id}")
        client = retained[client_id]
        n_remove = len(client.y) if fraction == 1.0 else max(1, int(round(len(client.y) * fraction)))
        indices = np.sort(rng.choice(len(client.y), size=n_remove, replace=False))
        index_tensor = torch.tensor(indices, dtype=torch.long)
        forgotten_x.append(client.x[index_tensor])
        forgotten_y.append(client.y[index_tensor])
        rows.extend(
            {
                "request_index": request["request_index"],
                "client_id": client_id,
                "sample_index": int(sample_index),
                "fraction": fraction,
                "action": "forget",
            }
            for sample_index in indices
        )
        if n_remove == len(client.y):
            del retained[client_id]
        else:
            keep = torch.ones(len(client.y), dtype=torch.bool)
            keep[index_tensor] = False
            retained[client_id] = ClientDataset(client_id=client_id, x=client.x[keep], y=client.y[keep])
    if not retained:
        raise ValueError("Deletion cannot remove every client.")
    return DeletionResult(
        retained=retained,
        forgotten_x=torch.cat(forgotten_x),
        forgotten_y=torch.cat(forgotten_y),
        rows=rows,
    )


def write_client_deletion_manifest(
    federation: Federation, forgotten_client_id: int, directory: str | Path
) -> tuple[Path, str]:
    """Backwards-compatible wrapper for one complete-client deletion."""
    result = apply_client_deletion_requests(federation, {"client_id": forgotten_client_id}, seed=0)
    return write_deletion_manifest(result.rows, directory)


def retained_clients(federation: Federation, forgotten_client_id: int) -> dict[int, ClientDataset]:
    """Compatibility helper for a complete one-client deletion."""
    return apply_client_deletion_requests(federation, {"client_id": forgotten_client_id}, seed=0).retained
