"""Small deterministic FedAvg simulator with client-update central DP."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import random
import secrets
import time
from typing import Protocol

import numpy as np
import torch
from torch import nn

from .data import ClientDataset
from .models import GroupNormResNet18, GroupNormWideResNet
from .privacy import _positive_integer


class BinaryLinearModel(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class SmallGroupNormCNN(nn.Module):
    """Compact image model that avoids batch-dependent normalization."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class OriginalSmallGroupNormCNN(nn.Module):
    """The original buggy architecture with spatial-destroying AdaptiveAvgPool2d."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def infer_n_features(sample_x: torch.Tensor) -> int | None:
    """Return the flat feature count for 2-D client tensors, or None for image tensors.

    ``BinaryLinearModel`` needs an explicit input width; ``SmallGroupNormCNN``
    ignores this argument entirely and reads channel count from the data itself,
    so image-shaped ``(N, C, H, W)`` client tensors return ``None``.
    """
    if sample_x.ndim == 2:
        return int(sample_x.shape[1])
    return None


def make_model(model_config: dict | None, n_features: int | None) -> nn.Module:
    name = (model_config or {}).get("name", "binary_linear")
    if name == "binary_linear":
        if n_features is None:
            raise ValueError("model.name='binary_linear' requires 2-D client features; got image-shaped data.")
        return BinaryLinearModel(n_features)
    if name == "small_groupnorm_cnn":
        return SmallGroupNormCNN(int((model_config or {}).get("num_classes", 10)))
    if name == "groupnorm_resnet18":
        return GroupNormResNet18(
            num_classes=int((model_config or {}).get("num_classes", 10)),
            base_width=int((model_config or {}).get("base_width", 64)),
        )
    if name == "groupnorm_wideresnet":
        return GroupNormWideResNet(
            num_classes=int((model_config or {}).get("num_classes", 10)),
            depth=int((model_config or {}).get("depth", 28)),
            widen_factor=int((model_config or {}).get("widen_factor", 4)),
            dropout=float((model_config or {}).get("dropout", 0.2)),
        )
    if name == "original_small_groupnorm_cnn":
        return OriginalSmallGroupNormCNN(int((model_config or {}).get("num_classes", 10)))
    raise ValueError(f"Unknown model.name: {name}")


def _per_example_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float(), reduction="none")
    return nn.functional.cross_entropy(logits, labels.long(), reduction="none")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return ((torch.sigmoid(logits.squeeze(-1)) >= 0.5) == labels.bool()).float().mean()
    return (logits.argmax(dim=1) == labels.long()).float().mean()


@dataclass
class TrainCost:
    rounds: int = 0
    client_updates: int = 0
    local_examples: int = 0
    clipped_client_updates: int = 0
    preclip_update_norm_sum: float = 0.0
    max_preclip_update_norm: float = 0.0
    # Legacy field retained for historical artifact compatibility. Its meaning
    # remains method-specific; use the directional fields for new comparisons.
    communicated_bytes: int = 0
    uplink_tensor_payload_bytes: int = 0
    downlink_tensor_payload_bytes: int = 0
    persistent_storage_bytes: int = 0
    runtime_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "rounds": self.rounds,
            "client_updates": self.client_updates,
            "local_examples": self.local_examples,
            "clipped_client_updates": self.clipped_client_updates,
            "clipping_fraction": (
                self.clipped_client_updates / self.client_updates if self.client_updates else 0.0
            ),
            "mean_preclip_update_norm": (
                self.preclip_update_norm_sum / self.client_updates if self.client_updates else 0.0
            ),
            "max_preclip_update_norm": self.max_preclip_update_norm,
            "communicated_bytes": self.communicated_bytes,
            "communication_accounting_version": 2,
            "uplink_tensor_payload_bytes": self.uplink_tensor_payload_bytes,
            "downlink_tensor_payload_bytes": self.downlink_tensor_payload_bytes,
            "total_tensor_payload_bytes": (
                self.uplink_tensor_payload_bytes + self.downlink_tensor_payload_bytes
            ),
            "persistent_storage_bytes": self.persistent_storage_bytes,
            "runtime_seconds": round(float(self.runtime_seconds), 3),
        }
        # Method-specific provenance stays out of the dataclass fields because
        # those fields define the resumable training-checkpoint schema.
        method_details = getattr(self, "method_details", None)
        if method_details is not None:
            result["method_details"] = method_details
        return result


class IncompatibleSecurityRegimeError(RuntimeError):
    """Raised when an unlearning method violates the active security or privacy contract (e.g. Secure Aggregation)."""
    pass


class SecureAggregator:
    """In-process illustration of pairwise zero-sum masking.

    Under Secure Aggregation:
    1. Participating clients generate pairwise random blinding masks P_{i,j} = -P_{j,i}.
    2. Client i adds its sum of masks to its local clipped delta:
         tilde_delta_i = delta_i + sum_{j != i} P_{i, j}
    3. The central server receives only masked updates tilde_delta_i and sums them.
       All masks cancel out exactly:
         sum_i tilde_delta_i == sum_i delta_i
    This helper itself receives plaintext updates in one Python process before
    masking. It tests cancellation only; it is not cryptographic Secure
    Aggregation and provides no confidentiality against the server.
    """

    @staticmethod
    def aggregate_masked(
        deltas_by_client: dict[int, dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ) -> dict[str, torch.Tensor]:
        client_ids = list(deltas_by_client.keys())
        if not client_ids:
            return {}
        if len(client_ids) == 1:
            return {name: val.detach().clone() for name, val in deltas_by_client[client_ids[0]].items()}

        param_names = list(deltas_by_client[client_ids[0]].keys())
        pairwise_masks: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

        for i_idx, c_i in enumerate(client_ids):
            for c_j in client_ids[i_idx + 1:]:
                pair_seed = int(rng.integers(0, 2**31 - 1))
                pair_gen = torch.Generator().manual_seed(pair_seed)
                mask_ij = {}
                mask_ji = {}
                for name in param_names:
                    ref_tensor = deltas_by_client[c_i][name]
                    noise = torch.randn(ref_tensor.shape, generator=pair_gen, dtype=ref_tensor.dtype).to(ref_tensor.device)
                    mask_ij[name] = noise
                    mask_ji[name] = -noise
                pairwise_masks[(c_i, c_j)] = mask_ij
                pairwise_masks[(c_j, c_i)] = mask_ji

        masked_updates = []
        for c_i in client_ids:
            client_masked = {}
            for name in param_names:
                total_mask = sum(pairwise_masks[(c_i, c_j)][name] for c_j in client_ids if c_j != c_i)
                client_masked[name] = deltas_by_client[c_i][name] + total_mask
            masked_updates.append(client_masked)

        aggregate = {}
        for name in param_names:
            aggregate[name] = torch.stack([mu[name] for mu in masked_updates]).sum(dim=0)
        return aggregate


@dataclass
class RoundHistory:
    selected_client_ids: tuple[int, ...]
    deltas_by_client: dict[int, dict[str, torch.Tensor]] | None
    gaussian_noise: dict[str, torch.Tensor] | None
    aggregate_delta: dict[str, torch.Tensor] | None = None


@dataclass
class FederatedHistory:
    """Private server-side training history for a direct-accumulation FU baseline.

    Individual client updates are sensitive server artifacts. Under Secure
    Aggregation, deltas_by_client is None because individual client updates are
    never revealed to the server.
    """

    initial_state: dict[str, torch.Tensor] | None = None
    rounds: list[RoundHistory] | None = None
    secure_aggregation: bool = False

    def __post_init__(self) -> None:
        if self.rounds is None:
            self.rounds = []

    def storage_bytes(self) -> int:
        """Return tensor-payload bytes retained by this history.

        This is a lower bound on serialized/in-memory storage: it includes the
        initial state and all tensor payloads, but not Python/container
        metadata, allocator overhead, or compression effects.
        """
        total = 0
        if self.initial_state is not None:
            total += sum(value.numel() * value.element_size() for value in self.initial_state.values())
        for round_record in self.rounds or []:
            if round_record.deltas_by_client is not None:
                for delta in round_record.deltas_by_client.values():
                    total += sum(value.numel() * value.element_size() for value in delta.values())
            elif round_record.aggregate_delta is not None:
                total += sum(value.numel() * value.element_size() for value in round_record.aggregate_delta.values())
            if round_record.gaussian_noise is not None:
                total += sum(value.numel() * value.element_size() for value in round_record.gaussian_noise.values())
        return total


def _history_copy_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy retained server history off accelerator memory.

    Per-client model deltas can be many gigabytes over a full ResNet run. They
    are archival inputs to later reconstruction, so keeping them on CPU avoids
    exhausting GPU memory while preserving exact tensor values.
    """
    return {name: value.detach().to(device="cpu", copy=True) for name, value in state.items()}


def set_seed(seed: int) -> None:
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def state_delta(new_state: dict[str, torch.Tensor], old_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: new_state[name] - old_state[name].to(new_state[name].device) for name in old_state}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resume_contract_hash(
    *, clients: dict[int, ClientDataset], seed: int, federated_config: dict,
    privacy_config: dict, model_config: dict | None, n_features: int | None,
    initial_state: dict[str, torch.Tensor] | None,
) -> str:
    """Fingerprint the data and training contract before accepting a checkpoint."""
    digest = hashlib.sha256()
    metadata = {
        "seed": int(seed),
        "federated_config": federated_config,
        "privacy_config": privacy_config,
        "model_config": model_config,
        "n_features": n_features,
        "clients": sorted((int(client_id), int(len(client.y))) for client_id, client in clients.items()),
    }
    digest.update(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8"))
    for client_id in sorted(clients):
        client = clients[client_id]
        digest.update(str(int(client_id)).encode("ascii"))
        for name, tensor in (("x", client.x), ("y", client.y)):
            value = tensor.detach().contiguous().cpu()
            digest.update(name.encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(memoryview(value.numpy()).cast("B"))
    if initial_state is not None:
        for name in sorted(initial_state):
            value = initial_state[name].detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def l2_norm(delta: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum(torch.sum(value.float() ** 2) for value in delta.values()))


def clip_delta(delta: dict[str, torch.Tensor], clip_norm: float) -> dict[str, torch.Tensor]:
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive.")
    norm = l2_norm(delta)
    factor = min(1.0, float(clip_norm / (norm.item() + 1e-12)))
    return {name: value * factor for name, value in delta.items()}


def _local_update(
    global_state: dict[str, torch.Tensor],
    client: ClientDataset,
    config: dict,
    n_features: int | None = None,
    model_config: dict | None = None,
    model: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    device = get_device()
    if model is None:
        model = make_model(model_config, n_features).to(device)
    # Ensure global_state is on the correct device when loading
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    model.train()
    lr = float(config.get("learning_rate", config.get("local_lr", 0.05)))
    optimizer_name = str(config.get("optimizer", "sgd")).lower()
    weight_decay = float(config.get("local_weight_decay", config.get("weight_decay", 0.0)))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("local_momentum", config.get("momentum", 0.0))),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported federated.optimizer: {optimizer_name!r}")
    batch_size = int(config["local_batch_size"])
    local_epochs = int(config["local_epochs"])
    schedule = str(config.get("local_lr_schedule", "constant")).lower()
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, local_epochs))
    elif schedule == "constant":
        scheduler = None
    else:
        raise ValueError(f"Unsupported federated.local_lr_schedule: {schedule!r}")
    label_smoothing = float(config.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("federated.label_smoothing must be in [0, 1).")
    augment = bool(config.get("augment_cifar", False))
    if augment:
        if not (len(client.x.shape) == 4 and client.x.shape[1:] == (3, 32, 32)):
            raise ValueError("federated.augment_cifar requires normalized RGB 32x32 image tensors.")
        mean = torch.as_tensor(config.get("normalization_mean", (0.4914, 0.4822, 0.4465)), device=device).view(1, 3, 1, 1)
        std = torch.as_tensor(config.get("normalization_std", (0.2470, 0.2435, 0.2616)), device=device).view(1, 3, 1, 1)
        if mean.shape != std.shape or torch.any(std <= 0):
            raise ValueError("CIFAR normalization mean/std must be three channels with positive std.")
        crop_padding = int(config.get("crop_padding", 4))
        if crop_padding < 0:
            raise ValueError("federated.crop_padding cannot be negative.")

    for _ in range(local_epochs):
        order = torch.randperm(len(client.y))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            x = client.x[batch_indices].to(device, non_blocking=True)
            y = client.y[batch_indices].to(device, non_blocking=True)
            if augment:
                # Undo the fixed public normalization, apply independent
                # per-image zero-padded random crop and horizontal flip, then
                # normalize again. This keeps padding in pixel space.
                pixels = (x * std + mean).clamp(0.0, 1.0)
                padded = nn.functional.pad(
                    pixels, (crop_padding, crop_padding, crop_padding, crop_padding), value=0.0
                )
                if crop_padding:
                    batch_indices = torch.arange(len(x), device=device)
                    top = torch.randint(0, 2 * crop_padding + 1, (len(x),), device=device)
                    left = torch.randint(0, 2 * crop_padding + 1, (len(x),), device=device)
                    row_indices = top[:, None] + torch.arange(32, device=device)[None, :]
                    col_indices = left[:, None] + torch.arange(32, device=device)[None, :]
                    pixel_grid = padded.permute(0, 2, 3, 1)
                    crops = pixel_grid[
                        batch_indices[:, None, None],
                        row_indices[:, :, None],
                        col_indices[:, None, :],
                    ].permute(0, 3, 1, 2)
                else:
                    crops = pixels
                flip_mask = torch.rand(len(x), device=device) < 0.5
                x = torch.where(flip_mask[:, None, None, None], crops.flip(-1), crops)
                x = (x - mean) / std
            optimizer.zero_grad()
            logits = model(x)
            if label_smoothing and logits.ndim > 1 and logits.shape[-1] > 1:
                loss = nn.functional.cross_entropy(
                    logits, y.long(), label_smoothing=label_smoothing
                )
            else:
                loss = _per_example_loss(logits, y).mean()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    new_state = clone_state(model)
    return {name: new_state[name] - global_state[name].to(new_state[name].device) for name in trainable_names}


class _RandomSource(Protocol):
    def random(self) -> float: ...


def _sample_clients(client_ids: list[int], sample_rate: float, generator: _RandomSource) -> list[int]:
    selected = [client_id for client_id in client_ids if generator.random() < sample_rate]
    return selected


def _make_private_noise_generator(device: torch.device, secure_rng: bool):
    """Create the configured private-mechanism noise source.

    ``secure_rng=False`` preserves the fast, OS-seeded Torch PRNG path used by
    earlier experiments. ``secure_rng=True`` requires torchcsprng and obtains
    its key material from the OS random device; it never falls back silently.
    """
    if secure_rng:
        try:
            import torchcsprng
        except ImportError as error:
            raise RuntimeError(
                "privacy.secure_rng=true requires torchcsprng. Install the pinned "
                "optional secure-DP dependency; refusing to fall back to Torch PRNG."
            ) from error
        return torchcsprng.create_random_device_generator("/dev/urandom")
    generator = torch.Generator(device=device)
    generator.manual_seed(secrets.randbits(64))
    return generator


def _draw_gaussian_like(
    reference: torch.Tensor, std: float, generator: torch.Generator
) -> torch.Tensor:
    sample = torch.empty_like(reference)
    return sample.normal_(mean=0.0, std=std, generator=generator)


def _gaussian_noise_like(
    reference: torch.Tensor,
    std: float,
    generator: torch.Generator,
    *,
    secure_rng: bool,
) -> torch.Tensor:
    """Sample Gaussian noise; secure mode also mitigates known float attacks.

    The secure path follows the current Opacus secure-mode construction: discard
    one Gaussian draw, then sum four independent draws with the target standard
    deviation and divide by two. The resulting ideal variance is unchanged.
    """
    if not secure_rng:
        return _draw_gaussian_like(reference, std, generator)

    _draw_gaussian_like(reference.new_empty((1,)), std, generator)  # discard
    noise = torch.zeros_like(reference)
    for _ in range(4):
        noise.add_(_draw_gaussian_like(reference, std, generator))
    return noise.mul_(0.5)


def _round_learning_rate(config: dict, round_index: int, total_rounds: int) -> float:
    """Return the configured base learning rate after round-level scheduling.

    ``round_lr_schedule`` is separate from ``local_lr_schedule``: this schedule
    decays across communication rounds, while the local schedule (if enabled)
    still controls epochs within each selected client's update.
    """
    base_lr = float(config.get("learning_rate", config.get("local_lr", 0.05)))
    schedule = str(config.get("round_lr_schedule", "constant")).lower()
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unsupported federated.round_lr_schedule: {schedule!r}")
    min_ratio = float(config.get("round_lr_min_ratio", 0.01))
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("federated.round_lr_min_ratio must be in [0, 1].")
    if total_rounds <= 1:
        return base_lr
    progress = min(max(round_index, 0), total_rounds - 1) / (total_rounds - 1)
    cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * float(cosine_factor))


def train_federated(
    *,
    clients: dict[int, ClientDataset],
    n_features: int | None = None,
    federated_config: dict,
    privacy_config: dict,
    seed: int,
    initial_state: dict[str, torch.Tensor] | None = None,
    history: FederatedHistory | None = None,
    model_config: dict | None = None,
    model: nn.Module | None = None,
    round_callback: Callable[[int, nn.Module], None] | None = None,
    resume_state: dict | None = None,
    checkpoint_callback: Callable[[dict], None] | None = None,
    checkpoint_interval: int = 0,
) -> tuple[nn.Module, TrainCost]:
    """Train with an accountant-aligned Poisson-subsampled Gaussian mechanism.

    For private rounds, the server sums clipped updates, adds `N(0, (sigma*C)^2)`
    noise to the sum, then divides by the *fixed public* `q*N` normalizer. This
    is a post-processing of the Poisson-sampled Gaussian mechanism accounted for
    in `privacy.py`. It never divides by the realized sample size.
    """
    if not clients:
        raise ValueError("Cannot train without retained clients.")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval cannot be negative.")
    if checkpoint_callback is not None and checkpoint_interval <= 0:
        raise ValueError("A positive checkpoint_interval is required with checkpoint_callback.")
    if checkpoint_interval > 0 and checkpoint_callback is None:
        raise ValueError("checkpoint_callback is required with a positive checkpoint_interval.")
    if history is not None and (resume_state is not None or checkpoint_callback is not None):
        raise ValueError(
            "Round checkpoint/resume is not supported with FederatedHistory: saving per-client updates "
            "would persist sensitive training history. Use a history-free utility run."
        )
    set_seed(seed)
    device = get_device()
    private = bool(privacy_config["enabled"])
    # A published experiment seed must not reveal the Poisson participation
    # schedule or Gaussian noise. Use independent OS-entropy-seeded generators
    # for private mechanism randomness; keep deterministic experiment seeding
    # for model initialization, local training, and non-private controls.
    sampling_rng = secrets.SystemRandom() if private else np.random.default_rng(seed)
    sec_agg = bool(privacy_config.get("secure_aggregation", False)) or bool(federated_config.get("secure_aggregation", False))
    mask_rng = np.random.default_rng(secrets.randbits(128) if private else seed) if sec_agg else None
    secure_noise_rng = private and bool(privacy_config.get("secure_rng", False))
    noise_generator = None
    if private:
        noise_generator = _make_private_noise_generator(device, secure_noise_rng)
    if model is not None:
        model = model.to(device)
        local_model = deepcopy(model).to(device)
    else:
        model = make_model(model_config, n_features).to(device)
        local_model = make_model(model_config, n_features).to(device)
    if initial_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    global_state = clone_state(model)
    client_ids = sorted(clients)
    sample_rate = float(federated_config["client_sample_rate"])
    clip_norm = float(privacy_config["clip_norm"]) if private else None
    noise_multiplier = float(privacy_config["noise_multiplier"]) if private else 0.0
    normalizer = float(privacy_config["population_size"]) * sample_rate if private else None
    if not math.isfinite(sample_rate) or not 0.0 < sample_rate <= 1.0:
        raise ValueError("client_sample_rate must be finite and in (0, 1].")
    if private:
        public_population_size = _positive_integer(
            privacy_config["population_size"], "privacy.population_size"
        )
        if public_population_size != len(client_ids):
            raise ValueError(
                "Client-level Poisson DP requires privacy.population_size to equal the number "
                f"of clients sampled by this run; got {public_population_size} vs {len(client_ids)}."
            )
        if clip_norm is None or not math.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("Private training requires a finite positive clip_norm.")
        if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
            raise ValueError("Private training requires a finite positive noise_multiplier.")
        if normalizer is None or not math.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("Private Poisson aggregation needs positive fixed population_size * sample_rate.")
    raw_rounds = federated_config["rounds"]
    total_rounds = _positive_integer(raw_rounds, "federated.rounds")
    contract_hash = None
    start_round = 0
    prior_runtime_seconds = 0.0
    if checkpoint_callback is not None or resume_state is not None:
        contract_hash = _resume_contract_hash(
            clients=clients,
            seed=seed,
            federated_config=federated_config,
            privacy_config=privacy_config,
            model_config=model_config,
            n_features=n_features,
            initial_state=initial_state,
        )
    if resume_state is not None:
        checkpoint_version = int(resume_state.get("checkpoint_version", -1))
        if checkpoint_version not in {1, 2}:
            raise ValueError("Unsupported federated checkpoint version.")
        if resume_state.get("contract_sha256") != contract_hash:
            raise ValueError("Checkpoint data/config fingerprint does not match this training run.")
        if bool(resume_state.get("private")) != private:
            raise ValueError("Checkpoint privacy mode does not match this training run.")
        start_round = int(resume_state.get("completed_rounds", -1))
        if not 0 <= start_round <= total_rounds:
            raise ValueError("Checkpoint completed_rounds is outside this run's round budget.")
        if int(resume_state.get("total_rounds", -1)) != total_rounds:
            raise ValueError("Checkpoint total_rounds does not match this training run.")
        saved_cost = resume_state.get("cost", {})
        current_cost_fields = set(TrainCost.__dataclass_fields__)
        if checkpoint_version == 1:
            # Version 1 counted only client uploads in `communicated_bytes`.
            # Recover directional totals for prior rounds from that upload
            # counter, update count, and the unchanged model tensor shapes.
            legacy_cost_fields = current_cost_fields - {
                "uplink_tensor_payload_bytes",
                "downlink_tensor_payload_bytes",
            }
            if set(saved_cost) != legacy_cost_fields:
                raise ValueError("Checkpoint v1 training-cost fields are incomplete or unsupported.")
            saved_cost = {
                **saved_cost,
                "uplink_tensor_payload_bytes": int(saved_cost["communicated_bytes"]),
                "downlink_tensor_payload_bytes": int(saved_cost["client_updates"])
                * sum(value.numel() * value.element_size() for value in resume_state["model_state"].values()),
            }
        elif set(saved_cost) != current_cost_fields:
            raise ValueError("Checkpoint training-cost fields are incomplete or unsupported.")
        cost = TrainCost(**saved_cost)
        if cost.rounds != start_round:
            raise ValueError("Checkpoint round counter does not match its accumulated cost.")
        global_state = {
            name: value.detach().to(device).clone()
            for name, value in resume_state["model_state"].items()
        }
        model.load_state_dict(global_state)
        if not private:
            sampling_rng.bit_generator.state = resume_state["sampling_rng_state"]
            if mask_rng is not None:
                mask_rng.bit_generator.state = resume_state["mask_rng_state"]
        # Private Poisson sampling, Gaussian noise, and secure-aggregation
        # masks are freshly initialized from OS entropy. No mechanism RNG state
        # is serialized or restored by a checkpoint.
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if device.type == "cuda":
            saved_cuda_states = resume_state.get("cuda_rng_states", [])
            if len(saved_cuda_states) != torch.cuda.device_count():
                raise ValueError("Checkpoint CUDA RNG-state count differs from the visible device set.")
            torch.cuda.set_rng_state_all(saved_cuda_states)
        if resume_state.get("python_rng_state") is not None:
            random.setstate(resume_state["python_rng_state"])
        prior_runtime_seconds = float(cost.runtime_seconds)
    else:
        cost = TrainCost()
    if history is not None:
        if history.initial_state is not None or history.rounds:
            raise ValueError("Pass an empty FederatedHistory to one training run only.")
        history.initial_state = _history_copy_to_cpu(clone_state(model))
    start_time = time.perf_counter()

    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    round_config = dict(federated_config)
    for round_index in range(start_round, total_rounds):
        round_config["learning_rate"] = _round_learning_rate(federated_config, round_index, total_rounds)
        selected = _sample_clients(client_ids, sample_rate, sampling_rng)
        cost.rounds += 1
        if selected:
            model_payload_bytes = sum(
                value.numel() * value.element_size() for value in global_state.values()
            )
            cost.downlink_tensor_payload_bytes += len(selected) * model_payload_bytes
        if not selected and not private:
            continue
        deltas = []
        for client_id in selected:
            delta = _local_update(
                global_state, clients[client_id], round_config, n_features, model_config, model=local_model
            )
            if private:
                preclip_norm = float(l2_norm(delta).item())
                cost.preclip_update_norm_sum += preclip_norm
                cost.max_preclip_update_norm = max(cost.max_preclip_update_norm, preclip_norm)
                if preclip_norm > clip_norm:
                    cost.clipped_client_updates += 1
                delta = clip_delta(delta, clip_norm)
            deltas.append(delta)
            delta_payload_bytes = sum(
                value.numel() * value.element_size() for value in delta.values()
            )
            cost.uplink_tensor_payload_bytes += delta_payload_bytes
            # Preserve the historical upload-only counter's semantics.
            cost.communicated_bytes += delta_payload_bytes
            cost.client_updates += 1
            cost.local_examples += len(clients[client_id].y) * int(federated_config["local_epochs"])
        round_noise: dict[str, torch.Tensor] | None = {} if private else None

        # Secure Aggregation: pairwise zero-sum masking conceals individual deltas from server
        if sec_agg and selected:
            deltas_dict = {client_id: delta for client_id, delta in zip(selected, deltas)}
            if mask_rng is None:
                raise RuntimeError("Secure aggregation requires an initialized mask RNG.")
            raw_aggregate = SecureAggregator.aggregate_masked(deltas_dict, mask_rng)
        else:
            raw_aggregate = {}
            for name in trainable_names:
                raw_aggregate[name] = torch.stack([delta[name] for delta in deltas]).sum(dim=0) if deltas else torch.zeros_like(global_state[name])

        for name in trainable_names:
            aggregate = raw_aggregate[name] if name in raw_aggregate else torch.zeros_like(global_state[name])
            if private:
                noise = _gaussian_noise_like(
                    aggregate,
                    noise_multiplier * clip_norm,
                    noise_generator,
                    secure_rng=secure_noise_rng,
                )
                round_noise[name] = noise.detach().clone()
                aggregate = (aggregate + noise) / normalizer
            else:
                aggregate = aggregate / len(selected)
            global_state[name] = global_state[name] + aggregate
        model.load_state_dict(global_state)
        if round_callback is not None:
            round_callback(cost.rounds, model)
        if history is not None:
            if sec_agg:
                history.secure_aggregation = True
                history.rounds.append(
                    RoundHistory(
                        selected_client_ids=tuple(selected),
                        deltas_by_client=None,
                        gaussian_noise=_history_copy_to_cpu(round_noise) if round_noise is not None else None,
                        aggregate_delta=_history_copy_to_cpu(raw_aggregate),
                    )
                )
            else:
                history.rounds.append(
                    RoundHistory(
                        selected_client_ids=tuple(selected),
                        deltas_by_client={client_id: _history_copy_to_cpu(delta) for client_id, delta in zip(selected, deltas)},
                        gaussian_noise=_history_copy_to_cpu(round_noise) if round_noise is not None else None,
                    )
                )
        if checkpoint_callback is not None and (
            cost.rounds % checkpoint_interval == 0 or cost.rounds == total_rounds
        ):
            checkpoint_cost = {
                name: getattr(cost, name)
                for name in TrainCost.__dataclass_fields__
            }
            checkpoint_cost["runtime_seconds"] = prior_runtime_seconds + time.perf_counter() - start_time
            payload = {
                "checkpoint_version": 2,
                "contract_sha256": contract_hash,
                "private": private,
                "completed_rounds": cost.rounds,
                "total_rounds": total_rounds,
                "model_state": {name: value.detach().cpu().clone() for name, value in global_state.items()},
                "cost": checkpoint_cost,
                "torch_rng_state": torch.get_rng_state().cpu(),
                "cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()]
                if device.type == "cuda" else [],
                "python_rng_state": random.getstate(),
                "mechanism_rng_policy": (
                    "fresh_os_entropy_on_resume; private sampler/noise/mask state is not serialized"
                    if private else "experiment_rng_state_saved"
                ),
            }
            if not private:
                payload["sampling_rng_state"] = sampling_rng.bit_generator.state
                if mask_rng is not None:
                    payload["mask_rng_state"] = mask_rng.bit_generator.state
            checkpoint_callback(payload)
    cost.runtime_seconds = prior_runtime_seconds + time.perf_counter() - start_time
    if history is not None:
        cost.persistent_storage_bytes = history.storage_bytes()
    return model, cost


def reconstruct_from_cached_updates(
    *,
    history: FederatedHistory,
    forgotten_client_ids: set[int],
    n_features: int | None,
    federated_config: dict,
    privacy_config: dict,
    model_config: dict | None = None,
    model_template: nn.Module | None = None,
) -> tuple[nn.Module, TrainCost]:
    """Direct-accumulation reconstruction from cached per-client updates.

    This is a transparent FedEraser-family baseline, not a faithful implementation
    of every FedEraser calibration step and not certified unlearning. Stored
    client-specific updates are sensitive server-side history; this method has a
    separate exposure model from final-model DP post-processing. In private mode,
    it reuses each original round's Gaussian noise. If both the original and
    reconstructed model weights are released, their difference cancels that
    shared noise and exposes the cumulative deleted-client update contribution
    (under the fixed public normalizer). The original training ledger alone does
    not certify this joint release.
    """
    if history.initial_state is None or not history.rounds:
        raise ValueError("Cached-update reconstruction requires a non-empty training history.")
    if history.secure_aggregation or any(r.deltas_by_client is None for r in history.rounds):
        raise IncompatibleSecurityRegimeError(
            "Cached Reconstruction requires individual client historical updates, "
            "which are cryptographically hidden and strictly unavailable under Secure Aggregation."
        )
    start_time = time.perf_counter()
    device = get_device()
    private = bool(privacy_config["enabled"])
    sample_rate = float(federated_config["client_sample_rate"])
    if not math.isfinite(sample_rate) or not 0.0 < sample_rate <= 1.0:
        raise ValueError("client_sample_rate must be finite and in (0, 1].")
    normalizer = (
        _positive_integer(privacy_config["population_size"], "privacy.population_size") * sample_rate
        if private
        else None
    )
    global_state = {name: value.detach().to(device).clone() for name, value in history.initial_state.items()}
    update_names = {
        name
        for round_record in history.rounds
        for delta in round_record.deltas_by_client.values()
        for name in delta
    }
    cost = TrainCost(rounds=len(history.rounds), persistent_storage_bytes=history.storage_bytes())
    for round_record in history.rounds:
        retained_deltas = [delta for client_id, delta in round_record.deltas_by_client.items() if client_id not in forgotten_client_ids]
        for name in update_names:
            aggregate = (
                torch.stack([delta[name] for delta in retained_deltas]).sum(dim=0).to(device)
                if retained_deltas
                else torch.zeros_like(global_state[name])
            )
            if private:
                if round_record.gaussian_noise is None:
                    raise ValueError("Private cached reconstruction requires stored per-round Gaussian noise.")
                aggregate = (aggregate + round_record.gaussian_noise[name].to(device)) / normalizer
            elif retained_deltas:
                aggregate = aggregate / len(retained_deltas)
            global_state[name] = global_state[name] + aggregate
    cost.runtime_seconds = time.perf_counter() - start_time
    model = deepcopy(model_template).to(device) if model_template is not None else make_model(model_config, n_features).to(device)
    model.load_state_dict(global_state)
    return model, cost


def federated_eraser(
    *,
    history: FederatedHistory,
    clients: dict[int, ClientDataset],
    forgotten_client_ids: set[int],
    n_features: int | None,
    federated_config: dict,
    privacy_config: dict,
    calibration_ratio: float = 0.5,
    model_config: dict | None = None,
    model_template: nn.Module | None = None,
) -> tuple[nn.Module, TrainCost]:
    """FedEraser-style historical recalibration baseline; fidelity is unverified.

    FedEraser reconstructs the global model across rounds using historical updates
    and local calibration updates from retained clients:
    1. Historical update requirement: The server accesses historical per-client updates
       and per-round noise stored during initial training.
    2. Calibration procedure: At each round t, retained clients that participated in round t
       receive the current unlearned model w'_t and perform local calibration training
       to find the unlearning gradient direction delta_tilde.
    3. Update calibration: The update is calibrated by preserving the historical update's
       magnitude while adopting the new direction:
       delta_bar = ||delta_historical|| * (delta_tilde / ||delta_tilde||).
    4. Reconstruction: Calibrated updates are aggregated (with central DP noise if private)
       to produce w'_{t+1}.

    Assumptions & Privacy Implications:
    - Server must retain all historical client-level updates (high persistent storage).
    - Retained clients must be available and re-train locally (computation + communication).
    - Accesses retained raw data; therefore this is NOT DP post-processing of the final model.
    - In private mode, the original round noise is reused. If both original and
      erased weights are released, their shared noise is correlated; the original
      training ledger does not by itself certify that joint release.
    """
    if not 0.0 < calibration_ratio <= 1.0:
        raise ValueError("calibration_ratio must be in (0, 1].")
    if history.initial_state is None or not history.rounds:
        raise ValueError("FedEraser requires a non-empty training history with initial state.")
    if history.secure_aggregation or any(r.deltas_by_client is None for r in history.rounds):
        raise IncompatibleSecurityRegimeError(
            "FedEraser requires historical individual client update directions and magnitudes, "
            "which violates Secure Aggregation privacy guarantees."
        )

    start_time = time.perf_counter()
    device = get_device()
    private = bool(privacy_config.get("enabled", False))
    sample_rate = float(federated_config["client_sample_rate"])
    if not math.isfinite(sample_rate) or not 0.0 < sample_rate <= 1.0:
        raise ValueError("client_sample_rate must be finite and in (0, 1].")
    # Normalizer for Poisson DP matches retained clients count or privacy population_size
    normalizer = (
        _positive_integer(
            privacy_config.get("population_size", len(clients)), "privacy.population_size"
        ) * sample_rate
        if private
        else None
    )

    global_state = {name: value.detach().to(device).clone() for name, value in history.initial_state.items()}
    update_names = {
        name
        for round_record in history.rounds
        for delta in round_record.deltas_by_client.values()
        for name in delta
    }
    cost = TrainCost(rounds=0, persistent_storage_bytes=history.storage_bytes())

    local_epochs = int(federated_config.get("local_epochs", 1))
    cali_epochs = max(1, int(round(local_epochs * calibration_ratio)))
    cost.method_details = {
        "algorithm": "federated_eraser_style_recalibration",
        "requested_local_epochs": local_epochs,
        "requested_calibration_ratio": float(calibration_ratio),
        "realized_calibration_epochs_per_client_per_round": cali_epochs,
        "realized_calibration_ratio": cali_epochs / local_epochs if local_epochs else None,
        "calibration_rounds": len(history.rounds),
    }
    cali_config = {**federated_config, "local_epochs": cali_epochs}
    calibration_model = deepcopy(model_template).to(device) if model_template is not None else None

    for round_record in history.rounds:
        cost.rounds += 1
        # Retained participants in this round
        retained_cids = [
            cid for cid in round_record.selected_client_ids
            if cid not in forgotten_client_ids and cid in clients
        ]

        if not retained_cids and not private:
            continue

        calibrated_deltas = []
        for cid in retained_cids:
            client_data = clients[cid]
            # 1. Communication: server sends current unlearned model w'_t to client
            downlink_bytes = sum(v.numel() * v.element_size() for v in global_state.values())
            cost.communicated_bytes += downlink_bytes
            cost.downlink_tensor_payload_bytes += downlink_bytes

            # 2. Calibration training on client local data from w'_t
            delta_tilde = _local_update(
                global_state, client_data, cali_config, n_features, model_config, model=calibration_model
            )
            cost.client_updates += 1
            cost.local_examples += len(client_data.y) * cali_epochs

            # 3. Communication: client sends delta_tilde back to server
            uplink_bytes = sum(v.numel() * v.element_size() for v in delta_tilde.values())
            cost.communicated_bytes += uplink_bytes
            cost.uplink_tensor_payload_bytes += uplink_bytes

            # 4. Calibrate update: scale new direction by historical magnitude
            hist_delta = round_record.deltas_by_client.get(cid)
            if hist_delta is not None:
                h_norm = l2_norm(hist_delta)
                t_norm = l2_norm(delta_tilde)
                factor = float(h_norm) / (float(t_norm) + 1e-12)
                cal_delta = {k: v * factor for k, v in delta_tilde.items()}
            else:
                cal_delta = delta_tilde

            calibrated_deltas.append(cal_delta)

        # 5. Aggregate calibrated updates
        for name in update_names:
            if calibrated_deltas:
                aggregate = torch.stack([d[name] for d in calibrated_deltas]).sum(dim=0)
            else:
                aggregate = torch.zeros_like(global_state[name])

            if private:
                if round_record.gaussian_noise is None:
                    raise ValueError("Private FedEraser requires stored per-round Gaussian noise.")
                aggregate = (aggregate + round_record.gaussian_noise[name].to(device)) / normalizer
            elif calibrated_deltas:
                aggregate = aggregate / len(calibrated_deltas)

            global_state[name] = global_state[name] + aggregate

    cost.runtime_seconds = time.perf_counter() - start_time
    model = deepcopy(model_template).to(device) if model_template is not None else make_model(model_config, n_features).to(device)
    model.load_state_dict(global_state)
    return model, cost


def finetune_retained(
    *,
    full_model: nn.Module | None = None,
    model: nn.Module | None = None,
    clients: dict[int, ClientDataset] | None = None,
    retained_clients: dict[int, ClientDataset] | None = None,
    n_features: int | None = None,
    federated_config: dict,
    unlearning_rounds: int | None = None,
    seed: int,
    model_config: dict | None = None,
    privacy_config: dict | None = None,
) -> tuple[nn.Module, TrainCost]:
    """A labelled baseline, not a certified unlearning procedure.

    It accesses retained raw data. Its output therefore has a distinct privacy
    exposure from pure post-processing; callers must keep this distinction visible.
    """
    base_model = full_model if full_model is not None else model
    if base_model is None:
        raise ValueError("finetune_retained requires full_model or model.")
    client_dict = clients if clients is not None else retained_clients
    if client_dict is None:
        raise ValueError("finetune_retained requires clients or retained_clients.")
    updated_config = deepcopy(federated_config)
    if unlearning_rounds is not None:
        updated_config["rounds"] = int(unlearning_rounds)
    elif "rounds" not in updated_config:
        updated_config["rounds"] = 4
    priv_cfg = privacy_config if privacy_config is not None else {"enabled": False}
    # train_federated updates the model object it receives in place. Fine-tuning
    # must produce a separate candidate so callers can still evaluate the
    # untouched pre-deletion model as the no-action baseline.
    finetune_model = deepcopy(base_model)
    return train_federated(
        clients=client_dict,
        n_features=n_features,
        federated_config=updated_config,
        privacy_config=priv_cfg,
        seed=seed,
        initial_state=clone_state(base_model),
        model_config=model_config,
        model=finetune_model,
    )


def flatten_logits(model: nn.Module, x: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    device = get_device()
    model = model.to(device)
    model.eval()
    if len(x) <= batch_size:
        with torch.no_grad():
            return model(x.to(device)).detach().cpu()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = x[i : i + batch_size].to(device)
            outputs.append(model(batch_x).detach().cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0)


def evaluate_loss_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    device = get_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = flatten_logits(model, x)
        loss = _per_example_loss(logits, y.cpu()).mean().item()
    accuracy = _accuracy(logits, y.cpu()).item()
    return {"loss": float(loss), "accuracy": float(accuracy)}


def total_examples(clients: Iterable[ClientDataset]) -> int:
    return sum(len(client.y) for client in clients)
