"""Small deterministic FedAvg simulator with client-update central DP."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn

from .data import ClientDataset
from .models import GroupNormResNet18, GroupNormResNet50


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
    if name == "groupnorm_resnet50":
        return GroupNormResNet50(
            num_classes=int((model_config or {}).get("num_classes", 10)),
            base_width=int((model_config or {}).get("base_width", 64)),
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
    communicated_bytes: int = 0
    persistent_storage_bytes: int = 0
    runtime_seconds: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "rounds": self.rounds,
            "client_updates": self.client_updates,
            "local_examples": self.local_examples,
            "communicated_bytes": self.communicated_bytes,
            "persistent_storage_bytes": self.persistent_storage_bytes,
            "runtime_seconds": round(float(self.runtime_seconds), 3),
        }


@dataclass
class RoundHistory:
    selected_client_ids: tuple[int, ...]
    deltas_by_client: dict[int, dict[str, torch.Tensor]]
    gaussian_noise: dict[str, torch.Tensor] | None


@dataclass
class FederatedHistory:
    """Private server-side training history for a direct-accumulation FU baseline.

    Individual client updates are sensitive server artifacts. This object is kept
    only in memory in the pilot; using it for unlearning is not DP post-processing
    of the final released model and its storage cost is reported.
    """

    initial_state: dict[str, torch.Tensor] | None = None
    rounds: list[RoundHistory] | None = None

    def __post_init__(self) -> None:
        if self.rounds is None:
            self.rounds = []

    def storage_bytes(self) -> int:
        total = 0
        for round_record in self.rounds or []:
            for delta in round_record.deltas_by_client.values():
                total += sum(value.numel() * value.element_size() for value in delta.values())
            if round_record.gaussian_noise is not None:
                total += sum(value.numel() * value.element_size() for value in round_record.gaussian_noise.values())
        return total


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def state_delta(new_state: dict[str, torch.Tensor], old_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: new_state[name] - old_state[name] for name in old_state}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_norm(delta: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum(torch.sum(value.float() ** 2) for value in delta.values()))


def clip_delta(delta: dict[str, torch.Tensor], clip_norm: float) -> dict[str, torch.Tensor]:
    norm = l2_norm(delta)
    factor = min(1.0, float(clip_norm / (norm.item() + 1e-12)))
    return {name: value * factor for name, value in delta.items()}


def _local_update(
    global_state: dict[str, torch.Tensor], client: ClientDataset, config: dict, n_features: int, model_config: dict | None
) -> dict[str, torch.Tensor]:
    device = get_device()
    model = make_model(model_config, n_features).to(device)
    # Ensure global_state is on the correct device when loading
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config["learning_rate"]))
    batch_size = int(config["local_batch_size"])
    for _ in range(int(config["local_epochs"])):
        for start in range(0, len(client.y), batch_size):
            x = client.x[start : start + batch_size].to(device, non_blocking=True)
            y = client.y[start : start + batch_size].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = _per_example_loss(model(x), y).mean()
            loss.backward()
            optimizer.step()
    return state_delta(clone_state(model), global_state)


def _sample_clients(client_ids: list[int], sample_rate: float, generator: np.random.Generator) -> list[int]:
    selected = [client_id for client_id in client_ids if generator.random() < sample_rate]
    return selected


def train_federated(
    *,
    clients: dict[int, ClientDataset],
    n_features: int,
    federated_config: dict,
    privacy_config: dict,
    seed: int,
    initial_state: dict[str, torch.Tensor] | None = None,
    history: FederatedHistory | None = None,
    model_config: dict | None = None,
) -> tuple[nn.Module, TrainCost]:
    """Train with an accountant-aligned Poisson-subsampled Gaussian mechanism.

    For private rounds, the server sums clipped updates, adds `N(0, (sigma*C)^2)`
    noise to the sum, then divides by the *fixed public* `q*N` normalizer. This
    is a post-processing of the Poisson-sampled Gaussian mechanism accounted for
    in `privacy.py`. It never divides by the realized sample size.
    """
    if not clients:
        raise ValueError("Cannot train without retained clients.")
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()
    model = make_model(model_config, n_features).to(device)
    if initial_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    global_state = clone_state(model)
    if history is not None:
        if history.initial_state is not None or history.rounds:
            raise ValueError("Pass an empty FederatedHistory to one training run only.")
        history.initial_state = clone_state(model)
    client_ids = sorted(clients)
    sample_rate = float(federated_config["client_sample_rate"])
    private = bool(privacy_config["enabled"])
    clip_norm = float(privacy_config["clip_norm"]) if private else None
    noise_multiplier = float(privacy_config["noise_multiplier"]) if private else 0.0
    normalizer = float(privacy_config["population_size"]) * sample_rate if private else None
    if private and normalizer <= 0:
        raise ValueError("Private Poisson aggregation needs positive fixed population_size * sample_rate.")
    cost = TrainCost()
    start_time = time.perf_counter()

    for _round in range(int(federated_config["rounds"])):
        selected = _sample_clients(client_ids, sample_rate, rng)
        cost.rounds += 1
        if not selected and not private:
            continue
        deltas = []
        for client_id in selected:
            delta = _local_update(global_state, clients[client_id], federated_config, n_features, model_config)
            if private:
                delta = clip_delta(delta, clip_norm)
            deltas.append(delta)
            cost.client_updates += 1
            cost.local_examples += len(clients[client_id].y) * int(federated_config["local_epochs"])
        round_noise: dict[str, torch.Tensor] | None = {} if private else None
        for name in global_state:
            aggregate = torch.stack([delta[name] for delta in deltas]).sum(dim=0) if deltas else torch.zeros_like(global_state[name])
            if private:
                noise = torch.randn_like(aggregate) * (noise_multiplier * clip_norm)
                round_noise[name] = noise.detach().clone()
                aggregate = (aggregate + noise) / normalizer
            else:
                aggregate = aggregate / len(selected)
            global_state[name] = global_state[name] + aggregate
            cost.communicated_bytes += sum(delta[name].numel() * delta[name].element_size() for delta in deltas)
        model.load_state_dict(global_state)
        if history is not None:
            history.rounds.append(
                RoundHistory(
                    selected_client_ids=tuple(selected),
                    deltas_by_client={client_id: {name: value.detach().clone() for name, value in delta.items()} for client_id, delta in zip(selected, deltas)},
                    gaussian_noise=round_noise,
                )
            )
    cost.runtime_seconds = time.perf_counter() - start_time
    if history is not None:
        cost.persistent_storage_bytes = history.storage_bytes()
    return model, cost


def reconstruct_from_cached_updates(
    *,
    history: FederatedHistory,
    forgotten_client_ids: set[int],
    n_features: int,
    federated_config: dict,
    privacy_config: dict,
    model_config: dict | None = None,
) -> tuple[nn.Module, TrainCost]:
    """Direct-accumulation reconstruction from cached per-client updates.

    This is a transparent FedEraser-family baseline, not a faithful implementation
    of every FedEraser calibration step and not certified unlearning. Stored
    client-specific updates are sensitive server-side history; this method has a
    separate exposure model from final-model DP post-processing.
    """
    if history.initial_state is None or not history.rounds:
        raise ValueError("Cached-update reconstruction requires a non-empty training history.")
    start_time = time.perf_counter()
    device = get_device()
    private = bool(privacy_config["enabled"])
    sample_rate = float(federated_config["client_sample_rate"])
    normalizer = float(privacy_config["population_size"]) * sample_rate if private else None
    global_state = {name: value.detach().clone() for name, value in history.initial_state.items()}
    cost = TrainCost(rounds=len(history.rounds), persistent_storage_bytes=history.storage_bytes())
    for round_record in history.rounds:
        retained_deltas = [delta for client_id, delta in round_record.deltas_by_client.items() if client_id not in forgotten_client_ids]
        for name in global_state:
            aggregate = torch.stack([delta[name] for delta in retained_deltas]).sum(dim=0) if retained_deltas else torch.zeros_like(global_state[name])
            if private:
                if round_record.gaussian_noise is None:
                    raise ValueError("Private cached reconstruction requires stored per-round Gaussian noise.")
                aggregate = (aggregate + round_record.gaussian_noise[name]) / normalizer
            elif retained_deltas:
                aggregate = aggregate / len(retained_deltas)
            global_state[name] = global_state[name] + aggregate
    cost.runtime_seconds = time.perf_counter() - start_time
    model = make_model(model_config, n_features).to(device)
    model.load_state_dict(global_state)
    return model, cost


def federated_eraser(
    *,
    history: FederatedHistory,
    clients: dict[int, ClientDataset],
    forgotten_client_ids: set[int],
    n_features: int,
    federated_config: dict,
    privacy_config: dict,
    calibration_ratio: float = 0.5,
    model_config: dict | None = None,
) -> tuple[nn.Module, TrainCost]:
    """Faithful implementation of the FedEraser unlearning algorithm (Liu et al., 2021).

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
    """
    if history.initial_state is None or not history.rounds:
        raise ValueError("FedEraser requires a non-empty training history with initial state.")

    start_time = time.perf_counter()
    device = get_device()
    private = bool(privacy_config.get("enabled", False))
    sample_rate = float(federated_config["client_sample_rate"])
    # Normalizer for Poisson DP matches retained clients count or privacy population_size
    normalizer = (
        float(privacy_config.get("population_size", len(clients))) * sample_rate
        if private
        else None
    )

    global_state = {name: value.detach().clone() for name, value in history.initial_state.items()}
    cost = TrainCost(rounds=0, persistent_storage_bytes=history.storage_bytes())

    local_epochs = int(federated_config.get("local_epochs", 1))
    cali_epochs = max(1, int(round(local_epochs * calibration_ratio)))
    cali_config = {**federated_config, "local_epochs": cali_epochs}

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
            cost.communicated_bytes += sum(v.numel() * v.element_size() for v in global_state.values())

            # 2. Calibration training on client local data from w'_t
            delta_tilde = _local_update(global_state, client_data, cali_config, n_features, model_config)
            cost.client_updates += 1
            cost.local_examples += len(client_data.y) * cali_epochs

            # 3. Communication: client sends delta_tilde back to server
            cost.communicated_bytes += sum(v.numel() * v.element_size() for v in delta_tilde.values())

            # 4. Calibrate update: scale new direction by historical magnitude
            hist_delta = round_record.deltas_by_client.get(cid)
            if hist_delta is not None:
                h_norm = l2_norm(hist_delta)
                t_norm = l2_norm(delta_tilde)
                factor = float(h_norm / (t_norm + 1e-12))
                cal_delta = {k: v * factor for k, v in delta_tilde.items()}
            else:
                cal_delta = delta_tilde

            calibrated_deltas.append(cal_delta)

        # 5. Aggregate calibrated updates
        for name in global_state:
            if calibrated_deltas:
                aggregate = torch.stack([d[name] for d in calibrated_deltas]).sum(dim=0)
            else:
                aggregate = torch.zeros_like(global_state[name])

            if private:
                if round_record.gaussian_noise is None:
                    raise ValueError("Private FedEraser requires stored per-round Gaussian noise.")
                aggregate = (aggregate + round_record.gaussian_noise[name]) / normalizer
            elif calibrated_deltas:
                aggregate = aggregate / len(calibrated_deltas)

            global_state[name] = global_state[name] + aggregate

    cost.runtime_seconds = time.perf_counter() - start_time
    model = make_model(model_config, n_features).to(device)
    model.load_state_dict(global_state)
    return model, cost


def finetune_retained(
    *,
    full_model: nn.Module,
    clients: dict[int, ClientDataset],
    n_features: int,
    federated_config: dict,
    unlearning_rounds: int,
    seed: int,
    model_config: dict | None = None,
) -> tuple[nn.Module, TrainCost]:
    """A labelled baseline, not a certified unlearning procedure.

    It accesses retained raw data. Its output therefore has a distinct privacy
    exposure from pure post-processing; callers must keep this distinction visible.
    """
    updated_config = deepcopy(federated_config)
    updated_config["rounds"] = int(unlearning_rounds)
    non_private = {"enabled": False}
    return train_federated(
        clients=clients,
        n_features=n_features,
        federated_config=updated_config,
        privacy_config=non_private,
        seed=seed,
        initial_state=clone_state(full_model),
        model_config=model_config,
    )


def flatten_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    device = get_device()
    x = x.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        return model(x).detach().cpu()


def evaluate_loss_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    device = get_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = flatten_logits(model, x)
        loss = _per_example_loss(logits, y.cpu()).mean().item()
    accuracy = _accuracy(logits, y.cpu()).item()
    return {"loss": float(loss), "accuracy": float(accuracy)}


def total_examples(clients: Iterable[ClientDataset]) -> int:
    return sum(len(client.y) for client in clients)
