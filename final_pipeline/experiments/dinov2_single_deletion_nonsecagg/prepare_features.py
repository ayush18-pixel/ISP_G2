"""Precompute CIFAR-10 DINOv2-S/14 embeddings once for the grid.

Run with CUDA_VISIBLE_DEVICES set to an idle GPU. The source image tensors use
the repository's public CIFAR-10 normalization; FrozenBackboneLinear._encode
performs the same 224x224 resize and unit-sphere normalization as the existing
Tier-2 DINOv2 experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from dp_forgetbench.architectures import FrozenBackboneLinear
from dp_forgetbench.data import _load_cifar10_raw


def _image_tensor(array: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(array.copy(), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    return (x - mean) / std


def _encode(encoder: FrozenBackboneLinear, images: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    parts = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            parts.append(encoder._encode(images[start : start + batch_size].to(device)).cpu())
            done = min(start + batch_size, len(images))
            if done % 5000 == 0 or done == len(images):
                print(f"Encoded {done}/{len(images)}", flush=True)
    return torch.cat(parts).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "results/dinov2_single_deletion_full_metrics/feature_cache.pt"))
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if it is not present under --data-root.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DINOv2 feature extraction requires a visible CUDA GPU.")
    device = torch.device("cuda")
    train, train_y, test, test_y = _load_cifar10_raw(args.data_root, download=args.download)
    digest = hashlib.sha256()
    for array in (train, train_y, test, test_y):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    encoder = FrozenBackboneLinear(backbone_name="dinov2_small", num_classes=10).to(device).eval()
    train_features = _encode(encoder, _image_tensor(train), args.batch_size, device)
    test_features = _encode(encoder, _image_tensor(test), args.batch_size, device)
    payload = {
        "train_features": train_features,
        "train_labels": torch.as_tensor(train_y, dtype=torch.long),
        "test_features": test_features,
        "test_labels": torch.as_tensor(test_y, dtype=torch.long),
        "metadata": {
            "dataset": "CIFAR-10",
            "backbone": "facebookresearch/dinov2_vits14",
            "backbone_parameters_frozen": True,
            "feature_dim": int(train_features.shape[1]),
            "normalization": "CIFAR-10 channel mean/std, as make_cifar10_multiclass_federation",
            "resize": "bilinear 32x32 to 224x224, align_corners=False",
            "feature_normalization": "L2 unit norm",
            "source_image_arrays_sha256": digest.hexdigest(),
            "train_rows": len(train_features),
            "test_rows": len(test_features),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output)
    output.with_suffix(".metadata.json").write_text(json.dumps(payload["metadata"], indent=2) + "\n")
    print(f"Saved {output} ({len(train_features)} train + {len(test_features)} test embeddings)", flush=True)


if __name__ == "__main__":
    main()
