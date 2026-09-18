# Multiclass Client-DP Utility Protocol

## Purpose

This phase establishes a useful multiclass CIFAR-10 model under finite client-level DP. It is a training-only gate. It does not run unlearning, MUB, or MIA.

## Primary model

`groupnorm_resnet50` is the primary CIFAR-sized ResNet-50 with bottleneck blocks and GroupNorm in every normalization layer. The stem is a 3x3 stride-1 convolution, preserving CIFAR spatial detail. BatchNorm is deliberately avoided because client-local batch statistics complicate federated reproducibility and privacy interpretation.

The existing `groupnorm_resnet18` and `small_groupnorm_cnn` remain available as lower-cost controls.

## Fixed privacy contract

- Adjacency: add/remove one complete client dataset.
- Sampling: independent Poisson client participation.
- Per-client update: L2 clipped.
- Server mechanism: Gaussian noise added to the clipped update sum.
- Normalization: fixed public `q*N`.
- Accountant: repository RDP accountant.
- Public release: final model only.

The target epsilon is used only to calibrate a noise multiplier. The ledger-computed epsilon is authoritative and is saved in every output row.

## Sweep

The initial sweep uses target epsilon values `{8, 12, 16, 20, 24, 32}`. Each row records:

- actual epsilon and delta;
- noise multiplier and clipping norm;
- client count, sample rate, and aggregation normalizer;
- rounds, local epochs, and samples per client;
- test accuracy and loss;
- validity regime;
- runtime and training cost;
- device and PyTorch version.

The default configuration uses 100 clients, 200 samples per client, `q=0.5`, 60 rounds, three local epochs, and three seeds. These are starting points, not final claims.

## Validity regimes

For CIFAR-10:

- `INVALID`: accuracy `< 25%`;
- `WARNING`: accuracy `25%` to `<40%`;
- `VALID`: accuracy `>=40%`.

Near-chance results cannot support a redundancy conclusion. A healthy finite-epsilon point must be found and frozen before running the unlearning comparison.

## Required workflow

1. Run the non-private control.
2. Run the finite-epsilon utility sweep.
3. Inspect `utility_sweep.csv` and `utility_sweep.json`.
4. Choose healthy points without using unlearning results.
5. Freeze the selected model and training configuration.
6. Only then run `M_DP`, `M_FT`, `M_CR`, `M_FE`, and `M_R`.
7. Confirm transition cells with additional seeds.
