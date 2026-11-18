# DP-ForgetBench: final reproducible pipeline

A compact, standalone package for the CIFAR-10 **single-client deletion** experiments. It contains the runnable code, fixed experiment configurations, dependency snapshot, and aggregate results/audits. The rest of the research workspace is intentionally not copied here.

## What this pipeline studies

The classifier is trained on cached embeddings from a **frozen DINOv2 ViT-S/14 backbone** (384-dimensional features) using a 3,850-parameter linear head. Federated learning trains the head across 100 clients. The single deletion request removes all data belonging to client 2. The alpha/epsilon grid is `alpha ∈ {0.1, 1, 5, 20, 100}`, `epsilon ∈ {8, 16, 32, 64}`, with five seeds per combination.

The baseline grid evaluates DP-only/no action, retained-data fine-tuning, Cached Reconstruction, FedEraser-style recalibration, and target retraining. A second grid evaluates exploratory gradient ascent with retained-logit KL regularization (GA+retain-KL). That second stage reads deleted and retained data and is **non-private**; the original DP ledger does not cover it. This is a frozen-feature head study, not full-backbone training or unlearning.

## Contents

```text
final_pipeline/
├── README.md
├── requirements-lock-linux-cuda.txt
├── .gitignore
├── src/dp_forgetbench/                  # core data, FL, DP, and evaluation code
├── experiments/dinov2_single_deletion_nonsecagg/
│   ├── config.yaml                      # baseline grid
│   ├── prepare_features.py              # download/preprocess/cache frozen features
│   ├── run_grid.py                      # five-method baseline grid
│   ├── summarize_grid.py                # baseline schema/audit/CSV output
│   ├── format_metrics_schema.py
│   ├── ga_retain_kl_config_l100_s20.yaml
│   ├── run_ga_retain_kl_grid.py         # GA+retain-KL candidate grid
│   └── summarize_ga_retain_kl_grid.py
├── results/                             # aggregate CSVs and audit receipts only
└── docs/                                # protocol and result interpretation
```

Per-run JSONs, model checkpoints, CIFAR-10 files, and embedding caches are not bundled. The detailed per-run outputs are data-dependent research artifacts and should be reviewed before any public release. Running the scripts writes those files locally under `results/`.

## Environment

Use Python 3.10+ and a CUDA-enabled PyTorch install compatible with the host's driver. `requirements-lock-linux-cuda.txt` records the tested Linux/CUDA dependency versions; exact CUDA wheels can be platform-specific. Install a matching PyTorch/torchvision pair first if the listed CUDA build is not compatible with your machine, then install the remaining packages:

```bash
cd final_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA-compatible torch and torchvision pair for your host first.
python -m pip install -r requirements.txt
```

The scripts load DINOv2-S/14 through PyTorch Hub during feature extraction, so the first extraction needs network access to fetch the model weights. CIFAR-10 can also be downloaded by the preparation script. Neither download is included in this folder.

## Reproduce the study

Run from `final_pipeline/`.

### 1. Extract frozen features once

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/dinov2_single_deletion_nonsecagg/prepare_features.py --download
```

This creates `results/dinov2_single_deletion_full_metrics/feature_cache.pt`. It is a local cache and is ignored by Git.

### 2. Run and audit the baseline grid

Single-GPU form:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/dinov2_single_deletion_nonsecagg/run_grid.py \
  --config experiments/dinov2_single_deletion_nonsecagg/config.yaml
```

To split the grid across eight GPUs, launch one process per GPU with a distinct worker ID (each process must use the same `--num-workers`):

```bash
for worker in {0..7}; do
  CUDA_VISIBLE_DEVICES=$worker OMP_NUM_THREADS=4 \
    python experiments/dinov2_single_deletion_nonsecagg/run_grid.py \
      --config experiments/dinov2_single_deletion_nonsecagg/config.yaml \
      --worker-id $worker --num-workers 8 \
      > results/baseline_worker${worker}.log 2>&1 &
done
wait
```

After all workers finish successfully:

```bash
python experiments/dinov2_single_deletion_nonsecagg/format_metrics_schema.py
python experiments/dinov2_single_deletion_nonsecagg/summarize_grid.py
```

### 3. Run and audit GA+retain-KL

Run this after the baseline grid if you want each cell to include a descriptive baseline reference. The candidate's DP replay is independently retrained because the earlier baseline grid did not save checkpoints; treat the historical baseline references as descriptive, not paired comparisons.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/dinov2_single_deletion_nonsecagg/run_ga_retain_kl_grid.py \
  --config experiments/dinov2_single_deletion_nonsecagg/ga_retain_kl_config_l100_s20.yaml
python experiments/dinov2_single_deletion_nonsecagg/summarize_ga_retain_kl_grid.py \
  --config experiments/dinov2_single_deletion_nonsecagg/ga_retain_kl_config_l100_s20.yaml
```

The worker flags shown above also apply to the GA+KL runner for multi-GPU execution. Do not run two processes on the same worker ID.

## Current aggregate findings

These are results from the included completed run artifacts, not a promise that a rerun will produce identical GPU floating-point outputs.

- The baseline's strongest listed test-accuracy cell was `alpha=100, epsilon=64`: DP-only **88.31%**, retained fine-tuning **89.03%**, Cached Reconstruction **88.23%**, FedEraser-style **87.73%**, target retraining **87.80%** (five seeds).
- Across the 20 alpha/epsilon cells, equal-weight means from the completed baseline grid were: DP-only **75.86%**, fine-tuning **78.51%**, Cached Reconstruction **75.05%**, FedEraser-style **74.56%**, target retraining **75.42%** test accuracy.
- Across the GA+KL grid, the paired candidate-minus-fresh-DP-replay changes averaged over cells were **−0.74 percentage points test accuracy**, **−3.91 points forgotten-client accuracy**, and **−0.68 points retained-data accuracy**. The largest forgotten-accuracy decrease occurred at `alpha=0.1` (about **−14.58 points** averaged across epsilons); this is exploratory and does not establish certified or complete forgetting.
- The baseline audit reports 100/100 valid tuples. The GA+KL audit reports 100/100 completed tuples. See [`docs/RESULTS.md`](docs/RESULTS.md) for interpretation and limitations.

## Important interpretation and release limits

- Accuracy on the deleted client's examples is only a utility/behavior diagnostic. Lower accuracy alone does not prove unlearning.
- Membership-inference metrics measure one specified attack and population; an AUC near 0.5 is not a general privacy guarantee.
- Secure Aggregation is disabled in this experiment because the historical-update methods need individual client updates. The server can therefore inspect client updates.
- Cached Reconstruction and FedEraser-style recalibration reuse historical updates/noise. The original model's DP ledger alone does not certify joint release of original and transformed models.
- GA+retain-KL uses deleted examples and retained data after the deletion request and has no DP guarantee for that post-deletion stage.
- Frozen DINOv2 embeddings are centrally cached. Their extraction is outside the head's DP training ledger.
- The aggregate tables are descriptive across five seeds. They do not replace a threat-model-specific privacy analysis or a confirmatory statistical study.

For exact per-run results and logs, use the original research workspace. This package keeps only GitHub-friendly aggregate tables and audit files by default.

## Integrity manifest

`SHA256SUMS` records hashes for the bundled source, configs, documentation, and aggregate result files. Verify them from this directory with:

```bash
sha256sum --check SHA256SUMS
```

The explicitly requested sequential-deletion result bundle is included at `results/sequential_deletion_resnet64_eps64_alpha20_counts2_4_8_16/`. Its source run spec says `model_type: dinov2` and `backbone: dinov2_small`; the folder name and recorded model metadata differ, so use the metadata when describing this result.
