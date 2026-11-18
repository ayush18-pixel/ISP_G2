# DP-ForgetBench: final CIFAR-10 deletion pipeline

This folder packages the **frozen resnet-50 feature + federated linear-head** experiments for a complete deletion of one CIFAR-10 client. It includes executable code, fixed configurations, a dependency snapshot, audited aggregate tables, and one separately bundled sequential-deletion result. Start here to reproduce the two single-deletion grids and understand exactly what their numbers measure.

The single-deletion baseline and GA+retain-KL grids each contain **5 client-partition settings × 4 privacy budgets × 5 seeds = 100 runs**. Both bundled grid audits report 100/100 complete cells. The sequential result has a known fine-tuning evaluation defect, described below; it must not be cited as evidence for that method.

## What this pipeline studies

The classifier consumes cached embeddings from a **frozen resnet-50 ViT-S/14 backbone**. CIFAR-10 images are resized to 224×224 and encoded as unit-normalized 384-dimensional vectors. Federated learning trains only a cosine-normalized 384→10 linear head with **3,850 trainable parameters**. The backbone's pretrained weights never enter the local training updates or the head's DP clipping/noise vector.

The single deletion request removes all training examples belonging to **client 2**. The grid varies Dirichlet class-partition concentration `α ∈ {0.1, 1, 5, 20, 100}` and target privacy budget `ε ∈ {8, 16, 32, 64}`, with seeds `20260918`–`20260922`. Smaller α usually yields more unequal client class mixtures; the realized partition still depends on the seed. Each tuple starts its own original training run.

The baseline grid evaluates no action, retained-data fine-tuning, Cached Reconstruction, FedEraser-style recalibration, and a fresh retained-client retraining reference. A second grid tests exploratory gradient ascent regularized by KL divergence on retained outputs (GA+retain-KL). The latter uses deleted and retained features after the deletion request and has **no DP guarantee for that stage**. The reported nominal ε describes the original client-DP head training mechanism, subject to the ledger's assumptions; it does not certify every post-deletion model or saved metric.

```text
CIFAR-10 images ──frozen resnet-50-S/14──> 384-dimensional feature cache
                                                │
                                      100-client partition
                                                │
                                  40-round client-DP head training
                                                │
                                      delete complete client 2
                                                │
            ┌───────────────┬───────────────────┼───────────────────┐
         no action     retained fine-tune   cached replay    FedEraser-style
            └───────────────┴──────────┬────────┴───────────────────┘
                             fresh retained-client retrain
                                        │
                           accuracy / MIA / JS / cost / MUB
```

The diagram describes the baseline grid. GA+retain-KL is a separate follow-up that retrains a fresh DP original head per tuple because the baseline grid did not preserve reusable checkpoints.

## Contents

```text
final_pipeline/
├── README.md
├── requirements-lock-linux-cuda.txt
├── .gitignore
├── src/dp_forgetbench/                  # core data, FL, DP, and evaluation code
├── experiments/resnet-50_single_deletion_nonsecagg/
│   ├── config.yaml                      # baseline grid
│   ├── prepare_features.py              # download/preprocess/cache frozen features
│   ├── run_grid.py                      # five-method baseline grid
│   ├── summarize_grid.py                # baseline schema/audit/CSV output
│   ├── format_metrics_schema.py
│   ├── ga_retain_kl_config_l100_s20.yaml
│   ├── run_ga_retain_kl_grid.py         # GA+retain-KL candidate grid
│   └── summarize_ga_retain_kl_grid.py
├── results/baseline_nonsecagg/          # baseline aggregate CSVs and audit
├── results/ga_retain_kl/               # GA+KL aggregate CSVs and audit
├── results/sequential_deletion_resnet-50_eps64_alpha20_counts2_4_8_16/
│   └── ...                             # separately bundled three-seed result
└── docs/RESULTS.md                     # protocol and numerical interpretation
```

The 100 baseline and 100 GA+KL **per-run** JSONs, checkpoints, CIFAR-10 files, and embedding caches are not bundled. The sequential aggregate JSON is an intentional exception. Reproducing the grids writes new detailed files locally under the configured `results/` paths. The original research workspace holds the saved per-run source artifacts; this package's aggregate CSVs and audits provide a compact review surface.

### Fixed baseline protocol

| Component | Setting |
|---|---|
| Dataset | CIFAR-10, 10 classes; 100 clients × 200 training examples, 500 audit examples, 10,000 test examples |
| Federation | 40 rounds; five local epochs; batch size 64; SGD, learning rate 0.05; client sampling probability 0.5 |
| Client-level DP | Clip each participating client model update to L2 norm 1; add Gaussian noise to the summed updates; δ=10⁻⁵; calibrate noise multiplier for each ε |
| Deleted data | Entire training dataset of client 2 in each independent run |
| Retained fine-tune | Four further rounds on retained features; accesses data again |
| FedEraser-style | Historical update-direction recalibration, ratio 0.5; accesses retained data and saved per-client updates |
| Security regime | `secure_aggregation: false`, so Cached Reconstruction and FedEraser-style can access saved individual updates |

The privacy accountant is for the configured client-sampled Gaussian mechanism. The runner records mechanism and method-specific scope information; those records should be read before reporting any privacy conclusion. The pairwise-mask Secure Aggregation simulator used in a separate research grid is intentionally disabled here because the two historical-update methods cannot operate on hidden individual updates.

## Environment

Use Python 3.10+ and a CUDA-enabled PyTorch/torchvision pair compatible with the host driver. The small `requirements.txt` intentionally does **not** include PyTorch; `requirements-lock-linux-cuda.txt` is a captured version snapshot, including a particular Linux/CUDA PyTorch build, rather than a portable installer recipe. From the repository root containing this folder:

```bash
cd final_pipeline
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-compatible torch and torchvision pair for this machine first.
python -m pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

The first extraction loads `facebookresearch/resnet-50_vits14` through PyTorch Hub and may download its code and weights. `--download` allows CIFAR-10 download if it is absent. The preparation script requires a visible CUDA GPU. Download time depends on network speed; feature extraction then processes all 50,000 training and 10,000 test images once. Back up or retain the generated cache for subsequent grid runs.

## Reproduce the study

Run from `final_pipeline/`.

### 1. Extract frozen features once

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/resnet-50_single_deletion_nonsecagg/prepare_features.py --download
```

This creates `results/resnet-50_single_deletion_full_metrics/feature_cache.pt` and a small metadata JSON with the source-array digest and feature dimensions. Both baseline and GA+KL configurations point to this cache. They are ignored by Git.

### 2. Run and audit the baseline grid

Single-GPU form:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/resnet-50_single_deletion_nonsecagg/run_grid.py \
  --config experiments/resnet-50_single_deletion_nonsecagg/config.yaml
```

To split the grid across eight **available** GPUs, launch one process per GPU with a distinct worker ID. Each process sees its assigned GPU as local CUDA device 0; all workers use the same `--num-workers`. Create the log directory first, and inspect worker exits before summarizing:

```bash
for worker in {0..7}; do
  CUDA_VISIBLE_DEVICES=$worker OMP_NUM_THREADS=4 \
    python experiments/resnet-50_single_deletion_nonsecagg/run_grid.py \
      --config experiments/resnet-50_single_deletion_nonsecagg/config.yaml \
      --worker-id $worker --num-workers 8 \
      > results/baseline_worker${worker}.log 2>&1 &
done
wait
```

The command above waits for background jobs, but shell `wait` by itself does not produce a per-worker success report. Inspect the eight log tails and any `failures_worker*.json` files, then run the schema formatter and strict coverage audit:

```bash
python experiments/resnet-50_single_deletion_nonsecagg/format_metrics_schema.py
python experiments/resnet-50_single_deletion_nonsecagg/summarize_grid.py
```

The baseline run writes each tuple under `results/resnet-50_single_deletion_nonsecagg_full_metrics/alpha*/eps*/seed*/`. `metrics.json` follows the historical frozen-study schema; `metrics_extended.json` preserves the richer evaluator record. The summarizer writes `grid_audit.json`, `summary_per_seed.csv` and `summary_by_alpha_epsilon.csv` to that same run directory. Confirm `all_runs_complete: true`, 100 unique tuples, and no invalid runs before citing the new aggregate. The six packaged baseline files under `results/baseline_nonsecagg/` are the archived aggregate copy from the completed study, not the destination of a rerun.

### 3. Run and audit GA+retain-KL

Run this after the baseline grid if you want each cell to include a descriptive baseline reference. The candidate's DP replay is independently retrained because the earlier baseline grid did not save checkpoints; treat the historical baseline references as descriptive, not paired comparisons.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/resnet-50_single_deletion_nonsecagg/run_ga_retain_kl_grid.py \
  --config experiments/resnet-50_single_deletion_nonsecagg/ga_retain_kl_config_l100_s20.yaml
python experiments/resnet-50_single_deletion_nonsecagg/summarize_ga_retain_kl_grid.py \
  --config experiments/resnet-50_single_deletion_nonsecagg/ga_retain_kl_config_l100_s20.yaml
```

The GA+KL runner uses the same worker flags for multi-GPU execution. It writes tuple files under `results/ga_retain_kl_single_deletion_nonsecagg_l100_s20/alpha*/eps*/seed*/`; its summarizer writes aggregate CSVs and `grid_audit.json` there. Confirm `complete: true` and 100 valid tuples. `--only-alpha`, `--only-epsilon` and `--only-seed` can restrict either runner to a pilot; a restricted pilot does not satisfy the full-grid audit. Give simultaneous workers distinct IDs and avoid two processes targeting the same tuple/output directory.

The scripts resolve configured result paths relative to `final_pipeline/`. Run commands from this directory, keep source and config snapshots with the outputs, and use a fresh result root when repeating a full study so completed artifacts are not mistaken for a new run. OS-seeded client sampling/noise means a rerun should be interpreted as a new stochastic realization; the original five-seed numeric values are not guaranteed to repeat bit-for-bit.

## How to read the metrics

| Field or population | Meaning |
|---|---|
| `test_accuracy` | Correct classifications on the held-out CIFAR-10 test images; generalization utility |
| `forgotten_accuracy` | Correct classifications on the deleted client's former **training** examples; behavior diagnostic, not a forgetting proof |
| `retained_accuracy` | Correct classifications on training examples of the clients still presnet |
| MIA AUC / advantage | Performance of the specified member-versus-unseen attack, including its calibration and held-out evaluation protocol; AUC near 0.5 is inconclusive for that attack |
| JS divergence / prediction agreement | Output similarity to a particular retrained reference; low distance does not establish deletion |
| MUB | Marginal change toward retraining relative to no action; interpret each metric separately and with retrain variability |
| Runtime, storage, communication | Method-specific measured or estimated costs; legacy byte counters can have different directional scope |

The bundled `summary_by_alpha_epsilon.csv` files contain five-seed means and descriptive 95% Student-t intervals for each pair. They do not implement a multiple-comparison-adjusted equivalence test. The historical `mia_lira` name in compatibility-formatted metrics refers to the project's loss-based MIA fields; do not cite it as a validated LiRA result.

## Current aggregate findings

These are results from the included completed run artifacts, not a promise that a rerun will produce identical GPU floating-point outputs.

- The baseline's strongest listed test-accuracy cell was `alpha=100, epsilon=64`: DP-only **88.31%**, retained fine-tuning **89.03%**, Cached Reconstruction **88.23%**, FedEraser-style **87.73%**, target retraining **87.80%** (five seeds).
- Across the 20 alpha/epsilon cells, equal-weight means from the completed baseline grid were: DP-only **75.86%**, fine-tuning **78.51%**, Cached Reconstruction **75.05%**, FedEraser-style **74.56%**, target retraining **75.42%** test accuracy.
- Across the GA+KL grid, the paired candidate-minus-fresh-DP-replay changes averaged over cells were **−0.74 percentage points test accuracy**, **−3.91 points forgotten-client accuracy**, and **−0.68 points retained-data accuracy**. The largest forgotten-accuracy decrease occurred at `alpha=0.1` (about **−14.58 points** averaged across epsilons); this is exploratory and does not establish certified or complete forgetting.
- The baseline audit reports 100/100 valid tuples with both historical-update methods evaluated. The GA+KL audit reports 100/100 completed tuples. See [the detailed results note](docs/RESULTS.md), [the baseline audit](results/baseline_nonsecagg/grid_audit.json), and [the GA+KL audit](results/ga_retain_kl/grid_audit.json) for the full scope.

### Bundled sequential-deletion result: use with caution

The extra JSON under `results/sequential_deletion_resnet-50_eps64_alpha20_counts2_4_8_16/` contains a **three-seed CIFAR-10 frozen-resnet-50-head** study at α=20 and ε=64, evaluating independent deletion sizes 2, 4, 8 and 16 from each seed's original federation. The directory name contains `resnet-50`, but the result's `model_type`/`backbone_name` fields say `resnet-50`/`resnet-50_small`. The bundled `run_spec.json` also names a different intended output path; use the JSON metadata and preserve this provenance discrepancy.

| Deleted clients | No-action test | Exact-retrain test | Cached test | FedEraser-style test |
|---:|---:|---:|---:|---:|
| 2 | 90.62% | 90.25% | 90.58% | 90.49% |
| 4 | 90.62% | 90.27% | 90.29% | 90.32% |
| 8 | 90.62% | 90.11% | 90.17% | 90.24% |
| 16 | 90.62% | 89.93% | 89.50% | 89.74% |

**Known invalid fields:** the source sequential runner called `finetune_retained(...)` but discarded its returned trained model and evaluated the unchanged copied head. Therefore its fine-tuning accuracy, fine-tuning-to-retrain similarity, fine-tuning MUB, and fine-tuning-based pre/post attack interpretation are invalid as fine-tuning measurements. The displayed four-method table excludes those fields. This standalone package bundles the saved sequential result but **does not include the sequential runner** needed to regenerate it; obtain the corrected source and rerun before presneting a fine-tuning comparison. The result also records `secure_rng: false`; its noise uses an OS-entropy-seeded Torch PRNG, so it is exploratory rather than secure-RNG-audited DP evidence.

## Important interpretation and release limits

- Accuracy on the deleted client's examples is only a utility/behavior diagnostic. Lower accuracy alone does not prove unlearning.
- Membership-inference metrics measure one specified attack and population; an AUC near 0.5 is not a general privacy guarantee.
- Secure Aggregation is disabled in this experiment because the historical-update methods need individual client updates. The server can therefore inspect client updates.
- Cached Reconstruction and FedEraser-style recalibration reuse historical updates/noise. The original model's DP ledger alone does not certify joint release of original and transformed models.
- GA+retain-KL uses deleted examples and retained data after the deletion request and has no DP guarantee for that post-deletion stage.
- Frozen resnet-50 embeddings are centrally cached. Their extraction is outside the head's DP training ledger.
- The aggregate tables are descriptive across five seeds. They do not replace a threat-model-specific privacy analysis or a confirmatory statistical study.

For exact per-run results and logs, use the original research workspace. This package keeps only GitHub-friendly aggregate tables and audit files by default.

## Integrity manifest

`SHA256SUMS` records hashes for the bundled source, configs, documentation, and aggregate result files. After a local README edit, its README hash must be refreshed. The package's manifest has been updated for this version. Verify from this directory with:

```bash
sha256sum --check SHA256SUMS
```

The checksum verifies byte integrity of the bundled files. It does not certify the experiment's privacy guarantee, reproduce the training environment, or resolve the sequential provenance and fine-tuning issues described above.
