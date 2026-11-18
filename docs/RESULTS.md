# Results and protocol notes

## Completed studies included here

### Five-method baseline

- Dataset: CIFAR-10, 20,000 client-training examples (100 clients × 200 examples), 500 held-out audit examples, and 10,000 test examples.
- Feature extractor: frozen resnet-50 ViT-S/14, 384-dimensional unit-normalized embeddings.
- Trainable head: linear 384-to-10 classifier (3,850 parameters).
- Federation: 100 clients, 40 rounds, 5 local epochs, local batch size 64, client sampling rate 0.5.
- DP settings: client-update clipping norm 1, Poisson client sampling, δ=10⁻⁵; Gaussian noise calibrated per target ε. Secure Aggregation is disabled.
- Deletion: delete all examples from client 2; each alpha/epsilon/seed tuple starts as an independent run.
- Grid: five alpha values × four epsilon values × five seeds = 100 tuples.

The main metrics are `test`, `forgotten_client`, `retained_data`, membership-inference diagnostics, retraining similarity/MUB, runtime, communication, storage, and method-specific privacy/accounting metadata. The complete per-run JSONs remain in the source workspace and are not copied into this GitHub bundle.

### Exploratory GA+retain-KL

The candidate starts from a newly trained DP head for each tuple. It maximizes cross-entropy on the deleted client's features while regularizing retained-set predictions toward the pre-deletion model with KL divergence. Configuration: 20 update steps, learning rate 0.05, KL weight 100, retained batch size 1,024. Five seeds are used in each of the same 20 alpha/epsilon cells (100 tuples total).

The paired comparison is candidate versus its same-cell fresh DP replay, so that difference is paired at the seed level. It is not paired to the earlier baseline grid: that grid did not save checkpoints and DP sampling/noise are OS-seeded on replay. Historical method numbers attached to GA+KL outputs are context only.

## Headline summary

| Method/result | Test accuracy |
|---|---:|
| Baseline DP-only, equal-weight mean over 20 alpha/epsilon cells | 75.86% |
| Retained fine-tuning, same summary | 78.51% |
| Cached Reconstruction, same summary | 75.05% |
| FedEraser-style, same summary | 74.56% |
| Target retraining, same summary | 75.42% |
| Best baseline cell, alpha=100, epsilon=64: DP-only | 88.31% |
| Same cell: retained fine-tuning | 89.03% |
| Same cell: Cached Reconstruction | 88.23% |
| Same cell: FedEraser-style | 87.73% |
| Same cell: target retraining | 87.80% |

Across the GA+KL grid, mean paired deltas (candidate minus its DP replay, equal-weighted across the 20 alpha/epsilon cells) were:

| Metric | Mean paired change |
|---|---:|
| Test accuracy | −0.74 percentage points |
| Forgotten-client accuracy | −3.91 percentage points |
| Retained-data accuracy | −0.68 percentage points |

At alpha 0.1, the forgotten-client accuracy change averaged −14.58 points across the four epsilon settings. At alpha ≥1, the averaged decreases were much smaller (approximately 0.8–2.2 points depending on alpha). The large alpha-0.1 effect is a useful follow-up signal, but it is not enough to claim reliable forgetting; include uncertainty intervals and attack results in any scientific claim.

## Included artifacts

- `../results/baseline_nonsecagg/summary_by_alpha_epsilon.csv`: method-level means and seed-based confidence intervals for each alpha/epsilon pair.
- `../results/baseline_nonsecagg/summary_per_seed.csv`: five seed rows per baseline cell, with aggregate performance and privacy diagnostics.
- `../results/baseline_nonsecagg/grid_audit.json`: 100/100 tuple coverage and audit status.
- `../results/ga_retain_kl/summary_by_alpha_epsilon.csv`: DP replay, GA+KL, historical reference, and paired accuracy summaries.
- `../results/ga_retain_kl/summary_per_seed.csv`: seed-level candidate/replay outcomes.
- `../results/ga_retain_kl/grid_audit.json`: GA+KL grid completeness and caveat.

The tables contain aggregate diagnostics rather than raw features, predictions, individual client updates, checkpoints, or full per-run JSONs. The original research results directory is the source of record for those artifacts.

## Additional sequential-deletion artifact

The complete requested bundle is copied at `../results/sequential_deletion_resnet-50_eps64_alpha20_counts2_4_8_16/`, including its run specification, aggregate JSON metrics, and logs. Its source `run_spec.json` identifies the backbone as `resnet-50_small` and points its `output` field to a differently named source-results directory. The artifact folder name contains `resnet-50`, but that name alone does not establish a ResNet architecture; inspect the recorded `model_type`/`backbone` fields before describing it. Files were copied without changing their contents.
