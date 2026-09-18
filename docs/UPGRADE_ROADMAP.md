# DP-ForgetBench — Upgrade Roadmap & Milestones

**Status date:** 2026-09-18
**Scope:** everything we can realistically upgrade in this repository, ordered by
what unblocks what, with milestones and measured evidence for each claim.

All accuracy numbers labelled *(measured)* were produced this session on an
NVIDIA B200 MIG slice through the repository's own `train_federated`, seed
`20260918`. Numbers labelled *(on record)* are quoted from `PROJECT_AUDIT.md`
and were produced before the fixes in Tier 1.

---

## 0. Where we actually are

### 0.1 The blocker we just removed

`make_cifar10_multiclass_federation` hands every client its examples **grouped by
class**, and `_local_update` iterated that tensor in storage order with **no
shuffling**. Measured on the shipped ResNet-50 config: a mean of **2.0 distinct
classes per mini-batch out of 10**, with several batches holding a single class.

This is specific to the `cifar10_multiclass` backend. The `synthetic` and
`cifar10_binary` backends already interleave labels, so **Phase 0 and Phase 1
results are unaffected**; everything from Phase 2 onward was trained through the
degraded loop.

### 0.2 Measured effect of the fix

**Experiment A** — N=20 clients x 250 samples (5,000 images), T=40, E=2, q=0.5,
lr=0.05, bs=32, **no DP at all (eps=inf)**:

| arm | model | change | accuracy |
|---|---|---|---|
| A | ResNet-50 bw16 | shipped recipe | **11.10%** |
| B | ResNet-50 bw16 | + shuffle | **20.05%** |
| C | ResNet-50 bw16 | + shuffle + local momentum 0.9 | 15.80% |
| D | SmallGroupNormCNN | shipped recipe | 35.95% |
| E | SmallGroupNormCNN | + shuffle | **50.15%** |
| F | ResNet-50 bw16 | + shuffle + momentum, T=150 | 31.30% |

**Experiment B** — N=100 x 200 (20,000 images), T=60, E=3, q=0.5, bs=64. This is
the exact configuration `PROJECT_AUDIT.md` used for its validated frontier:

| model | epsilon | shuffle | accuracy | on record (pre-fix) |
|---|---|---|---|---|
| SmallGroupNormCNN | inf | **on** | **55.82%** | 44.35% |
| SmallGroupNormCNN | inf | off | 46.78% | 44.35% |
| SmallGroupNormCNN | 32 | **on** | 39.80% | 41.75% |

Three conclusions, and the third is the one that shapes the roadmap:

1. The `shuffle=off` row (46.78%) reproduces the recorded control (44.35%),
   which validates the comparison.
2. Non-privately the fix is worth **+9.0 points** (46.78 -> 55.82).
3. **At eps=32 the fix buys nothing** (39.80% vs 41.75% on record, within seed
   noise). Under DP the model is *noise-limited, not optimizer-limited*. Raising
   DP accuracy therefore requires raising signal-to-noise, not better training.

### 0.3 The signal-to-noise law that governs every DP number here

Per round the server adds Gaussian noise of norm `sigma * C * sqrt(d)` to a
signal of at most `q*N` clipped updates of norm `C`. So:

```
SNR  =  q*N / (sigma * sqrt(d))          d = trainable parameter count
```

At the validated operating point (N=100, q=0.5, T=60, eps=32, sigma=0.9926):

| model | parameters | SNR | vs small CNN |
|---|---|---|---|
| SmallGroupNormCNN | 60,554 | **0.205** | — |
| GroupNormResNet50 bw16 | 1,484,186 | 0.041 | 5.0x worse |
| GroupNormResNet18 bw64 | 11,173,962 | 0.015 | 13.6x worse |
| GroupNormResNet50 bw64 | 23,520,842 | 0.010 | 19.7x worse |

A config search over N in {100..1000}, q in {0.5, 1.0}, eps in {8, 32} found
**no ResNet-50 setting that reaches usable SNR inside CIFAR-10's 50,000
images**. Even the most generous point (N=1000, q=1.0, eps=32) gives ResNet-50
bw16 an SNR of 0.496 against the small CNN's 2.454.

> **Consequence.** Scaling the *trainable* model up to raise DP metrics moves in
> the wrong direction — `sqrt(d)` always wins. Section 2.5 gives the one way to
> use a large network under DP that does work.

---

## 1. Tier 1 — Correctness (blocking; must land before any new claim)

| # | Item | State |
|---|---|---|
| 1.1 | Per-epoch local batch shuffling, seeded and reproducible | **Done** |
| 1.2 | Batched evaluation in `flatten_logits` | **Done** |
| 1.3 | `local_momentum` knob, default 0 on measured evidence | **Done** |
| 1.4 | Dead code: unused `mub_mia`, stray mid-file `import numpy` | **Done** |
| 1.5 | Re-run every `cifar10_multiclass` result | **Open — blocking** |
| 1.6 | Seed the bootstrap CI RNG | **Open** |
| 1.7 | Record resolved config + hash beside sweep outputs | **Open** |
| 1.8 | Stop re-enabling deterministic algorithms on every call | **Open** |

**1.1 Shuffling.** `_local_update` now reshuffles each epoch using a
`torch.Generator` threaded from `train_federated` and `federated_eraser`.
Verified bit-identical across repeat runs; `federated.shuffle_local_batches:
false` reproduces the old trajectory for archaeology.

**1.2 Batched evaluation.** `flatten_logits` ran one forward pass over the
entire tensor. The CIFAR ResNets keep a stride-1 stem, so layer1 holds an
`(N, 4*base_width, 32, 32)` activation — at `base_width=64` with
`test_samples: 10000` that single tensor is ~10.5 GB, and the 100-client sweep
config would have OOM'd before producing a number. Three callers
(`evaluate_loss_accuracy`, `js_divergence_to_target`, `per_sample_loss`) also
pre-moved the full tensor to the device, defeating the batching; all fixed.

**1.5 Re-run scope.** Invalidated by the shuffle bug: Phase 2 pilots, the
Phase 7 factorial, the utility frontier, the positive controls, and both ResNet
sweeps. `run_positive_controls.py` is a partial exception — its *centralized*
control already used `DataLoader(shuffle=True)`, but its FedAvg control routes
through `train_federated` and is affected.

**1.6 Bootstrap reproducibility.** `compute_bootstrap_ci` calls
`np.random.choice` against the unseeded global NumPy RNG, so the 95% CIs on
every MUB number are not reproducible run-to-run. Accept a `seed` argument and
thread the run seed in.

**1.7 Provenance gap.** `results/utility_sweep_resnet50/` records
`samples_per_client: 100`, but `configs/multiclass_resnet50_local_sweep.yaml`
says `250` — the config drifted after the run. `run.py` saves a `config_sha256`;
`run_multiclass_utility_sweep.py` saves neither the config nor a hash. Add both.

**1.8 Determinism cost.** `set_seed` calls
`torch.use_deterministic_algorithms(True, warn_only=True)` globally on every
invocation, including once per `train_federated`. Set it once at process start.

---

## 2. Tier 2 — Utility upgrades (how to actually raise the metrics)

Ordered by expected gain per unit of effort. Items 2.1–2.4 raise the
**non-private ceiling**; items 2.5–2.7 raise the **DP-constrained** numbers,
which is where the project's claims live.

### 2.1 Data augmentation — largest cheap win, and free under DP
Random crop (pad 4) plus horizontal flip is the standard CIFAR recipe and is
worth roughly 5–10 points on small convnets. Critically it is **fixed public
preprocessing with no learned parameters**, so it costs nothing in the privacy
ledger — the same argument `data.py` already makes for per-channel
normalization. Not currently implemented anywhere in the repo.

### 2.2 Use the full dataset
Current sweeps train on 20,000 of CIFAR-10's 50,000 images. `n_clients *
samples_per_client + audit_samples <= 50,000` allows e.g. N=100 x 495, or
N=500 x 99. More data raises the ceiling *and* raises `N`, which raises SNR —
this item pays twice.

### 2.3 Learning-rate schedule
`train_federated` uses a flat LR for the whole run. Cosine or step decay over
rounds is standard in FedAvg and typically worth a few points.

### 2.4 Server-side momentum (FedAvgM) instead of local momentum
We measured local momentum 0.9 to *hurt* (20.05% -> 15.80%): the optimizer state
is rebuilt from scratch every round, so 2 local epochs never amortize it.
Server-side momentum on the aggregated delta persists across rounds and is the
correct form for FedAvg. It is also DP-compatible — it is post-processing of an
already-noised aggregate.

### 2.5 Frozen public backbone + small private head — the way to use ResNet-50
This is the item that reconciles "we want ResNet-50" with the SNR law.

Under client-level DP the cost is driven by **trainable** `d`, not total model
size. Initialize a ResNet-50 from public pretrained weights, **freeze it**, and
train only a small head under DP. Trainable `d` drops from ~1.5M to ~5–20k,
which *raises* SNR above the small CNN's while giving far stronger features.
This is the standard construction behind published DP-CIFAR results
(Tramèr & Boneh 2021; De et al. 2022).

Requirements to keep the privacy story honest:
- the backbone must be pretrained on **public data disjoint from the federation**
  (ImageNet is the usual choice) and must never be updated on private data;
- the frozen forward pass is public preprocessing, so only the head enters the
  ledger;
- state the pretraining corpus explicitly in `THREAT_MODEL.md` — a reviewer will
  ask whether CIFAR-10 leaked into it.

Expected effect: the largest single jump available at fixed epsilon, and it lets
the paper legitimately say "ResNet-50".

### 2.6 Calibrate `clip_norm` instead of assuming 1.0
`clip_norm: 1.0` appears in every config and has never been measured against the
actual distribution of client update norms. If typical norms are far above 1.0
every update is scaled down hard, wasting signal; if far below, clipping is
inert and the noise is larger than necessary. Add a one-round diagnostic that
reports the median and 90th-percentile update norm per model, and set
`clip_norm` to roughly the median.

### 2.7 Raise N and q deliberately, using the SNR formula
SNR scales linearly in `q*N` and only as `1/sqrt(d)`. Buying SNR through
federation size is far cheaper than through model shrinkage. Use the formula in
0.3 as the design tool: pick the target SNR first, then solve for `(N, q, d)`.

---

## 3. Tier 3 — Scientific upgrades (the project's own Phases 8–11)

These were already on the roadmap and remain correct; they are simply gated on
Tier 1 re-runs.

- **Phase 8 — Privacy audit.** `lira.py` exists but has only ever run with 3–4
  shadow models for plumbing tests. A meaningful confirmation needs **>= 32**.
  Report TPR at low FPR, not just AUC and advantage.
- **Phase 9 — Federated-unlearning baselines.** `federated_eraser` and
  `reconstruct_from_cached_updates` are implemented; neither has been validated
  at a healthy operating point, because no healthy multiclass operating point
  existed until now.
- **Phase 10 — Stress tests.** Sequential, multi-client and influential-client
  deletions; harness exists in `scripts/run_phase10_stress_tests.py`.
- **Phase 11 — Frontier confirmation.** 10 seeds, statistical equivalence
  testing, cost accounting near the redundancy transition.

One structural cost note for planning: `run_experiment` trains
`1 + 1 + (ensemble_size - 1)` full federations per run — **seven** at the default
`retrain_ensemble_size: 5`, plus FedEraser calibration. Budget accordingly, and
consider lowering the ensemble for exploratory cells.

---

## 4. Tier 4 — Engineering & repository

| # | Item | State |
|---|---|---|
| 4.1 | Untrack `src/dp_forgetbench.egg-info/` | **Done** |
| 4.2 | Untrack `colab_bundle/*.zip` (duplicated the unzipped tree) | **Done** |
| 4.3 | Untrack `configs/temp_sweep.yaml` (runtime scratch file) | **Done** |
| 4.4 | Remove `__pycache__/`, extend `.gitignore` | **Done** |
| 4.5 | Consolidate 15 root-level markdown files into `docs/` | **Open — needs a call** |
| 4.6 | De-duplicate results | **Open — low value** |
| 4.7 | Pin the torch version | **Open** |
| 4.8 | De-duplicate eval code in `scripts/` | **Open** |

**4.5 Doc sprawl.** Fifteen markdown files sit at the repository root. Moving
them breaks links in `DOCS.md` and `README.md`, so this needs a deliberate pass
rather than a blind `git mv`.

**4.6 Results.** 38 result groups contain repeat runs, but only **11 are
byte-identical**; the other 27 differ, so they are re-runs after code changes,
not duplicates. At 7.7 MB total, deleting real experimental records to reclaim a
few megabytes is a bad trade. Recommend leaving `results/` alone and instead
adding a `SUPERSEDED.md` marker naming the runs invalidated by item 1.5.

**4.7 Version pin.** `requirements.txt` asks for `torch>=2.6`, but every shipped
result records `2.5.1+cu121`. Pin the version the results were produced with.

**4.8 Duplicated evaluation.** `run_positive_controls.py` carries its own
full-test-set forward pass rather than calling `evaluate_loss_accuracy`, so it
did not receive the Tier 1.2 batching fix. Route scripts through the library.

---

## 5. Milestones

Each milestone states what must be true to enter, what must be true to exit, and
rough GPU cost on one B200 MIG slice. For reference, a single small-CNN run at
N=100 x 200, T=60 is **~54 s**; a ResNet-50 bw16 run at the same scale is a few
minutes.

### M0 — Correctness restored *(entry: now)*
- **Do:** Tier 1.5–1.8. Re-run Phase 2 pilots, Phase 7 factorial, utility
  frontier and positive controls with `shuffle_local_batches: true`.
- **Exit:** every live number in `PROJECT_AUDIT.md` and `UTILITY_FRONTIER.md`
  regenerated post-fix; superseded runs marked; CIs reproducible.
- **Cost:** hours, not days. Phase 7 is the bulk.

### M1 — Non-private ceiling raised
- **Entry:** M0.
- **Do:** Tier 2.1–2.4 (augmentation, full dataset, LR schedule, FedAvgM).
- **Exit:** a documented non-private FedAvg ceiling. Target **>= 65%** for
  SmallGroupNormCNN; we are at 55.82% with none of these four applied.
- **Cost:** low. Each config is under a minute.

### M2 — DP frontier re-mapped
- **Entry:** M1.
- **Do:** Tier 2.6 (clip calibration) and 2.7 (N, q design), then re-sweep
  eps in {2, 4, 8, 16, 32} with the new ceiling and three seeds.
- **Exit:** a re-drawn `VALID` / `WARNING` / `INVALID` frontier, with the
  redundancy window restated. The current window (eps in [8, 32]) is derived
  from pre-fix numbers and will move.
- **Cost:** moderate — 5 epsilons x 3 seeds.

### M3 — Large-model DP via frozen backbone
- **Entry:** M2.
- **Do:** Tier 2.5. Frozen public pretrained ResNet-50 features plus a private
  head; compare head-only `d` and achieved accuracy against the small CNN at
  matched epsilon.
- **Exit:** a defensible answer to "can we use ResNet-50?" — yes, as a frozen
  public feature extractor, with the SNR arithmetic shown. This is the item most
  likely to produce a headline number.
- **Cost:** moderate. Feature extraction is one pass; head training is cheap.

### M4 — Unlearning comparison at a healthy operating point
- **Entry:** M2 (M3 optional).
- **Do:** Phases 8–10. LiRA with >= 32 shadow models; FedEraser and cached
  reconstruction at a frozen healthy config; stress tests.
- **Exit:** MUB with reproducible CIs and a 4-way redundancy classification at
  an operating point that is *not* in utility collapse.
- **Cost:** high — LiRA shadow models dominate.

### M5 — Paper-ready confirmation
- **Entry:** M4.
- **Do:** Phase 11. 10 seeds near the transition, equivalence testing, cost
  accounting.
- **Exit:** the claim in `LIMITATIONS.md` §4 supported end-to-end on post-fix
  numbers.
- **Cost:** highest; run last, change nothing during it.

---

## 6. What will *not* work, and why

Recording this so it is not re-attempted.

- **Scaling the trainable model up to raise DP metrics.** SNR falls as
  `1/sqrt(d)`. Measured: ResNet-50 bw16 reaches 20.05% where the 60k-parameter
  CNN reaches 50.15% on an identical non-private budget, and it is 5x worse on
  SNR at matched epsilon. No setting inside CIFAR-10 rescues it.
- **Local momentum at short local epochs.** Measured 20.05% -> 15.80%. Use
  server-side momentum instead (2.4).
- **Fixing the optimizer to raise *DP* accuracy.** At eps=32 the shuffle fix
  changed nothing (39.80% vs 41.75% on record) because the regime is
  noise-limited. Optimizer work raises the ceiling; only SNR work raises the
  DP numbers.
- **Treating the 54.55% pilot as a DP result.** `PROJECT_AUDIT.md` records its
  budget as **eps = 410.12** — effectively non-private. The correct DP reference
  is 41.75% at eps=32.

---

## 7. Immediate next three actions

1. Re-run Phase 2 and the positive controls with the shuffle fix, and mark the
   superseded runs (M0).
2. Add random-crop and horizontal-flip augmentation, and move to the full
   50,000-image pool (M1) — cheapest points available.
3. Measure the client update-norm distribution and set `clip_norm` from it (2.6)
   before re-sweeping epsilon.
