# DP-ForgetBench

## Project Purpose

DP-ForgetBench is a research benchmark for studying the relationship between
client-level differential privacy and federated unlearning.

The central question is:

> When a federated learning model was already trained with client-level
> differential privacy, does an explicit client-unlearning algorithm still add
> measurable value after a deletion request?

This project does not assume that differential privacy is the same thing as
deletion. It also does not assume that unlearning is always unnecessary. The
project instead treats both ideas as competing scientific hypotheses and tests
them against an exact retraining reference.

The benchmark asks whether, for a given privacy budget, model quality, data
heterogeneity, and deletion size, doing nothing to a differentially private
model is statistically indistinguishable from retraining the model without the
deleted client.

The goal is to map the boundary where explicit unlearning is useful, redundant,
or harmful.

## Why This Question Matters

Federated learning trains a model across many clients without centralizing raw
client data. This makes it attractive for privacy-sensitive domains, but it
creates a difficult deletion problem.

If a client later asks to be removed, the trained model may still contain some
statistical influence from that client's local data. One obvious answer is exact
retraining: train a new model from scratch using all retained clients and
excluding the deleted client. Exact retraining is conceptually clean, but it is
expensive.

Federated unlearning algorithms try to approximate exact retraining more cheaply.
They may use historical client updates, retained-client fine-tuning, server-side
reconstruction, or interactive calibration. These methods can be useful, but
they often introduce large storage costs, require extra communication, and may
query retained clients again after deletion.

Differential privacy changes the picture. If training already included
client-level differential privacy, then the final released model has a formal
bound on how much any single client's inclusion can affect the output
distribution. That does not automatically mean the client has been deleted, but
it raises a serious empirical question:

If the deleted client's influence is already smaller than normal retraining
variation, does explicit unlearning buy anything measurable?

DP-ForgetBench is built to answer that question experimentally and carefully.

## What We Compare

Every meaningful experiment is organized around three conceptual model families.

### 1. DP-only / No Action

This model is trained on the full client population with client-level
differential privacy. After a deletion request arrives, the model is left
unchanged.

This is the simplest possible response. It has no unlearning compute, no extra
client communication, no historical-update storage, and no additional access to
raw retained data.

The scientific question is whether this no-action model is already close enough
to the exact retrain reference under the chosen privacy regime.

### 2. Explicit Unlearning

These methods modify or reconstruct a model after deletion. The benchmark has
included several unlearning baselines:

- Retained-data fine-tuning.
- Cached update reconstruction.
- FedEraser-style historical reconstruction and calibration.
- Exact retraining as a gold-standard reference, not as a practical shortcut.

These baselines are evaluated by how close they get to exact retraining, how
much membership leakage they reduce, and how much compute, storage, and client
interaction they require.

### 3. Exact Retrain Reference

Exact retraining is the clean reference: train from scratch on the retained
client population after removing the deleted client.

A key design decision in this project is that exact retraining is not treated as
a single model. One retrain can be noisy because random initialization, client
sampling, and stochastic optimization all create natural variation. Therefore the
project uses a retrain ensemble across multiple seeds. The ensemble gives a
distribution of plausible retrained models.

This matters because unlearning should not be judged against a single lucky or
unlucky retrain. It should be judged against the natural variability of retrain
itself.

## Core Scientific Idea

The benchmark is built around this principle:

> A deletion effect is meaningful only if it is larger than ordinary retraining
> noise.

If deleting a client changes the model less than changing the random seed during
retraining, then a claim that an unlearning method improves deletion quality is
weak. The method may simply be chasing randomness.

To formalize this, the project compares:

- The distance between the full model and retrain models.
- The distance between unlearned models and retrain models.
- The natural spread inside the retrain ensemble itself.

When the DP-only model falls inside the retrain ensemble's variability band,
explicit unlearning is classified as redundant for that setting.

When the unlearned model is clearly closer to retrain than DP-only, unlearning is
classified as beneficial.

When unlearning moves away from retrain or damages utility, it is harmful.

When the evidence is too noisy, the result is inconclusive.

## Marginal Unlearning Benefit

The project uses Marginal Unlearning Benefit, or MUB, to measure whether
explicit unlearning improves over DP-only.

MUB is not one metric. It is a framework for asking three related questions.

### Utility MUB

Does explicit unlearning produce test accuracy closer to the retrain ensemble
than DP-only?

This matters because an unlearning method that destroys model utility is not a
good deletion solution, even if it reduces membership signals.

### Alignment MUB

Does explicit unlearning make the model's predictions behave more like exact
retraining?

The project uses prediction-distribution distances such as Jensen-Shannon
divergence to measure behavioral alignment. This is stricter than accuracy
alone. Two models can have similar accuracy while making different predictions.

### Privacy MUB

Does explicit unlearning reduce membership inference attack advantage on the
forgotten population beyond what DP-only already provides?

This matters because client deletion is partly about reducing remaining evidence
that the forgotten client participated in training.

## Validity Gates

A major lesson from the project is that unlearning claims are meaningless if the
model never learned the task.

For CIFAR-10, random guessing is about 10 percent accuracy. If a private model is
near random chance, a low membership inference signal does not prove successful
unlearning. It may only prove the model is useless.

Therefore the benchmark uses validity gates:

- INVALID: accuracy below 25 percent.
- WARNING: accuracy from 25 percent to below 40 percent.
- VALID: accuracy at or above 40 percent.

Only healthy or at least marginally useful models should be used for serious
claims about the redundancy frontier.

## Privacy Contract

The primary privacy contract is client-level central differential privacy.

That means:

- The protected unit is one complete client, not one image or one row.
- Client updates are clipped before aggregation.
- Gaussian noise is added at the server.
- Client sampling is accounted for using a Poisson-subsampled Gaussian mechanism.
- The released object is the final model.
- The privacy ledger records epsilon, delta, sampling rate, clipping norm, noise
  multiplier, population size, and release count.

This is important because client deletion is also a client-level request. The
privacy unit and the deletion unit must match.

Example-level DP and client-level DP answer different questions. This project is
mainly about client-level removal in federated learning.

## Why FedEraser Is Treated Carefully

FedEraser-style methods are important because they represent a serious
historical-update-based approach to federated unlearning.

However, the project treats FedEraser carefully because it is not simply free
post-processing of a DP model.

FedEraser-style calibration can require retained clients to receive intermediate
models and compute new updates on their raw local data. That means the method
performs fresh data-dependent access after the original model was trained.

This has two consequences:

- It can require additional privacy accounting.
- It can be much more expensive than DP-only no action.

The benchmark therefore compares FedEraser not only by accuracy and attack
metrics, but also by storage, communication, and privacy category.

## What Has Been Done So Far

### Phase 0: Synthetic Verification

The project began with a small synthetic federated setting. This phase was used
to verify the basic privacy accounting, deletion handling, deterministic
experiments, and end-to-end result writing.

This phase established that the benchmark could produce reproducible metrics,
privacy ledgers, deletion manifests, and experiment metadata.

### Phase 1: Real CIFAR-10 Binary Pipeline

The next step moved from synthetic data to CIFAR-10 in a CPU-feasible binary
classification setting.

This phase validated:

- Public dataset loading.
- Deterministic client partitioning.
- Client deletion manifests.
- Privacy accounting on real data.
- Basic membership inference probes.
- Exact retraining references.

The goal was not high image accuracy. The goal was to prove that the real-data
pipeline behaved correctly before moving to harder multiclass CNN experiments.

### Phase 2: CIFAR-10 Multiclass CNN

The benchmark then moved to 10-class CIFAR-10 with image tensors and convolutional
models.

Early experiments collapsed near chance. The main architecture issue was that an
early CNN used aggressive global pooling that destroyed too much spatial
information. Replacing that spatial bottleneck with a flattened classifier head
allowed the model to learn.

After this correction, the small GroupNorm CNN became a useful control model for
studying convergence, privacy noise, and the utility frontier.

### Positive Controls

Positive controls were added to prove that the data and model could learn before
making privacy or unlearning claims.

The important positive-control logic was:

- If centralized training cannot learn, the architecture or data pipeline is
  broken.
- If non-private FedAvg cannot learn, the federated optimization setup is broken.
- If deleting a deliberately influential client causes no measurable change, the
  unlearning evaluation is not sensitive enough.

These controls helped separate real privacy effects from simple training failure.

### Retrain Ensembles

The project moved from single retrain references to retrain ensembles.

This was a major scientific upgrade. It prevents over-interpreting the distance
to one retrained model. The ensemble gives a realistic band of retraining
variation, which is then used to judge whether DP-only or unlearning is actually
close to retrain.

### Marginal Unlearning Benefit

The project formalized MUB so unlearning benefit can be measured relative to
DP-only.

This changed the question from:

"Does unlearning change the model?"

to:

"Does unlearning improve over doing nothing, relative to exact retraining?"

That is the central benchmark question.

### Full Baseline Suite

The benchmark now includes:

- DP-only no action.
- Retained-data fine-tuning.
- Cached update reconstruction.
- FedEraser-style reconstruction and calibration.
- Exact retrain ensembles.

This gives a broad comparison across utility, alignment, privacy leakage,
storage, communication, and runtime.

### Utility Frontier Work

The project has explored the boundary between useful DP models and collapsed DP
models.

The observed regimes are:

- Low epsilon can collapse CIFAR-10 utility.
- Intermediate epsilon can produce marginal but useful models.
- High epsilon or non-private training gives stronger utility but weaker privacy.

The key scientific challenge is to find settings where the model is useful and
DP is strong enough that client influence is masked.

### ResNet-50 Upgrade

The latest upgrade adds a stronger CIFAR-sized GroupNorm ResNet-50 model.

The reason is simple: if DP noise hurts learning, a stronger architecture may
recover more useful accuracy under the same privacy accounting. ResNet-50 is now
available as `groupnorm_resnet50` and has its own utility sweep configs.

The model uses GroupNorm instead of BatchNorm because BatchNorm depends on batch
statistics, which are awkward in federated and privacy-sensitive settings.
GroupNorm keeps normalization independent of client-local batch statistics.

The project still keeps ResNet-18 and the small CNN as lower-cost controls.

## Current Status

At this point, the repository has:

- A deterministic experiment framework.
- Synthetic and real-data federation backends.
- Client-level DP accounting.
- Deletion manifests.
- Exact retrain references.
- Retrain ensembles.
- Multiple unlearning baselines.
- Membership inference audits.
- MUB-based redundancy classification.
- Validity gates for avoiding false claims from collapsed models.
- A GroupNorm ResNet-50 path for stronger local CIFAR-10 experiments.

The immediate next step is to run the ResNet-50 utility smoke and then the full
ResNet-50 DP utility sweep locally. The smoke verifies that the real CIFAR-10
loader and training path work on this machine. The full sweep is the meaningful
accuracy experiment.

## Local Testing Workflow

This repository is currently set up for local testing, not Colab-first testing.

From the repository root:

```powershell
python -m pytest -q -p no:cacheprovider
```

The `-p no:cacheprovider` flag avoids local Windows permission issues with
pytest cache directories.

To run a tiny local ResNet-50 federated smoke without touching CIFAR-10:

```powershell
$env:PYTHONPATH = "src"
python -c "import torch; from dp_forgetbench.data import ClientDataset; from dp_forgetbench.federated import train_federated, evaluate_loss_accuracy; clients={i: ClientDataset(i, torch.randn(4,3,32,32), torch.randint(0,10,(4,))) for i in range(3)}; model,cost=train_federated(clients=clients,n_features=None,federated_config={'rounds':1,'local_epochs':1,'local_batch_size':2,'learning_rate':0.01,'client_sample_rate':1.0},privacy_config={'enabled':False},seed=1,model_config={'name':'groupnorm_resnet50','num_classes':10,'base_width':8}); print(type(model).__name__, cost.as_dict()); print(evaluate_loss_accuracy(model, torch.randn(5,3,32,32), torch.randint(0,10,(5,))))"
```

This does not measure real accuracy. It only proves that the ResNet-50 model can
instantiate, train through the federated loop, and evaluate locally.

## Local ResNet-50 Training Workflow

For a quick real CIFAR-10 smoke test:

```powershell
python scripts/run_multiclass_utility_sweep.py --config configs/multiclass_resnet50_t4_smoke.yaml --output-dir results/local_resnet50_smoke
```

For the full ResNet-50 DP utility sweep:

```powershell
python scripts/run_multiclass_utility_sweep.py --config configs/multiclass_resnet50_utility_sweep.yaml
```

The full sweep writes:

```text
results/utility_sweep_resnet50/utility_sweep.csv
results/utility_sweep_resnet50/utility_sweep.json
```

The most important fields are:

- `epsilon_target`
- `epsilon_actual`
- `noise_multiplier`
- `rounds`
- `local_epochs`
- `test_accuracy`
- `validity_status`
- `runtime_seconds`

If the run is marked INVALID, it should not be used for unlearning claims.

## Important Local Data Note

The real CIFAR-10 smoke and full sweep need a readable and writable CIFAR-10 data
directory.

On this machine, the existing `data/cifar-10-batches-py` directory may have
Windows permission problems. If the real-data smoke fails with `PermissionError`,
the code path is not necessarily broken. The local dataset directory needs to be
fixed, deleted and re-extracted, or replaced with a fresh readable copy.

The in-memory ResNet-50 smoke avoids this dataset issue and is useful for testing
the model and federated training path.

## How To Use k.ipynb

The notebook `k.ipynb` is now local-only.

Run the cells in this order:

1. Verify local Python, PyTorch, and CUDA.
2. Check local project files and data-directory accessibility.
3. Set `PYTHONPATH` to the local `src` directory.
4. Run the full local test suite.
5. Run an in-memory ResNet-50 federated training smoke.
6. Optionally run the real CIFAR-10 ResNet-50 smoke.
7. Run the full local ResNet-50 DP utility sweep.
8. Inspect the resulting accuracy table.

Cells 1 to 5 should run even if the local CIFAR-10 directory is broken. Cells 6
and 7 require real CIFAR-10 access.

## Interpretation Of Future ResNet-50 Results

The ResNet-50 sweep should be interpreted using the same scientific rules as the
earlier CNN experiments.

High accuracy with weak privacy is not enough. Strong privacy with random-chance
accuracy is also not enough.

The useful regime is where:

- Test accuracy is meaningfully above chance.
- The privacy ledger reports the intended epsilon.
- The DP-only model can be compared against an exact retrain ensemble.
- Any explicit unlearning method is judged by marginal benefit over DP-only.

Only then can we say whether unlearning is beneficial, redundant, harmful, or
inconclusive.

## Bottom Line

DP-ForgetBench is not just training a model. It is building a careful scientific
test for a specific question:

> Does explicit client unlearning still matter once the model has already been
> trained with client-level differential privacy?

The answer is expected to depend on the regime. In non-private or weakly private
settings, unlearning can matter. In sufficiently private and still useful
settings, client influence may be masked enough that explicit unlearning becomes
redundant. In over-noised settings, the model collapses and no unlearning claim
is valid.

The project so far has built the machinery to distinguish those regimes instead
of guessing.
