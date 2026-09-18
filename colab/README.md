# DP-ForgetBench Colab T4 Utility Sweep

This bundle is the first-stage multiclass utility gate. It runs GroupNorm ResNet-50 on CIFAR-10 and sweeps finite client-level-DP epsilon values on a Colab T4 GPU. It does not run unlearning yet.

## Colab setup

1. Open a Colab notebook.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. In the VS Code Colab **Files** sidebar, drag `colab_bundle/dp_forgetbench_colab_utility.zip` into `/content`.
4. Run the notebook bootstrap cell. It discovers the archive under `/content`, extracts it, installs dependencies, and runs the smoke test.
5. Run the commands below from the extracted bundle root if using a fresh notebook.

The notebook intentionally does not use `google.colab.files.upload()`: that widget requires a browser-hosted Colab frontend and does not reliably render through the VS Code extension.


```python
!nvidia-smi
!python -m pip install -q -r colab/requirements-colab.txt
!python -c "import torch, torchvision; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
!python -m pytest -q tests/test_multiclass_cnn.py
```

If Colab reports an incompatible preinstalled torchvision, restart the runtime and use the matching PyTorch/torchvision pair already provided by the runtime. Do not silently replace the CUDA PyTorch installation with a CPU wheel.

## Run the sweep

```python
!python scripts/run_multiclass_utility_sweep.py --config configs/multiclass_resnet50_utility_sweep.yaml
```

## Outputs

```python
import pandas as pd
results = pd.read_csv("results/utility_sweep_resnet50/utility_sweep.csv")
display(results[["epsilon_target", "epsilon_actual", "seed", "test_accuracy", "validity_status"]])
```

Download results:

```python
from google.colab import files
files.download("results/utility_sweep_resnet50/utility_sweep.csv")
files.download("results/utility_sweep_resnet50/utility_sweep.json")
```

## Flow

```text
T4 check -> dependency check -> model smoke test -> non-private/finite-DP utility sweep
         -> inspect actual ledger epsilon and accuracy -> freeze healthy config
         -> later run unlearning comparison
```

The ledger-computed epsilon is authoritative. Do not start FedEraser or MUB experiments from an `INVALID` near-chance model.
