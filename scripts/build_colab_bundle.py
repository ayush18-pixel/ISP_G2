"""Build a minimal Colab utility-sweep bundle and zip archive."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "colab_bundle" / "dp_forgetbench_colab_utility"


def main() -> None:
    if BUNDLE.parent.exists():
        shutil.rmtree(BUNDLE.parent)
    (BUNDLE / "src").mkdir(parents=True)
    (BUNDLE / "scripts").mkdir(parents=True)
    (BUNDLE / "configs").mkdir(parents=True)
    (BUNDLE / "tests").mkdir(parents=True)
    files = {
        ROOT / "colab" / "README.md": BUNDLE / "colab" / "README.md",
        ROOT / "colab" / "requirements-colab.txt": BUNDLE / "colab" / "requirements-colab.txt",
        ROOT / "scripts" / "run_multiclass_utility_sweep.py": BUNDLE / "scripts" / "run_multiclass_utility_sweep.py",
        ROOT / "configs" / "multiclass_resnet50_utility_sweep.yaml": BUNDLE / "configs" / "multiclass_resnet50_utility_sweep.yaml",
        ROOT / "configs" / "multiclass_resnet50_local_sweep.yaml": BUNDLE / "configs" / "multiclass_resnet50_local_sweep.yaml",
        ROOT / "configs" / "multiclass_resnet50_t4_smoke.yaml": BUNDLE / "configs" / "multiclass_resnet50_t4_smoke.yaml",
        ROOT / "configs" / "multiclass_resnet18_utility_sweep.yaml": BUNDLE / "configs" / "multiclass_resnet18_utility_sweep.yaml",
        ROOT / "configs" / "multiclass_resnet18_t4_smoke.yaml": BUNDLE / "configs" / "multiclass_resnet18_t4_smoke.yaml",
        ROOT / "configs" / "multiclass_cnn_nonprivate_control.yaml": BUNDLE / "configs" / "multiclass_cnn_nonprivate_control.yaml",
        ROOT / "configs" / "multiclass_cnn_dp_diagnostic.yaml": BUNDLE / "configs" / "multiclass_cnn_dp_diagnostic.yaml",
        ROOT / "configs" / "multiclass_cnn_rounds_ablation.yaml": BUNDLE / "configs" / "multiclass_cnn_rounds_ablation.yaml",
    }
    for source, destination in files.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copytree(
        ROOT / "src" / "dp_forgetbench",
        BUNDLE / "src" / "dp_forgetbench",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "tests" / "test_multiclass_cnn.py", BUNDLE / "tests" / "test_multiclass_cnn.py")
    archive = shutil.make_archive(str(ROOT / "colab_bundle" / "dp_forgetbench_colab_utility"), "zip", BUNDLE.parent, BUNDLE.name)
    print(f"Created {archive}")


if __name__ == "__main__":
    main()
