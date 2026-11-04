"""Shared constants and the data-layer defaults. Row ids (E1, T3A, ...) refer to decisions.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# T2: per-channel standardization, computed on all 50k training images
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
GRAY = (0.299, 0.587, 0.114)

# T3B: train-set color PCA (exploration/data_diagnostics.py); columns are PC1..PC3, rows are R, G, B
EIGVAL = (0.1719, 0.0139, 0.0029)
EIGVEC = ((-0.5580, 0.7063, -0.4356),
          (-0.5775, 0.0464, 0.8151),
          (-0.5960, -0.7063, -0.3820))

# Data-layer defaults (the "A" row of each decision). Experiments override individual keys.
DATA = dict(
    flip=True,                # T1A
    aug_repeats=1,            # T1B: 2 = two independently augmented copies of each image per batch
    photometric="jitter",     # T3: "none" | "jitter" (A) | "jitter+pca" (B) | "trivialaugment" (C)
    jitter=0.2,               # T3A: brightness/contrast factor range 1 +- jitter
    pca_lighting=0.1,         # T3B: std of the per-image PCA lighting coefficients
    strong_aug_off_last=0,    # T4B: epochs at the end with photometric augmentation switched off
    crop_pad=4,               # T5: reflect-padded random crop
    occlusion="cutmix",       # T6: "cutmix" (A) | "cutout" (B) | "none"
    cutout=16,                # T6B: square side
    cutmix_p=0.5,             # T6A: probability a batch gets CutMix
    cutmix_alpha=1.0,         # T6A: lambda ~ Beta(alpha, alpha)
    grayscale_p=0.0,          # T7: 0 (A) | 0.1 (B)
    tta="flip",               # I2: "none" | "flip" (A) | "flip+shift" (B)
)
