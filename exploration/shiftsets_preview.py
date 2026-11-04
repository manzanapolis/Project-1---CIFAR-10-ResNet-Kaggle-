"""Preview of the robustness evaluation sets (src/shiftsets.py): one row per image, one column per variant.
  python exploration/shiftsets_preview.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.shiftsets import build  # noqa: E402

CL = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
d = np.load(ROOT / "data" / "cache_uint8.npz")
folds = np.load(ROOT / "data" / "folds.npy")
idx = np.where(folds == 0)[0]
pick = np.concatenate([idx[d["y_train"][idx] == c][:1] for c in range(10)])
sets = build(d["x_train"][idx], idx, d["y_train"][idx])
pos = np.searchsorted(idx, pick)

names = list(sets)
fig, axes = plt.subplots(len(pick), len(names), figsize=(len(names) * 1.5, len(pick) * 1.6))
for r, p in enumerate(pos):
    for c, n in enumerate(names):
        ax = axes[r, c]
        ax.imshow(sets[n][p], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(n, fontsize=11, fontweight="bold")
        if c == 0:
            ax.set_ylabel(CL[d["y_train"][pick[r]]], fontsize=10)
plt.tight_layout()
out = ROOT / "exploration" / "diagnostics" / "shiftsets_preview.png"
fig.savefig(out, dpi=110)
print(out)
