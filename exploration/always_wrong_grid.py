"""Grid of training images that every held-out model got wrong (screens of 2026-11-04).

Input: a text file of comma-separated base-36 codes, code = image_index * 10 + most common wrong prediction.
Output: diagnostics/always_wrong_grid.png, one row block per true class.

  python exploration/always_wrong_grid.py <codes.txt>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLS = 16

codes = [int(c, 36) for c in Path(sys.argv[1]).read_text().strip().split(",")]
d = np.load(ROOT / "data" / "cache_uint8.npz")
x, y = d["x_train"], d["y_train"]
by_class = {k: [(c // 10, c % 10) for c in codes if y[c // 10] == k] for k in range(10)}

rows = sum(-(-len(v) // COLS) for v in by_class.values() if v)
fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 1.05, rows * 1.25))
r = 0
for k, items in by_class.items():
    for i in range(-(-len(items) // COLS)):
        for j in range(COLS):
            ax = axes[r, j]
            ax.axis("off")
            n = i * COLS + j
            if n < len(items):
                g, p = items[n]
                ax.imshow(x[g], interpolation="nearest")
                ax.set_title(f"{g}\n→{CLASSES[p]}", fontsize=6.5, pad=1)
        axes[r, 0].text(-0.15, 0.5, f"{CLASSES[k]} ({len(items)})" if i == 0 else "", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9, fontweight="bold")
        r += 1
fig.suptitle("Training images every held-out model got wrong (label on left; title = index, →most common prediction)", fontsize=11)
plt.subplots_adjust(left=0.07, right=0.995, top=0.985, bottom=0.005, wspace=0.05, hspace=0.45)
out = ROOT / "exploration" / "diagnostics" / "always_wrong_grid.png"
fig.savefig(out, dpi=110)
print(out, len(codes), {CLASSES[k]: len(v) for k, v in by_class.items()})
