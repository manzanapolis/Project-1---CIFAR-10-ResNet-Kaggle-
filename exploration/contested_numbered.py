"""Numbered grid of the 64 most-contested Kaggle test images, for sorting by image TYPE only
(1 unidentifiable, 2 identifiable but drawn/odd, 3 normal photo). No predictions or labels are shown.

  python exploration/contested_numbered.py <contested.txt>
Writes diagnostics/contested_numbered.png and diagnostics/contested_numbered_ids.csv (number -> test id).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLS = 8

line = [l for l in Path(sys.argv[1]).read_text().splitlines() if l.startswith("T ")][0]
ids = [int(t.split(".")[0]) for t in line[2:].split()][:64]
x = np.load(ROOT / "data" / "cache_uint8.npz")["x_test"]

fig, axes = plt.subplots(8, COLS, figsize=(COLS * 1.9, 8 * 2.05))
for n, ax in enumerate(axes.flat):
    ax.imshow(x[ids[n]], interpolation="nearest")
    ax.set_title(str(n + 1), fontsize=13, fontweight="bold", pad=2)
    ax.axis("off")
fig.suptitle("Most-contested test images — sort by type: 1 unidentifiable · 2 identifiable but drawn/odd · 3 normal photo",
             fontsize=12)
plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.08, hspace=0.25)
out = ROOT / "exploration" / "diagnostics" / "contested_numbered.png"
fig.savefig(out, dpi=110)
(out.parent / "contested_numbered_ids.csv").write_text("number,test_id\n" + "".join(f"{n+1},{i}\n" for n, i in enumerate(ids)))
print(out)
