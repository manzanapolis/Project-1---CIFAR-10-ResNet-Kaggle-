"""Side-by-side grids of the images our 50 models disagree on most: Kaggle test (left) vs held-out train (right).

Input: text file with two lines,
  T <test_idx.majority.second.share%> ...
  H <train_idx.true.majority.second.share%> ...
Output: diagnostics/contested_test_vs_heldout.png
  python exploration/contested_grid.py <contested.txt>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CL = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLS = 8

lines = dict(l.split(" ", 1) for l in Path(sys.argv[1]).read_text().strip().splitlines())
T = [list(map(int, t.split("."))) for t in lines["T"].split()]
H = [list(map(int, t.split("."))) for t in lines["H"].split()]
d = np.load(ROOT / "data" / "cache_uint8.npz")

fig = plt.figure(figsize=(2 * COLS * 1.15 + 0.6, 8 * 1.35))
for side, (items, imgs, title) in enumerate([(T, d["x_test"], "Kaggle TEST: most contested (majority / runner-up, vote share)"),
                                             (H, d["x_train"], "HELD-OUT train: most contested (TRUE label; majority / runner-up)")]):
    for n, it in enumerate(items[: COLS * 8]):
        ax = fig.add_axes([side * 0.505 + (n % COLS) * 0.495 / COLS, 0.95 - (n // COLS + 1) * 0.94 / 8,
                           0.495 / COLS * 0.95, 0.94 / 8 * 0.75])
        ax.imshow(imgs[it[0]], interpolation="nearest")
        ax.axis("off")
        if side == 0:
            cap = f"{CL[it[1]]}/{CL[it[2]]} {it[3]}%"
        else:
            cap = f"TRUE {CL[it[1]]}\n{CL[it[2]]}/{CL[it[3]]} {it[4]}%"
        ax.set_title(cap, fontsize=6.5, pad=1)
    fig.text(side * 0.505 + 0.25, 0.985, title, ha="center", fontsize=10, fontweight="bold")
out = ROOT / "exploration" / "diagnostics" / "contested_test_vs_heldout.png"
fig.savefig(out, dpi=120)
print(out)
