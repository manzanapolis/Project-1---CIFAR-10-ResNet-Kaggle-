"""Kaggle test images our trained models split their votes on most (descriptive only; nothing here feeds training or selection).

Every run in runs/ with test predictions votes once (flip TTA). Images are ranked by the size of the winning vote,
then by vote entropy. Captions: vote counts per class, and what the two best leaderboard models predicted
(E = M3E all-50k sub07, F = M3F all-50k sub09).
Output: exploration/diagnostics/contested_vote_grid.png
  python exploration/vote_grid.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from src import inference as I  # noqa: E402

CL = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLS, ROWS = 12, 8
E, F = "db5220449ec5", "1a453b9d969a"

runs = sorted(p.parent for p in (ROOT / "runs").glob("*/preds.npz") if (p.parent / "config.json").exists())
preds = np.stack([I.tta(np.load(r / "preds.npz")["test"], "flip").argmax(1) for r in runs])   # (models, N)
names = [r.name for r in runs]
counts = np.stack([(preds == k).sum(0) for k in range(10)], 1)                                 # (N, 10)
p = counts / len(runs)
entropy = -(np.where(p > 0, p * np.log(p), 0)).sum(1)
top = counts.max(1)
order = np.lexsort((-entropy, top))                                                            # smallest winning vote first

print(f"{len(runs)} models; test images by size of winning vote:")
for v in range(len(runs), 0, -1):
    n = int((top == v).sum())
    if n:
        print(f"  {v:2d}/{len(runs)}: {n:5d} ({100 * n / len(top):.1f}%)")
e, f = preds[names.index(E)], preds[names.index(F)]
print(f"sub07 vs sub09 disagree on {(e != f).sum()} images; of those, {(top[e != f] < len(runs) * 0.75).mean():.0%} are split votes")

x = np.load(ROOT / "data" / "cache_uint8.npz")["x_test"]
fig = plt.figure(figsize=(COLS * 1.25, ROWS * 1.6))
for n, i in enumerate(order[: COLS * ROWS]):
    ax = fig.add_axes([(n % COLS) / COLS, 0.96 - (n // COLS + 1) * 0.95 / ROWS, 1 / COLS * 0.92, 0.95 / ROWS * 0.68])
    ax.imshow(x[i], interpolation="nearest")
    ax.axis("off")
    votes = " ".join(f"{CL[k]}{counts[i, k]}" for k in np.argsort(-counts[i])[:3] if counts[i, k])
    ax.set_title(f"#{i}  {votes}\nE:{CL[e[i]]} F:{CL[f[i]]}", fontsize=6.5, pad=1)
fig.text(0.5, 0.985, f"Kaggle TEST: images {len(runs)} models disagree on most (votes per class; E/F = our two best LB models)",
         ha="center", fontsize=11, fontweight="bold")
out = ROOT / "exploration" / "diagnostics" / "contested_vote_grid.png"
fig.savefig(out, dpi=110)
print(out)
json.dump({"models": names}, open(out.with_suffix(".json"), "w"))
