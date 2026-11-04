"""Quick look at the competition data: example grid per class, test samples, basic stats.

Usage: python exploration/look_at_data.py   (run from the project folder)
"""
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
PER_CLASS = 10
random.seed(0)


def load(folder, name):
    return np.asarray(Image.open(DATA / folder / name).convert("RGB"))


def save(fig, name):
    """Save a figure, skipping it if the file is locked (e.g. open in the Photos app)."""
    try:
        fig.savefig(OUT / name, dpi=110)
    except OSError:
        print(f"skipped {name} (file is locked, probably open in a viewer)")


with open(DATA / "train_labels.csv") as f:
    train = [(r["filename"], int(r["label"])) for r in csv.DictReader(f)]
with open(DATA / "test.csv") as f:
    test = [r["filename"] for r in csv.DictReader(f)]

# --- Grid: one row per class -------------------------------------------------
fig, axes = plt.subplots(10, PER_CLASS + 1, figsize=(PER_CLASS + 2.5, 10.5),
                         gridspec_kw={"width_ratios": [2.2] + [1] * PER_CLASS})
for c, cname in enumerate(CLASSES):
    picks = random.sample([fn for fn, y in train if y == c], PER_CLASS)
    axes[c, 0].text(0.95, 0.5, f"{c}  {cname}", ha="right", va="center", fontsize=12)
    axes[c, 0].axis("off")
    for j, fn in enumerate(picks):
        axes[c, j + 1].imshow(load("train_images", fn), interpolation="nearest")
        axes[c, j + 1].axis("off")
fig.suptitle("Training images (32x32, shown with no smoothing)", fontsize=14)
fig.tight_layout()
save(fig, "train_grid.png")

# --- Test samples (unlabeled) ------------------------------------------------
fig, axes = plt.subplots(3, 10, figsize=(11, 3.6))
for ax, fn in zip(axes.flat, random.sample(test, 30)):
    ax.imshow(load("test_images", fn), interpolation="nearest")
    ax.axis("off")
fig.suptitle("Random test images (no labels)", fontsize=14)
fig.tight_layout()
save(fig, "test_grid.png")

# --- Stats -------------------------------------------------------------------
counts = np.bincount([y for _, y in train], minlength=10)
print("class counts:", dict(zip(CLASSES, counts.tolist())))


def channel_stats(folder, names):
    x = np.stack([load(folder, n) for n in names]).astype(np.float32) / 255
    return x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2)), x.shape[1:]


m, s, shape = channel_stats("train_images", [fn for fn, _ in random.sample(train, 5000)])
print(f"train (5k sample) shape {shape}  mean RGB {m.round(3)}  std RGB {s.round(3)}")
m, s, shape = channel_stats("test_images", test)
print(f"test  (all 2000)  shape {shape}  mean RGB {m.round(3)}  std RGB {s.round(3)}")
