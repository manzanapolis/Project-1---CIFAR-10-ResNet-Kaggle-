"""Report figures for the data section: representative training images per class, CutMix examples, class means, TrivialAugment examples.

Uses the training cache and the project's own cutmix() (src/transform.py), so the CutMix panel shows exactly
what training produced.
  python report/figures/make_data_figures.py      -> report/figures/fig_samples, fig_cutmix, fig_class_means, fig_trivialaugment (.pdf, .png)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import transform as T  # noqa: E402
from src.config import CLASSES  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans", "sans-serif"], "font.size": 9,
                     "svg.fonttype": "none"})
d = np.load(ROOT / "data" / "cache_uint8.npz")
x, y = d["x_train"], d["y_train"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(OUT / f"{name}.pdf")


# --- Figure 1: representative training images, one row per class ---
PER = 10
rng = np.random.default_rng(0)
fig, axes = plt.subplots(len(CLASSES), PER, figsize=(6.5, 6.9), gridspec_kw={"wspace": 0.04, "hspace": 0.04})
for r, name in enumerate(CLASSES):
    for c, i in enumerate(rng.choice(np.flatnonzero(y == r), PER, replace=False)):
        ax = axes[r, c]
        ax.imshow(x[i], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", labelpad=6)
save(fig, "fig_samples")
plt.close(fig)

# --- Figure 2: CutMix examples (image A, image B, result, label weights) ---
N_EX, BATCH = 6, 16
rng = np.random.default_rng(3)
g = torch.Generator().manual_seed(3)
fig, axes = plt.subplots(3, N_EX, figsize=(6.5, 3.9), gridspec_kw={"wspace": 0.08, "hspace": 0.25})
done = 0
while done < N_EX:
    idx = rng.choice(len(x), BATCH, replace=False)
    xb = torch.from_numpy(x[idx]).permute(0, 3, 1, 2).float() / 255
    yb = torch.from_numpy(y[idx]).long()
    xm, ya, yb2, lam = T.cutmix(xb, yb, 1.0, rng, g)
    if ya[0] == yb2[0] or not 0.2 < lam < 0.85:                  # show mixes of two different classes
        continue
    # the pasted image is the one cutmix chose; find it by matching the pasted region
    diff = (xm[0] != xb[0]).any(dim=0)
    ys, xs = torch.nonzero(diff, as_tuple=True)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cand = [k for k in range(BATCH) if torch.equal(xb[k, :, y1:y2, x1:x2], xm[0, :, y1:y2, x1:x2])]
    k = cand[0]
    for row, (img, title) in enumerate([(xb[0], f"A: {CLASSES[ya[0]]}"), (xb[k], f"B: {CLASSES[yb2[0]]}"),
                                        (xm[0], f"{lam:.0%} {CLASSES[ya[0]]}\n{1 - lam:.0%} {CLASSES[yb2[0]]}")]):
        ax = axes[row, done]
        ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(row == 2)
            s.set_linewidth(1.2)
        if row < 2:
            ax.set_title(title, fontsize=8, pad=2)
        else:
            ax.set_xlabel(title, fontsize=8, labelpad=2)
    done += 1
for row, label in enumerate(["image A", "image B", "CutMix"]):
    axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=6)
save(fig, "fig_cutmix")
plt.close(fig)

# --- Figure 3: mean training image per class (what is consistent across a class: background, position) ---
fig, axes = plt.subplots(1, len(CLASSES), figsize=(6.5, 0.95), gridspec_kw={"wspace": 0.06})
for c, ax in enumerate(axes):
    ax.imshow(x[y == c].mean(0).astype(np.uint8), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(CLASSES[c], fontsize=7, pad=2)
save(fig, "fig_class_means")
plt.close(fig)

# --- Figure 4: TrivialAugment examples (torchvision TrivialAugmentWide logic, with the chosen op recorded) ---
from torchvision.transforms import TrivialAugmentWide  # noqa: E402
from torchvision.transforms.autoaugment import _apply_op  # noqa: E402

ta, tg = TrivialAugmentWide(), torch.Generator().manual_seed(7)
space = ta._augmentation_space(ta.num_magnitude_bins)
ops = [o for o in space if o != "Identity"]
ROWS, COLS = 4, 7
rng = np.random.default_rng(7)
fig, axes = plt.subplots(ROWS, COLS + 1, figsize=(6.5, 3.75), gridspec_kw={"wspace": 0.06, "hspace": 0.35})
for r, cls in enumerate(rng.choice(len(CLASSES), ROWS, replace=False)):
    img = torch.from_numpy(x[rng.choice(np.flatnonzero(y == cls))]).permute(2, 0, 1)          # uint8 CHW
    chosen = [ops[i] for i in torch.randperm(len(ops), generator=tg)[:COLS]]                 # distinct ops per row
    panels = [(img, "original")]
    for op in chosen:
        mags, signed = space[op]
        m = float(mags[torch.randint(len(mags), (1,), generator=tg)]) if mags.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,), generator=tg):
            m = -m
        panels.append((_apply_op(img, op, m, interpolation=ta.interpolation, fill=None), op))
    for c, (im, title) in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(im.permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(c == 0)
            s.set_linewidth(1.2)
        ax.set_title(title, fontsize=7, pad=2)
save(fig, "fig_trivialaugment")
plt.close(fig)

# --- Figure 5: every copy of the largest duplicate group (one white Audi, x54) ---
groups = np.load(ROOT / "data" / "dup_groups.npy")
ids, counts = np.unique(groups, return_counts=True)
audi = np.flatnonzero(groups == ids[counts.argmax()])
COLS = 18
fig, axes = plt.subplots(int(np.ceil(len(audi) / COLS)), COLS, figsize=(6.5, 1.15 * np.ceil(len(audi) / COLS) * 6.5 / COLS / 1.0 * 2.7 / 2.7),
                         gridspec_kw={"wspace": 0.05, "hspace": 0.05})
for ax in axes.flat:
    ax.axis("off")
for ax, i in zip(axes.flat, audi):
    ax.imshow(x[i], interpolation="nearest")
save(fig, "fig_dup_audi")
plt.close(fig)
print(f"Audi group: {len(audi)} images, labels {np.bincount(y[audi], minlength=10).tolist()}")

# --- Figure 6: duplicate groups that are internet memes / reposts (picked by eye from the group sheet) ---
MEMES = [(35031, "meme (Impact-font caption)"), (5507, "inspirational picture"), (19716, "stamp")]
rows = [(np.flatnonzero(groups == gid), label) for gid, label in MEMES]
COLS = max(len(m) for m, _ in rows)
fig, axes = plt.subplots(len(rows), COLS, figsize=(6.5 * COLS / 11, 0.72 * len(rows) * 6.5 / 11 * 1.25),
                         gridspec_kw={"wspace": 0.05, "hspace": 0.12}, squeeze=False)
for r, (members, label) in enumerate(rows):
    for c in range(COLS):
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if c < len(members):
            ax.imshow(x[members[c]], interpolation="nearest")
    axes[r, 0].set_ylabel(f"{label}\n{CLASSES[y[members[0]]]}, x{len(members)}", rotation=0, ha="right",
                          va="center", labelpad=6, fontsize=7)
save(fig, "fig_dup_memes")
plt.close(fig)
