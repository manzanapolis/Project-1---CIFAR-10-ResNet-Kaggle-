"""Extract-stage integrity statistics: duplicates within train/test and overlap between them.

Usage: python exploration/integrity_check.py   (statistics only, no models; needs data/cache_uint8.npz)

Near-duplicates are found with cosine similarity of 16x16 grayscale thumbnails after removing
each image's mean and scale, so brightness/contrast changes don't hide a match.
"""
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "diagnostics"
OUT.mkdir(exist_ok=True)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
THRESH = (0.95, 0.98, 0.99)

d = np.load(ROOT / "data" / "cache_uint8.npz")
x_tr, y_tr, x_te = d["x_train"], d["y_train"], d["x_test"]


def save(fig, name):
    """Save a figure, skipping it if the file is locked (e.g. open in the Photos app)."""
    try:
        fig.savefig(OUT / name, dpi=110)
    except OSError:
        print(f"skipped {name} (file is locked, probably open in a viewer)")
    plt.close(fig)


# --- Exact duplicates (byte-identical) ---------------------------------------
def hashes(A):
    return [hashlib.md5(a.tobytes()).hexdigest() for a in A]


h_tr, h_te = hashes(x_tr), hashes(x_te)
print(f"exact duplicates within train: {len(h_tr) - len(set(h_tr))}")
print(f"exact duplicates within test:  {len(h_te) - len(set(h_te))}")
print(f"test images byte-identical to a train image: {len(set(h_te) & set(h_tr))}")


# --- Near duplicates ---------------------------------------------------------
def feats(A):
    g = torch.from_numpy(A).float().mul(torch.tensor([0.299, 0.587, 0.114])).sum(-1)
    g = torch.nn.functional.avg_pool2d(g[:, None], 2).flatten(1)          # 16x16 thumbnail
    g = g - g.mean(1, keepdim=True)
    return g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)


f_tr, f_te = feats(x_tr), feats(x_te)


def nearest(q, ref, exclude_self=False):
    best_s, best_i = [], []
    for i in range(0, len(q), 2000):
        s = q[i:i + 2000] @ ref.T
        if exclude_self:
            s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -1
        v, j = s.max(1)
        best_s.append(v); best_i.append(j)
    return torch.cat(best_s).numpy(), torch.cat(best_i).numpy()


s_tt, i_tt = nearest(f_tr, f_tr, exclude_self=True)
s_te, i_te = nearest(f_te, f_tr)
print("\nnear-duplicates (cosine of mean/scale-normalized 16x16 gray thumbnails)")
for t in THRESH:
    m = s_tt >= t
    pairs = {tuple(sorted((a, b))) for a, b in zip(np.where(m)[0], i_tt[m])}
    diff = sum(y_tr[a] != y_tr[b] for a, b in pairs)
    print(f"  >= {t}: train-train pairs {len(pairs):4d} (label disagreements {diff:3d})   "
          f"test images with a train match {int((s_te >= t).sum()):4d} of {len(s_te)}")
print("test -> nearest train similarity percentiles [50, 90, 99, max]:",
      np.percentile(s_te, [50, 90, 99, 100]).round(3))
print("train -> nearest other train similarity percentiles [50, 90, 99, max]:",
      np.percentile(s_tt, [50, 90, 99, 100]).round(3))

# --- Figure: closest pairs ---------------------------------------------------
fig, axes = plt.subplots(4, 8, figsize=(12, 6.6))
for row, (src, s, idx, lab) in enumerate([("test", s_te, i_te, "test"), ("train", s_tt, i_tt, "train")]):
    order = np.argsort(-s)[:8]
    if src == "train":                                   # skip the mirrored copy of each pair
        seen, keep = set(), []
        for a in np.argsort(-s):
            key = tuple(sorted((a, idx[a])))
            if key not in seen:
                seen.add(key); keep.append(a)
            if len(keep) == 8:
                break
        order = keep
    for k, a in enumerate(order):
        top = x_te[a] if src == "test" else x_tr[a]
        axes[2 * row, k].imshow(top, interpolation="nearest")
        axes[2 * row, k].set_title(f"{lab} {a}" + ("" if src == "test" else f" {CLASSES[y_tr[a]][:5]}"), fontsize=7)
        axes[2 * row + 1, k].imshow(x_tr[idx[a]], interpolation="nearest")
        axes[2 * row + 1, k].set_title(f"train {idx[a]} {CLASSES[y_tr[idx[a]]][:5]}  s={s[a]:.3f}", fontsize=7)
for a in axes.flat:
    a.axis("off")
fig.suptitle("Closest pairs: rows 1-2 test -> nearest train, rows 3-4 train -> nearest other train")
fig.tight_layout()
save(fig, "near_duplicates.png")
