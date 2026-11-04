"""Data-layer diagnostics: color structure, train/test shift, background shortcuts, locality.

Usage: python exploration/data_diagnostics.py   (statistics only, no models trained; ~1 min after caching)

Sections
  1. Cache all images into data/cache_uint8.npz (decoded once, reused by later runs)
  2. Color structure: RGB correlation + color PCA (is late fusion / CMY / gray worth it?)
  3. Train vs test shift: per-image brightness/contrast, borders, spectrum, edge energy,
     KS distances
  4. Background shortcut: class mean images + per-region class separability (ANOVA ratio)
"""
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "diagnostics"
OUT.mkdir(exist_ok=True)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def save(fig, name):
    """Save a figure, skipping it if the file is locked (e.g. open in the Photos app)."""
    try:
        fig.savefig(OUT / name, dpi=110)
    except OSError:
        print(f"skipped {name} (file is locked, probably open in a viewer)")
    plt.close(fig)


def section(title):
    print(f"\n=== {title} " + "=" * (70 - len(title)))


# --- 1. Cache ----------------------------------------------------------------
cache = DATA / "cache_uint8.npz"
if not cache.exists():
    t = time.time()
    with open(DATA / "train_labels.csv") as f:
        rows = list(csv.DictReader(f))
    x_tr = np.stack([np.asarray(Image.open(DATA / "train_images" / r["filename"]).convert("RGB"))
                     for r in rows])
    y_tr = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    with open(DATA / "test.csv") as f:
        rows = list(csv.DictReader(f))
    x_te = np.stack([np.asarray(Image.open(DATA / "test_images" / r["filename"]).convert("RGB"))
                     for r in rows])
    id_te = np.array([int(r["id"]) for r in rows], dtype=np.int64)
    np.savez(cache, x_train=x_tr, y_train=y_tr, x_test=x_te, id_test=id_te)
    print(f"cached images in {time.time() - t:.0f}s -> {cache.name}")
d = np.load(cache)
x_tr, y_tr, x_te = d["x_train"], d["y_train"], d["x_test"]
X = x_tr.astype(np.float32) / 255       # (50000, 32, 32, 3)
T = x_te.astype(np.float32) / 255       # (2000, 32, 32, 3)
print(f"train {X.shape}  test {T.shape}")

# --- 2. Color structure ------------------------------------------------------
section("2. Color structure")
pix = X.reshape(-1, 3)
corr = np.corrcoef(pix.T)
print("pixel-level RGB correlation:\n", corr.round(3))
cov = np.cov(pix.T)
evals, evecs = np.linalg.eigh(cov)
evals, evecs = evals[::-1], evecs[:, ::-1]
print("color PCA variance share:", (evals / evals.sum()).round(4))
for k in range(3):
    print(f"  PC{k + 1} weights (R,G,B): {evecs[:, k].round(3)}")
# correlation of edges (high-pass) per channel: do channels carry the *same* structure?
hp = X - 0.25 * (np.roll(X, 1, 1) + np.roll(X, -1, 1) + np.roll(X, 1, 2) + np.roll(X, -1, 2))
print("high-pass (edge) RGB correlation:\n", np.corrcoef(hp[:, 1:-1, 1:-1].reshape(-1, 3).T).round(3))

# --- 3. Train vs test shift --------------------------------------------------
section("3. Train vs test shift")


def per_image(A):
    g = A @ GRAY_W
    return g.mean(axis=(1, 2)), g.std(axis=(1, 2))


m_tr, s_tr = per_image(X)
m_te, s_te = per_image(T)
q = [5, 25, 50, 75, 95]
print("per-image brightness (gray mean) percentiles", q)
print("  train", np.percentile(m_tr, q).round(3), "  test", np.percentile(m_te, q).round(3))
print("per-image contrast (gray std) percentiles", q)
print("  train", np.percentile(s_tr, q).round(3), "  test", np.percentile(s_te, q).round(3))
pooled_var = (s_tr ** 2).mean() + m_tr.var()
print(f"train pooled gray variance split: within-image {(s_tr ** 2).mean() / pooled_var:.0%}"
      f", between-image {m_tr.var() / pooled_var:.0%}")


def border_flat(A, w=2, tol=0.04):
    """Fraction of images whose outer w-pixel ring is nearly uniform (a frame / padding)."""
    g = A @ GRAY_W
    ring = np.concatenate([g[:, :w].reshape(len(g), -1), g[:, -w:].reshape(len(g), -1),
                           g[:, w:-w, :w].reshape(len(g), -1), g[:, w:-w, -w:].reshape(len(g), -1)], 1)
    return (ring.std(1) < tol).mean()


print(f"near-uniform 2px frame: train {border_flat(X):.1%}  test {border_flat(T):.1%}")


def radial_spectrum(A):
    g = A @ GRAY_W
    g = g - g.mean(axis=(1, 2), keepdims=True)
    P = (np.abs(np.fft.fftshift(np.fft.fft2(g), axes=(1, 2))) ** 2).mean(0)
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - 16, xx - 16).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.bincount(r.ravel())


sp_tr, sp_te = radial_spectrum(X), radial_spectrum(T)
ratio = sp_te[:17] / sp_tr[:17]
print("test/train power by radial frequency (0=DC..16=Nyquist):\n ", ratio.round(2))

sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)


def edge_energy(A):
    g = torch.from_numpy(A @ GRAY_W)[:, None]
    k = torch.from_numpy(np.stack([sob, sob.T]))[:, None]
    e = F.conv2d(g, k)
    return (e ** 2).sum(1).sqrt().mean(dim=(1, 2)).numpy()


print(f"mean Sobel edge magnitude: train {edge_energy(X).mean():.3f}  test {edge_energy(T).mean():.3f}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].hist(m_tr, 60, density=True, alpha=.6, label="train"); ax[0].hist(m_te, 60, density=True, alpha=.6, label="test")
ax[0].set_title("per-image brightness"); ax[0].legend()
ax[1].hist(s_tr, 60, density=True, alpha=.6); ax[1].hist(s_te, 60, density=True, alpha=.6)
ax[1].set_title("per-image contrast")
ax[2].semilogy(sp_tr[:17], label="train"); ax[2].semilogy(sp_te[:17], label="test")
ax[2].set_title("radial power spectrum"); ax[2].set_xlabel("radial freq (16 = Nyquist)"); ax[2].legend()
fig.tight_layout(); save(fig, "shift.png")



def ks(a, b):
    """Kolmogorov-Smirnov distance: max gap between the two empirical CDFs (0 = same, 1 = disjoint)."""
    grid = np.sort(np.concatenate([a, b]))
    return np.abs(np.searchsorted(np.sort(a), grid, "right") / len(a)
                  - np.searchsorted(np.sort(b), grid, "right") / len(b)).max()


print(f"KS distance train vs test: brightness {ks(m_tr, m_te):.3f}  contrast {ks(s_tr, s_te):.3f}")
for c, ch in enumerate("RGB"):
    print(f"  channel {ch} mean: KS {ks(X[..., c].mean(axis=(1, 2)), T[..., c].mean(axis=(1, 2))):.3f}")

# --- 4. Background shortcut --------------------------------------------------
section("4. Background shortcut (no models: class statistics only)")
fig, axes = plt.subplots(1, 10, figsize=(14, 1.9))
for c, a in enumerate(axes):
    a.imshow(X[y_tr == c].mean(0)); a.set_title(CLASSES[c], fontsize=9); a.axis("off")
fig.suptitle("mean image per class (what the background alone looks like)")
fig.tight_layout(); save(fig, "class_means.png")
print("mean RGB of top 8 rows / bottom 8 rows per class:")
for c in range(10):
    top = X[y_tr == c, :8].mean(axis=(0, 1, 2)); bot = X[y_tr == c, -8:].mean(axis=(0, 1, 2))
    print(f"  {CLASSES[c]:10s} top {top.round(2)}  bottom {bot.round(2)}")

center = np.zeros((32, 32), bool); center[8:24, 8:24] = True   # 256 px center vs 768 px border


def fisher(F_):
    """Per-feature one-way ANOVA ratio: between-class variance / within-class variance."""
    mu = np.stack([F_[y_tr == c].mean(0) for c in range(10)])
    within = np.stack([F_[y_tr == c].var(0) for c in range(10)]).mean(0)
    return mu.var(0) / within


regions = {"border (outer ring)": ~center, "center 16x16": center,
           "top 8 rows": np.arange(32)[:, None] < 8, "bottom 8 rows": np.arange(32)[:, None] >= 24}
print("class separability of a region's mean color (between/within variance; higher = more label info)")
for name, r in regions.items():
    r = np.broadcast_to(r, (32, 32))
    f = fisher(X[:, r].mean(1))
    print(f"  {name:20s} R {f[0]:.3f}  G {f[1]:.3f}  B {f[2]:.3f}")
