"""Atypicality of each image relative to the training data (model-independent).

Features: 16x16 area-downsampled RGB thumbnails, standardized, PCA to 64 dims (fit on train only).
Score for a train image: mean cosine distance to its 10 nearest same-class images in the *other*
folds (i.e. the training split that held-out image's models saw). Score for a test image: mean distance
to its 10 nearest training images of any class.

Writes data/atypicality.npz (train_score, test_score) and prints how the test set compares.
  python exploration/atypicality.py
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = 10


def features(x):
    n = len(x)
    t = x.reshape(n, 16, 2, 16, 2, 3).mean((2, 4)).reshape(n, -1).astype(np.float32) / 255
    return t


def knn_mean_dist(q, ref, k=K, batch=2000):
    out = np.empty(len(q), np.float32)
    for i in range(0, len(q), batch):
        s = q[i:i + batch] @ ref.T
        part = np.partition(-s, k, axis=1)[:, :k]
        out[i:i + batch] = 1 + part.mean(1)          # cosine distance = 1 - similarity
    return out


d = np.load(ROOT / "data" / "cache_uint8.npz")
x, y, xt = d["x_train"], d["y_train"], d["x_test"]
folds = np.load(ROOT / "data" / "folds.npy")

f, ft = features(x), features(xt)
mu, sd = f.mean(0), f.std(0) + 1e-6
f, ft = (f - mu) / sd, (ft - mu) / sd
_, _, vt = np.linalg.svd(f[::5] - f[::5].mean(0), full_matrices=False)
P = vt[:64].T
f, ft = f @ P, ft @ P
f /= np.linalg.norm(f, axis=1, keepdims=True)
ft /= np.linalg.norm(ft, axis=1, keepdims=True)

score = np.empty(len(x), np.float32)
for k in range(5):
    for c in range(10):
        q = np.where((folds == k) & (y == c))[0]
        r = np.where((folds != k) & (y == c))[0]
        score[q] = knn_mean_dist(f[q], f[r])
test_score = knn_mean_dist(ft, f)
# comparable reference for held-out: distance to nearest training images of any class (other folds)
heldout_any = np.empty(len(x), np.float32)
for k in range(5):
    q, r = np.where(folds == k)[0], np.where(folds != k)[0]
    heldout_any[q] = knn_mean_dist(f[q], f[r])

np.savez(ROOT / "data" / "atypicality.npz", train_score=score, test_score=test_score, heldout_any=heldout_any)
qs = [0.1, 0.25, 0.5, 0.75, 0.9]
print("held-out (any class) quantiles:", np.round(np.quantile(heldout_any, qs), 4))
print("test     (any class) quantiles:", np.round(np.quantile(test_score, qs), 4))
for p in (0.8, 0.9):
    thr = np.quantile(heldout_any, p)
    print(f"share of test above held-out {int(p*100)}th pct: {(test_score > thr).mean():.3f} (held-out: {1-p:.2f})")
