"""Build the cross-validation folds (L1): stratified, grouped 5-fold.

Usage: python -m src.folds   (writes data/folds.npy; run once, then commit the file)

Each duplicate group from data/dup_groups_loose.npy goes entirely into one fold. Within each class,
groups are assigned largest-first to the fold holding the fewest images of that class, so every
fold gets ~1,000 images per class.
"""
import numpy as np

from .config import DATA_DIR

K, SEED = 5, 0


def make_folds(y, groups, k=K, seed=SEED):
    rng = np.random.default_rng(seed)
    folds = np.full(len(y), -1)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        gids, inv = np.unique(groups[idx], return_inverse=True)
        members = [idx[inv == g] for g in range(len(gids))]
        order = sorted(rng.permutation(len(members)), key=lambda g: -len(members[g]))  # big first, ties random
        count = np.zeros(k, int)
        for g in order:
            f = int(np.argmin(count + rng.random(k) * 1e-3))     # least-filled fold, random tie-break
            folds[members[g]] = f
            count[f] += len(members[g])
    return folds


if __name__ == "__main__":
    y = np.load(DATA_DIR / "cache_uint8.npz")["y_train"]
    groups = np.load(DATA_DIR / "dup_groups_loose.npy")
    folds = make_folds(y, groups)
    np.save(DATA_DIR / "folds.npy", folds)
    per = np.array([[np.sum((folds == f) & (y == c)) for c in range(10)] for f in range(K)])
    print("images per fold (rows) x class (cols):\n", per)
    split = sum(len(set(folds[groups == g])) > 1 for g in np.unique(groups))
    print(f"duplicate groups split across folds: {split}")
