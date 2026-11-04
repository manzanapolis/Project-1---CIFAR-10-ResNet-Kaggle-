"""Extract (E1, E3): verify the Kaggle files and decode them once into a single uint8 cache.

Usage:
  python -m src.extract           verify files, build data/cache_uint8.npz if missing (~5 min on a laptop)
  python -m src.extract --check   verify files only

Expected layout (unzipped Kaggle download): data/train_labels.csv, data/test.csv,
data/sample_submission.csv, data/train_images/*.png, data/test_images/*.png.
Duplicate groups (E2) come from exploration/dedup_transforms.py -> data/dup_groups*.npy.
"""
import csv
import sys

import numpy as np
from PIL import Image

from .config import DATA_DIR

CACHE = DATA_DIR / "cache_uint8.npz"


def read_csv(name):
    with open(DATA_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def verify():
    """E1 integrity checks; raises AssertionError with a message on the first failure."""
    train, test, sample = read_csv("train_labels.csv"), read_csv("test.csv"), read_csv("sample_submission.csv")
    assert [int(r["id"]) for r in train] == list(range(len(train))), "train ids not contiguous from 0"
    assert [int(r["id"]) for r in test] == list(range(len(test))), "test ids not contiguous from 0"
    assert all(r["filename"] == f"train_{int(r['id']):05d}.png" for r in train), "train filenames don't match ids"
    assert all(r["filename"] == f"test_{int(r['id']):04d}.png" for r in test), "test filenames don't match ids"
    assert [r["id"] for r in sample] == [r["id"] for r in test], "sample_submission ids not in test.csv order"
    assert {p.name for p in (DATA_DIR / "train_images").iterdir()} == {r["filename"] for r in train}, \
        "train_images/ doesn't match train_labels.csv"
    assert {p.name for p in (DATA_DIR / "test_images").iterdir()} == {r["filename"] for r in test}, \
        "test_images/ doesn't match test.csv"
    counts = np.bincount([int(r["label"]) for r in train], minlength=10)
    print(f"verified: {len(train)} train, {len(test)} test, per-class counts {counts.tolist()}")
    return train, test


def decode(folder, rows):
    out = np.empty((len(rows), 32, 32, 3), np.uint8)
    for i, r in enumerate(rows):
        with Image.open(DATA_DIR / folder / r["filename"]) as im:
            assert im.mode == "RGB" and im.size == (32, 32), f"{r['filename']}: {im.mode} {im.size}"
            out[i] = np.asarray(im)
    return out


def build_cache():
    train, test = verify()
    np.savez(CACHE,
             x_train=decode("train_images", train),
             y_train=np.array([int(r["label"]) for r in train], np.int64),
             x_test=decode("test_images", test),
             id_test=np.array([int(r["id"]) for r in test], np.int64))
    print(f"wrote {CACHE}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        verify()
    elif CACHE.exists():
        verify()
        print(f"{CACHE.name} already exists; delete it to rebuild")
    else:
        build_cache()
