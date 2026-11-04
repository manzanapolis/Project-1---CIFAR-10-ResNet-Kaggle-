"""Load (L1-L5): splits, seeded generators, the training batch stream, and the run registry.

  L1  fold splits from data/folds.npy (stratified, grouped 5-fold)
  L2  fold="final" trains on all 50k with no held-out data
  L3  fresh permutation each epoch, without replacement, last partial batch dropped
  L4  uint8 cache held on the GPU, augmentation batched there; every random source seeded
  L5  separate generators for data order and augmentation, seeded by (seed, fold) only, so arms
      in the same fold see the same images in the same order; runs keyed by a config hash
"""
import hashlib
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import TrivialAugmentWide

from . import transform as T
from .config import DATA_DIR, RUNS_DIR


def load_cache():
    d = np.load(DATA_DIR / "cache_uint8.npz")
    return {k: d[k] for k in ("x_train", "y_train", "x_test", "id_test")}


def split(fold):
    """Train / held-out indices for fold 0-4, or all 50k (and no held-out data) for fold="final"."""
    folds = np.load(DATA_DIR / "folds.npy")
    if fold == "final":
        return np.arange(len(folds)), np.array([], dtype=np.int64)
    return np.where(folds != fold)[0], np.where(folds == fold)[0]


def derive_seed(*parts):
    parts = [99 if p == "final" else int(p) for p in parts]
    return int(np.random.SeedSequence(parts).generate_state(1)[0])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                       # also seeds CUDA


class _TrivialAugmentSet(Dataset):
    """Per-image TrivialAugment on the CPU (T3B); returns uint8 HWC like the cache."""
    def __init__(self, x):
        self.x, self.ta = x, TrivialAugmentWide()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i]).permute(2, 0, 1)
        return self.ta(img).permute(1, 2, 0)


class TrainStream:
    """Yields augmented training batches (x, y_a, y_b, lam) for one epoch at a time.

    Batch size and epoch count belong to the training layer and are passed in.
    """
    def __init__(self, x, y, idx, cfg, batch_size, epochs, seed, fold, device, workers=4):
        self.cfg, self.batch_size, self.epochs, self.device = cfg, batch_size, epochs, device
        self.x_np = x[idx]
        self.x = torch.from_numpy(self.x_np).to(device)              # L4: whole split on the GPU
        self.y = torch.from_numpy(y[idx]).to(device)
        self.order_g = torch.Generator().manual_seed(derive_seed(seed, fold, 0))
        self.aug_g = torch.Generator(device=device).manual_seed(derive_seed(seed, fold, 1))
        self.aug_np = np.random.default_rng(derive_seed(seed, fold, 2))
        self.ta_g = torch.Generator().manual_seed(derive_seed(seed, fold, 3))
        self.workers = workers
        self.steps = len(idx) // batch_size

    def _batches(self):
        perm = torch.randperm(len(self.x_np), generator=self.order_g)
        return perm[: self.steps * self.batch_size].view(self.steps, self.batch_size)

    def epoch(self, e):
        cfg, r = self.cfg, self.cfg["aug_repeats"]
        photometric_on = e < self.epochs - cfg["strong_aug_off_last"]           # annealing, rejected in T4A (default 0)
        use_ta = cfg["photometric"] == "trivialaugment" and photometric_on
        batches = self._batches()
        if use_ta:
            loader = DataLoader(_TrivialAugmentSet(self.x_np),
                                batch_sampler=batches.repeat_interleave(r, dim=1).tolist(),
                                num_workers=self.workers, generator=self.ta_g, pin_memory=True)
            raw = (xb.to(self.device, non_blocking=True) for xb in loader)
        else:
            raw = (self.x[b.repeat_interleave(r).to(self.device)] for b in batches)
        for b, x_u8 in zip(batches, raw):
            bi = b.repeat_interleave(r).to(self.device)
            y = self.y[bi]
            x_raw = self.x[bi] if use_ta and cfg.get("rendition_p") else None      # T7C starts from the raw image
            x = T.augment(x_u8, cfg, self.aug_g, photometric_on=photometric_on, x_raw_u8=x_raw)   # TA already applied on CPU
            if cfg["occlusion"] == "cutmix" and self.aug_np.random() < cfg["cutmix_p"]:
                yield T.cutmix(x, y, cfg["cutmix_alpha"], self.aug_np, self.aug_g)
            else:
                yield x, y, y, 1.0


# --- Run registry (L5) -----------------------------------------------------------
def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_dir(cfg):
    return RUNS_DIR / config_hash(cfg)


def is_done(cfg):
    return (run_dir(cfg) / "metrics.json").exists()
