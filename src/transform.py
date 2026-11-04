"""Transform (T1-T7): batched GPU augmentation, normalization, Cutout and CutMix.

Per-image pipeline (decisions.md execution order):
  geometric (flip T1, reflect-pad crop T5) -> photometric (jitter / PCA lighting T3, grayscale T7)
  -> normalize (T2) -> Cutout (T6B)
CutMix (T6A) is batch-level and runs after the per-image pipeline. TrivialAugment (T3C) applies a
different random op to each image, so it runs per image on the CPU before this pipeline (see load.py).

All random draws come from the generators passed in, so runs are reproducible (L4) and arms that
share a seed draw the same flips/crops (L5).
"""
import math

import torch
import torch.nn.functional as F

from .config import EIGVAL, EIGVEC, GRAY, MEAN, STD


def _const(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device)


def to_float(x_u8):
    """(B, 32, 32, 3) uint8 -> (B, 3, 32, 32) float in [0, 1]."""
    return x_u8.permute(0, 3, 1, 2).float().div_(255)


def normalize(x):
    return (x - _const(MEAN, x.device).view(1, 3, 1, 1)) / _const(STD, x.device).view(1, 3, 1, 1)


def rand(n, g, device):
    return torch.rand(n, generator=g, device=device)


# --- Geometric -----------------------------------------------------------------
def random_flip(x, g):
    flip = rand(len(x), g, x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def random_crop_reflect(x, pad, g):
    """Shift each image by up to +-pad px, filling the exposed border by mirroring (T5)."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (b,), generator=g, device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), generator=g, device=x.device)
    ar = torch.arange(h, device=x.device)
    xp = xp.gather(2, (oy[:, None] + ar)[:, None, :, None].expand(b, c, h, w + 2 * pad))
    return xp.gather(3, (ox[:, None] + ar)[:, None, None, :].expand(b, c, h, w))


# --- Photometric ---------------------------------------------------------------
def gray_of(x):
    return (x * _const(GRAY, x.device).view(1, 3, 1, 1)).sum(1, keepdim=True)


def jitter(x, s, g):
    """Brightness then contrast, each a random factor in [1 - s, 1 + s] (T3A, ColorJitter semantics)."""
    b = 1 + (rand(len(x), g, x.device) * 2 - 1) * s
    x = (x * b[:, None, None, None]).clamp_(0, 1)
    c = 1 + (rand(len(x), g, x.device) * 2 - 1) * s
    m = gray_of(x).mean((2, 3), keepdim=True)
    return ((x - m) * c[:, None, None, None] + m).clamp_(0, 1)


def pca_lighting(x, std, g):
    """Add a random offset along the train-set color principal components (T3B, "fancy PCA")."""
    alpha = torch.randn(len(x), 3, generator=g, device=x.device) * std
    rgb = (alpha * _const(EIGVAL, x.device)) @ _const(EIGVEC, x.device).T
    return (x + rgb[:, :, None, None]).clamp_(0, 1)


def random_grayscale(x, p, g):
    pick = rand(len(x), g, x.device) < p
    return torch.where(pick[:, None, None, None], gray_of(x).expand_as(x), x)


# --- Occlusion / mixing --------------------------------------------------------
def cutout(x, size, g):
    """Zero a size x size square per image after normalization (= mean color); may extend past the edge (T6B)."""
    b, _, h, w = x.shape
    cy = torch.randint(0, h, (b,), generator=g, device=x.device)
    cx = torch.randint(0, w, (b,), generator=g, device=x.device)
    ys, xs = torch.arange(h, device=x.device), torch.arange(w, device=x.device)
    half = size // 2
    inside_y = (ys[None] >= (cy - half)[:, None]) & (ys[None] < (cy + half)[:, None])
    inside_x = (xs[None] >= (cx - half)[:, None]) & (xs[None] < (cx + half)[:, None])
    keep = ~(inside_y[:, :, None] & inside_x[:, None, :])
    return x * keep[:, None]


def cutmix(x, y, alpha, rng, g):
    """Paste a random box from a shuffled copy of the batch; lam = area kept from the original (T6A).

    Returns (x, y_a, y_b, lam); the loss is lam * CE(y_a) + (1 - lam) * CE(y_b).
    """
    b, _, h, w = x.shape
    lam = rng.beta(alpha, alpha)
    cut = math.sqrt(1 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    perm = torch.randperm(b, generator=g, device=x.device)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)                      # exact kept area after clipping
    return x, y, y[perm], lam


def mixed_cross_entropy(logits, y_a, y_b, lam, label_smoothing=0.0):
    ce = lambda t: F.cross_entropy(logits, t, label_smoothing=label_smoothing)
    return ce(y_a) if lam >= 1 else lam * ce(y_a) + (1 - lam) * ce(y_b)


# --- Full training pipeline ----------------------------------------------------
def augment(x_u8, cfg, g, photometric_on=True):
    """Per-image training pipeline for a uint8 batch on the target device -> normalized float batch."""
    x = to_float(x_u8)
    if cfg["flip"]:
        x = random_flip(x, g)
    if cfg["crop_pad"]:
        x = random_crop_reflect(x, cfg["crop_pad"], g)
    if photometric_on:
        if cfg["photometric"] in ("jitter", "jitter+pca"):
            x = jitter(x, cfg["jitter"], g)
        if cfg["photometric"] == "jitter+pca":
            x = pca_lighting(x, cfg["pca_lighting"], g)
        if cfg["grayscale_p"]:
            x = random_grayscale(x, cfg["grayscale_p"], g)
    x = normalize(x)
    if cfg["occlusion"] == "cutout":
        x = cutout(x, cfg["cutout"], g)
    return x
