"""Transform (T1-T7): batched GPU augmentation, normalization, Cutout and CutMix.

Per-image pipeline (decisions.md execution order):
  geometric (flip T1, reflect-pad crop T5) -> photometric (jitter / PCA lighting T3, grayscale T7)
  -> normalize (T2) -> Cutout (T6B)
CutMix (T6A) is batch-level and runs after the per-image pipeline. TrivialAugment (T3B) applies a
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
    """Add a random offset along the train-set color principal components ("fancy PCA"; rejected in T3A, kept for reference)."""
    alpha = torch.randn(len(x), 3, generator=g, device=x.device) * std
    rgb = (alpha * _const(EIGVAL, x.device)) @ _const(EIGVEC, x.device).T
    return (x + rgb[:, :, None, None]).clamp_(0, 1)


def random_grayscale(x, p, g):
    pick = rand(len(x), g, x.device) < p
    return torch.where(pick[:, None, None, None], gray_of(x).expand_as(x), x)


# --- Renditions (T7C) ------------------------------------------------------------
# Re-depict the object as a sketch, flat-colour graphic, painting, grayscale or framed picture. Parameters
# are drawn per image. The robustness sets (shiftsets.py) use different generators and hold line art out.
RENDITIONS = ("sketch", "poster", "paint", "gray", "framed")
_SOBEL = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))


def _sobel_mag(g):
    k = _const(_SOBEL, g.device)
    p = F.pad(g, (1, 1, 1, 1), mode="replicate")
    return torch.hypot(F.conv2d(p, k.view(1, 1, 3, 3)), F.conv2d(p, k.T.contiguous().view(1, 1, 3, 3)))


def _median3(x):
    b, c, h, w = x.shape
    return F.unfold(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3).view(b, c, 9, h, w).median(2).values


def _quantize(x, levels):
    lv = levels.view(-1, 1, 1, 1)
    return (torch.floor(x * lv).clamp(max=lv - 1) + 0.5) / lv


def render(x, kind, g):
    """x: float (B, 3, H, W) in [0, 1] -> same shape, re-depicted."""
    b, dev = len(x), x.device
    if kind == "gray":
        return gray_of(x).expand_as(x).contiguous()
    if kind == "sketch":                                       # dark Sobel edges on white
        e = _sobel_mag(gray_of(x)).flatten(1)
        q = ((0.9 + 0.09 * rand(b, g, dev)) * (e.shape[1] - 1)).long()
        s = e.sort(1).values.gather(1, q[:, None])
        return (1 - (e / (s + 1e-6)).clamp(0, 1)).view(b, 1, *x.shape[2:]).expand_as(x).contiguous()
    if kind == "poster":                                       # 2-4 levels per channel
        return _quantize(x, torch.randint(2, 5, (b,), generator=g, device=dev).float())
    if kind == "paint":                                        # smoothed, saturated, coarse colour steps
        y = F.avg_pool2d(F.pad(_median3(x), (1, 1, 1, 1), mode="replicate"), 3, 1)
        m = gray_of(y)
        y = (m + (1.2 + 0.8 * rand(b, g, dev)).view(-1, 1, 1, 1) * (y - m)).clamp(0, 1)
        return _quantize(y, torch.randint(5, 9, (b,), generator=g, device=dev).float())
    if kind == "framed":                                       # centred picture on a mat with a frame line
        out = torch.empty_like(x)
        for i in range(b):                                     # ~2% of a batch
            s = int(torch.randint(18, 27, (1,), generator=g, device=dev))
            fw = int(torch.randint(1, 3, (1,), generator=g, device=dev))
            mat = 0.6 + 0.4 * rand(3, g, dev)
            o = (32 - s) // 2
            can = mat.view(3, 1, 1).repeat(1, 32, 32)
            can[:, o - fw:o + s + fw, o - fw:o + s + fw] = (mat * (0.2 + 0.3 * rand(1, g, dev))).view(3, 1, 1)
            can[:, o:o + s, o:o + s] = F.interpolate(x[i:i + 1], size=(s, s), mode="bilinear", align_corners=False)[0]
            out[i] = can
        return out
    raise ValueError(kind)


def random_rendition(x, base, pick, g):
    """Replace the picked images of x with one random rendition each, made from base (pre-photometric)."""
    kinds = torch.randint(0, len(RENDITIONS), (len(x),), generator=g, device=x.device)
    x = x.clone()
    for k, name in enumerate(RENDITIONS):
        sel = pick & (kinds == k)
        if sel.any():
            x[sel] = render(base[sel], name, g)
    return x


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
def augment(x_u8, cfg, g, photometric_on=True, x_raw_u8=None):
    """Per-image training pipeline for a uint8 batch on the target device -> normalized float batch.

    x_raw_u8: the same images before any CPU-side policy (TrivialAugment); renditions start from these.
    """
    x = to_float(x_u8)
    pick = None
    if cfg.get("rendition_p") and photometric_on:          # T7C; absent from older configs (hash unchanged)
        pick = rand(len(x), g, x.device) < cfg["rendition_p"]
        if x_raw_u8 is not None:
            x = torch.where(pick[:, None, None, None], to_float(x_raw_u8), x)
    if cfg["flip"]:
        x = random_flip(x, g)
    if cfg["crop_pad"]:
        x = random_crop_reflect(x, cfg["crop_pad"], g)
    base = x
    if photometric_on:
        if cfg["photometric"] in ("jitter", "jitter+pca"):
            x = jitter(x, cfg["jitter"], g)
        if cfg["photometric"] == "jitter+pca":
            x = pca_lighting(x, cfg["pca_lighting"], g)
        if cfg["grayscale_p"]:
            x = random_grayscale(x, cfg["grayscale_p"], g)
    if pick is not None and pick.any():
        x = random_rendition(x, base, pick, g)               # re-depicted images skip the photometric stage
    x = normalize(x)
    if cfg["occlusion"] == "cutout":
        x = cutout(x, cfg["cutout"], g)
    return x
