"""Robustness evaluation sets built from held-out training images (labels unchanged).

Motivated by the kinds of test images our models found hard (decisions log, 2026-11-04): objects shown as
pictures, paintings or diagrams rather than photos (depiction), and photos with an extra out-of-class object
in frame (distractors). Every transform is generic and deterministic per training image (seeded by its
global index), so all folds and runs see the same version of the same image.

  depiction:  sketch, lineart, poster, paint, gray, framed
  distractor: distract (an off-centre patch from another image of a different class)

L6: these generators differ from the training renditions (T7C, transform.py), so a gain here reflects
invariance rather than a memorized filter: sketch uses difference-of-Gaussians edges (training: Sobel),
poster a mode filter + fixed 2 bits (training: 2-4 levels, no filter), paint PIL median/smooth + 3 bits
(training: median + box blur, 5-8 levels), framed an off-centre 20 px picture (training: centred, 18-26 px).
Line art is never used in training. Grayscale is the same transform by definition.
"""
import numpy as np
from PIL import Image, ImageFilter

from .config import GRAY

GRAY_W = np.array(GRAY, np.float32)


def _gray(x):
    return (x.astype(np.float32) * GRAY_W).sum(-1)


def _sobel(g):
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) * 2 + (p[:, :-2, 2:] - p[:, :-2, :-2]) + (p[:, 2:, 2:] - p[:, 2:, :-2])
    gy = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) * 2 + (p[:, 2:, :-2] - p[:, :-2, :-2]) + (p[:, 2:, 2:] - p[:, :-2, 2:])
    return np.hypot(gx, gy)


def _rgb(g):
    return np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, -1)


def sketch(x, idx=None):
    """Pencil-sketch look: dark difference-of-Gaussians edges on white (grayscale)."""
    out = np.empty(x.shape[:3], np.float32)
    for i, g in enumerate(_gray(x)):
        im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
        a = np.asarray(im.filter(ImageFilter.GaussianBlur(0.6)), np.float32)
        b = np.asarray(im.filter(ImageFilter.GaussianBlur(1.6)), np.float32)
        out[i] = np.abs(a - b)
    e = out / (np.percentile(out, 97, axis=(1, 2), keepdims=True) + 1e-6)
    return _rgb(255 * (1 - np.clip(e, 0, 1)))


def lineart(x, idx=None):
    """Diagram / clip-art look: binary black lines on white."""
    e = _sobel(_gray(x))
    t = np.percentile(e, 80, axis=(1, 2), keepdims=True)
    return _rgb(np.where(e > t, 0, 255))


def poster(x, idx=None):
    """Flat-colour graphic: 2 bits per channel after a mode filter (few colours, flat regions)."""
    out = np.empty_like(x)
    for i, im in enumerate(x):
        out[i] = np.asarray(Image.fromarray(im).filter(ImageFilter.ModeFilter(3)))
    return (out >> 6 << 6) + 32


def paint(x, idx=None):
    """Painting look: smoothed, strongly saturated, coarse colour steps."""
    out = np.empty_like(x)
    for i, im in enumerate(x):
        p = Image.fromarray(im).filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.SMOOTH_MORE)
        out[i] = np.asarray(p)
    f = out.astype(np.float32)
    g = _gray(out)[..., None]
    f = np.clip(g + 1.6 * (f - g), 0, 255)
    return ((f.astype(np.uint8) >> 5) << 5) + 16


def gray(x, idx=None):
    return _rgb(_gray(x))


def framed(x, idx):
    """A picture of a picture: the image shrunk to 20x20, off-centre on a mat, with a thin dark frame line."""
    out = np.empty_like(x)
    for i, im in enumerate(x):
        rng = np.random.default_rng(int(idx[i]) + 777)
        mat = rng.integers(170, 256, 3)
        oy, ox = rng.integers(2, 11, 2)                        # top-left corner of the frame line
        can = np.tile(mat.astype(np.uint8), (32, 32, 1))
        can[oy:oy + 22, ox:ox + 22] = (mat * 0.35).astype(np.uint8)
        can[oy + 1:oy + 21, ox + 1:ox + 21] = np.asarray(Image.fromarray(im).resize((20, 20), Image.BILINEAR))
        out[i] = can
    return out


def distract(x, idx, y):
    """Paste a 14x14 patch from another image of a different class into one corner (centre 12x12 untouched)."""
    rng = np.random.default_rng(np.asarray(idx) + 991)
    out = x.copy()
    n = len(x)
    for i in range(n):
        j = rng.integers(n)
        while y[j] == y[i]:
            j = rng.integers(n)
        sy, sx = rng.integers(0, 19, 2)                       # source patch anywhere
        cy, cx = [(0, 0), (0, 18), (18, 0), (18, 18)][rng.integers(4)]
        out[i, cy:cy + 14, cx:cx + 14] = x[j, sy:sy + 14, sx:sx + 14]
    return out


DEPICTION = {"sketch": sketch, "lineart": lineart, "poster": poster, "paint": paint, "gray": gray, "framed": framed}


def build(x, idx, y):
    """All evaluation variants of the given held-out images: {name: uint8 array}, 'clean' included."""
    sets = {"clean": x}
    for name, f in DEPICTION.items():
        sets[name] = f(x, idx)
    sets["distract"] = distract(x, idx, y)
    return sets
