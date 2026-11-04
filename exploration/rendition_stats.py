"""Do test images include more non-photographic renditions (drawings, clip art, graphics) than train?

Per-image statistics that separate graphics from photos:
  white   share of near-white pixels (all channels >= 240): drawings / clip art on white
  colors  number of distinct colors after 4-bit quantization: graphics use few colors
  flat    share of pixels whose 4-neighbours are identical: flat fills, not photo texture
  edges   mean gradient magnitude: line art is edge-heavy
Compares train vs test tails: share of images beyond the train 95th/99th percentile (or below the 5th/1st).
  python exploration/rendition_stats.py
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stats(x):
    x = x.astype(np.int16)
    white = (x >= 240).all(-1).mean((1, 2))
    q = (x >> 4)
    codes = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    colors = np.array([len(np.unique(c)) for c in codes])
    g = x.mean(-1)
    same = ((g[:, 1:-1, 1:-1] == g[:, :-2, 1:-1]) & (g[:, 1:-1, 1:-1] == g[:, 2:, 1:-1]) &
            (g[:, 1:-1, 1:-1] == g[:, 1:-1, :-2]) & (g[:, 1:-1, 1:-1] == g[:, 1:-1, 2:]))
    flat = same.mean((1, 2))
    edges = (np.abs(np.diff(g, axis=1)).mean((1, 2)) + np.abs(np.diff(g, axis=2)).mean((1, 2))) / 2
    return {"white": white, "colors": colors, "flat": flat, "edges": edges}


d = np.load(ROOT / "data" / "cache_uint8.npz")
tr, te = stats(d["x_train"]), stats(d["x_test"])
print(f"{'stat':7s} {'train median':>12s} {'test median':>11s}   tail (train share is the reference)")
for k, side in [("white", "high"), ("colors", "low"), ("flat", "high"), ("edges", "high")]:
    msg = []
    for p in (95, 99):
        if side == "high":
            t = np.percentile(tr[k], p)
            msg.append(f"> train p{p}: test {(te[k] > t).mean():.3f} vs {(tr[k] > t).mean():.3f}")
        else:
            t = np.percentile(tr[k], 100 - p)
            msg.append(f"< train p{100-p}: test {(te[k] < t).mean():.3f} vs {(tr[k] < t).mean():.3f}")
    print(f"{k:7s} {np.median(tr[k]):12.3f} {np.median(te[k]):11.3f}   " + " | ".join(msg))
# any-signal graphics flag: beyond the train 99th pct tail on at least one statistic
def flag(s, ref):
    f = (s["white"] > np.percentile(ref["white"], 99)) | (s["colors"] < np.percentile(ref["colors"], 1)) | \
        (s["flat"] > np.percentile(ref["flat"], 99)) | (s["edges"] > np.percentile(ref["edges"], 99))
    return f
print(f"flagged by any 1%-tail: test {flag(te, tr).mean():.3f} vs train {flag(tr, tr).mean():.3f}")
np.savez(ROOT / "data" / "rendition_stats.npz", **{f"train_{k}": v for k, v in tr.items()}, **{f"test_{k}": v for k, v in te.items()})
