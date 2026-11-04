"""Transform-invariant near-duplicate search with verification (statistics only, no models).

Usage: python exploration/dedup_transforms.py   (~4 min on a laptop CPU; needs data/cache_uint8.npz)

Stage 1, coarse search: every query image is compared under 18 transforms (identity, center zoom
90/80/70/60%, four 28x28 corner crops; each with and without a mirror flip) against the
untransformed reference set, using cosine similarity of 16x16 grayscale thumbnails with each
image's own mean/scale removed (so brightness, contrast, tint and grayscale copies still match).

Stage 2, verification: the coarse score is dominated by layout (sky band, water band), so lookalike
scenes pass it. Candidates are re-scored at full resolution on edge structure: blur 3x3, Laplacian,
then remove each row's mean (purely horizontal structure such as a horizon line is shared by
lookalike ship/sea scenes but says nothing about identity), then normalized correlation.

Each query keeps its top-K matches (not just the best), so a cluster of near-identical copies can
still link to a second cluster of the same photo.

Duplicate rule (thresholds set by visual inspection, see exploration/diagnostics/manual_labels.csv):
  coarse >= COARSE and edge >= EDGE; train-train pairs must also share a label.
  A looser edge cutoff (EDGE_LOOSE) gives over-inclusive groups for splitting, where a missed
  duplicate leaks across the split but a false merge is harmless.

Outputs
  data/dup_groups.npy                          group id per train image (strict rule; for counts)
  data/dup_groups_loose.npy                    group id per train image (loose rule; for splitting)
  exploration/diagnostics/dup_pairs.csv        train-train candidates with coarse + edge scores
  exploration/diagnostics/dup_test_train.csv   best train match per test image, with scores
  exploration/diagnostics/dup_test_test.csv    best other-test match per test image, with scores
  exploration/diagnostics/dedup_*.png          groups, test->train, test->test, borderline pairs
"""
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "diagnostics"
OUT.mkdir(exist_ok=True)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
COARSE, EDGE, EDGE_LOOSE = 0.95, 0.50, 0.40
K = 5                           # matches kept per query
CAND = 0.90                     # coarse score kept as a candidate for verification
GRAY = torch.tensor([0.299, 0.587, 0.114])
LAP = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
torch.set_num_threads(4)
t0 = time.time()

d = np.load(ROOT / "data" / "cache_uint8.npz")
x_tr, y_tr, x_te = d["x_train"], d["y_train"], d["x_test"]
g_tr = torch.from_numpy(x_tr).float().mul(GRAY).sum(-1)[:, None]      # (N, 1, 32, 32)
g_te = torch.from_numpy(x_te).float().mul(GRAY).sum(-1)[:, None]


def save(fig, name):
    """Save a figure, skipping it if the file is locked (e.g. open in the Photos app)."""
    try:
        fig.savefig(OUT / name, dpi=110)
    except OSError:
        print(f"skipped {name} (file is locked, probably open in a viewer)")
    plt.close(fig)


# --- Stage 1: coarse search ----------------------------------------------------
def thumb(g):
    f = F.avg_pool2d(g, 2).flatten(1)
    f = f - f.mean(1, keepdim=True)
    return f / f.norm(dim=1, keepdim=True).clamp_min(1e-6)


def crop_resize(g, top, left, size):
    c = g[:, :, top:top + size, left:left + size]
    return F.interpolate(c, size=32, mode="bilinear", align_corners=False, antialias=True)


TRANSFORMS = {"identity": lambda g: g}
for size in (29, 26, 22, 19):                      # ~90 / 80 / 70 / 60 % center zoom
    o = (32 - size) // 2
    TRANSFORMS[f"zoom{size}"] = lambda g, o=o, s=size: crop_resize(g, o, o, s)
for top, left in ((0, 0), (0, 4), (4, 0), (4, 4)):  # 28x28 corner crops: shift + slight zoom
    TRANSFORMS[f"corner{top}{left}"] = lambda g, t=top, l=left: crop_resize(g, t, l, 28)
for name in list(TRANSFORMS):
    TRANSFORMS[name + "+flip"] = lambda g, fn=TRANSFORMS[name]: fn(g).flip(3)


def family(name):
    base = name.replace("+flip", "")
    fam = "identity" if base == "identity" else base.rstrip("0123456789")
    return fam + ("+flip" if "+flip" in name else "")


def search(gq, ref, exclude_self):
    """Top-K matches in `ref` for every query over all transforms -> (sim, idx, transform), each (N, K)."""
    n = len(gq)
    best_s = torch.full((n, K), -2.0)
    best_i = torch.full((n, K), -1, dtype=torch.long)
    best_t = np.full((n, K), "", dtype=object)
    earlier = torch.tril(torch.ones(2 * K, 2 * K, dtype=torch.bool), -1)
    for name, fn in TRANSFORMS.items():
        fq = thumb(fn(gq))
        for i in range(0, n, 2000):
            s = fq[i:i + 2000] @ ref.T
            if exclude_self:
                s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -2
            v, j = s.topk(K, dim=1)
            sl = slice(i, i + len(s))
            cs, ci = torch.cat([best_s[sl], v], 1), torch.cat([best_i[sl], j], 1)
            ct = np.concatenate([best_t[sl], np.full(v.shape, name, dtype=object)], 1)
            order = cs.argsort(1, descending=True)
            cs, ci, ct = cs.gather(1, order), ci.gather(1, order), np.take_along_axis(ct, order.numpy(), 1)
            repeat = ((ci[:, :, None] == ci[:, None, :]) & earlier).any(2)     # same index seen at a higher score
            cs = cs.masked_fill(repeat, -3)
            order = cs.argsort(1, descending=True)[:, :K]
            best_s[sl], best_i[sl] = cs.gather(1, order), ci.gather(1, order)
            best_t[sl] = np.take_along_axis(ct, order.numpy(), 1)
    print(f"  search done ({time.time() - t0:.0f}s)", flush=True)
    return best_s.numpy(), best_i.numpy(), best_t


f_tr, f_te = thumb(g_tr), thumb(g_te)
print("coarse search: train -> train")
s_tt, i_tt, t_tt = search(g_tr, f_tr, exclude_self=True)
print("coarse search: test -> train")
s_te, i_te, t_te = search(g_te, f_tr, exclude_self=False)
print("coarse search: test -> test")
s_ee, i_ee, t_ee = search(g_te, f_te, exclude_self=True)


# --- Stage 2: edge verification ------------------------------------------------
def edges(g):
    g = F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), 3, 1)
    e = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), LAP)[:, :, 2:-2, 2:-2]
    return e - e.mean(3, keepdim=True)


def ncc(a, b):
    a, b = a.flatten(1), b.flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-6)).item()


def edge_score(ga, gb, t):
    """The coarse transform was applied to whichever image was the query: try both directions."""
    fn = TRANSFORMS[t]
    return max(ncc(edges(fn(ga)), edges(gb)), ncc(edges(fn(gb)), edges(ga)))


pairs = {}
for a, c in zip(*np.where(s_tt >= CAND)):
    k = tuple(sorted((int(a), int(i_tt[a, c]))))
    if k not in pairs or s_tt[a, c] > pairs[k][0]:
        pairs[k] = (float(s_tt[a, c]), t_tt[a, c])
pair_rows = []
for (a, b), (s, t) in pairs.items():
    e = edge_score(g_tr[[a]], g_tr[[b]], t) if s >= COARSE else float("nan")
    pair_rows.append((a, b, s, t, e))
with open(OUT / "dup_pairs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["a", "b", "coarse", "transform", "edge", "label_a", "label_b", "duplicate"])
    for a, b, s, t, e in sorted(pair_rows, key=lambda r: -r[2]):
        dup = s >= COARSE and e >= EDGE and y_tr[a] == y_tr[b]
        w.writerow([a, b, f"{s:.4f}", t, f"{e:.3f}", CLASSES[y_tr[a]], CLASSES[y_tr[b]], int(dup)])
dups = [(a, b, s, t, e) for a, b, s, t, e in pair_rows if s >= COARSE and e >= EDGE and y_tr[a] == y_tr[b]]

parent = np.arange(len(y_tr))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def make_groups(links):
    parent[:] = np.arange(len(y_tr))
    for a, b, *_ in links:
        parent[find(a)] = find(b)
    return np.array([find(a) for a in range(len(y_tr))])


loose = [r for r in pair_rows if r[2] >= COARSE and r[4] >= EDGE_LOOSE and y_tr[r[0]] == y_tr[r[1]]]
g_loose = make_groups(loose)
np.save(ROOT / "data" / "dup_groups_loose.npy", g_loose)
groups = make_groups(dups)
np.save(ROOT / "data" / "dup_groups.npy", groups)
gid, gsize = np.unique(groups, return_counts=True)
multi = gsize > 1


def verify_test(s, idx, t, ref_g, name):
    out, seen = [], set()
    for a, c in zip(*np.where(s >= CAND)):
        b = int(idx[a, c])
        if name.endswith("test.csv") and (b, int(a)) in seen:      # test-test pairs appear twice
            continue
        seen.add((int(a), b))
        e = edge_score(g_te[[a]], ref_g[[b]], t[a, c])
        out.append((int(a), b, float(s[a, c]), t[a, c], e, bool(e >= EDGE)))
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_index", "match_index", "coarse", "transform", "edge", "flagged"])
        for r in sorted(out, key=lambda r: -r[2]):
            w.writerow([r[0], r[1], f"{r[2]:.4f}", r[3], f"{r[4]:.3f}", int(r[5])])
    return out


# test images have no labels, so the label check is unavailable; the coarse bar is relaxed to 0.90
# (zoomed copies score lower) and every flagged test pair is meant to be confirmed by eye
te_tr = verify_test(s_te, i_te, t_te, g_tr, "dup_test_train.csv")
te_te = verify_test(s_ee, i_ee, t_ee, g_te, "dup_test_test.csv")

print(f"\n=== train duplicates (coarse >= {COARSE}, edge >= {EDGE}, same label)")
fam = {}
for *_, t, _ in dups:
    fam[family(t)] = fam.get(family(t), 0) + 1
gl, gls = np.unique(g_loose, return_counts=True)
print(f"  loose rule (edge >= {EDGE_LOOSE}) for splitting: groups {int((gls > 1).sum())}, "
      f"images in a group {int(gls[gls > 1].sum())} ({gls[gls > 1].sum() / len(y_tr):.2%}), largest {gls.max()}")
print(f"  pairs {len(dups)} -> groups {int(multi.sum())}, images in a group {int(gsize[multi].sum())} "
      f"({gsize[multi].sum() / len(y_tr):.2%}), largest group {gsize.max()}")
print("  by transform:", dict(sorted(fam.items(), key=lambda kv: -kv[1])))
diff = [r for r in pair_rows if r[2] >= COARSE and y_tr[r[0]] != y_tr[r[1]]]
print(f"  different-label candidates (coarse >= {COARSE}): {len(diff)}, of which pass the edge check: "
      f"{sum(r[4] >= EDGE for r in diff)} (max edge {max(r[4] for r in diff):.2f})")
for name, rows in (("test -> train", te_tr), ("test -> test", te_te)):
    flagged = [r for r in rows if r[5]]
    print(f"=== {name}: candidates (coarse >= {CAND}) {len(rows)}, flagged {len(flagged)}")
    for r in flagged:
        print(f"    test {r[0]:4d} ~ {r[1]:5d}  coarse {r[2]:.3f}  edge {r[4]:.2f}  {r[3]}")


# --- Figures --------------------------------------------------------------------
def pair_grid(rows, title, name, n=6):
    fig, axes = plt.subplots(len(rows), 2 * n, figsize=(2 * n * 1.15, len(rows) * 1.6))
    axes = np.atleast_2d(axes)
    for r, (label, items) in enumerate(rows):
        for k in range(n):
            for c in range(2):
                ax = axes[r, 2 * k + c]; ax.axis("off")
                if k < len(items):
                    img, cap = items[k][c]
                    ax.imshow(img, interpolation="nearest"); ax.set_title(cap, fontsize=6)
        axes[r, 0].text(-0.15, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0.1, 0, 1, 1))
    save(fig, name)


def tr(i):
    return x_tr[i], f"{CLASSES[y_tr[i]][:5]} {i}"


# largest groups, shown as the verified links that formed them (strongest link first)
links = {}
for a, b, s, t, e in dups:
    links.setdefault(groups[a], []).append((e, a, b, t))
big = gid[np.argsort(-gsize)][:8]
pair_grid([(f"group of {int((groups == g).sum())}",
            [((x_tr[a], f"{CLASSES[y_tr[a]][:5]} {a}"), (x_tr[b], f"{b} e={e:.2f} {t}"))
             for e, a, b, t in sorted(links[g], reverse=True)[:6]]) for g in big],
          "Largest train duplicate groups: each pair is one verified link (strongest first)", "dedup_groups.png")


def test_rows(rows, ref_x, ref_cap):
    rows = sorted(rows, key=lambda r: (-r[5], -r[4]))[:24]
    return [(f"{'flagged' if rows[i * 6][5] else 'candidates'}\n{i * 6 + 1}-{i * 6 + 6}",
             [((x_te[r[0]], f"test {r[0]}"), (ref_x[r[1]], f"{ref_cap(r[1])} c={r[2]:.2f} e={r[4]:.2f}"))
              for r in rows[i * 6:(i + 1) * 6]]) for i in range(int(np.ceil(len(rows) / 6)))]


pair_grid(test_rows(te_tr, x_tr, lambda i: f"{CLASSES[y_tr[i]][:5]} {i}"),
          "Test -> closest train image (flagged pairs first, then highest edge score)", "dedup_test_train.png")
pair_grid(test_rows(te_te, x_te, lambda i: f"test {i}"),
          "Test -> closest other test image (flagged pairs first)", "dedup_test_test.png")

border = sorted([r for r in pair_rows if r[2] >= COARSE and y_tr[r[0]] == y_tr[r[1]]
                 and EDGE - 0.15 <= r[4] < EDGE + 0.15], key=lambda r: r[4])
pick = [border[i] for i in np.linspace(0, len(border) - 1, min(24, len(border))).astype(int)]
pair_grid([(f"edge {pick[i * 6][4]:.2f}-{pick[min(i * 6 + 5, len(pick) - 1)][4]:.2f}",
            [((x_tr[a], f"{CLASSES[y_tr[a]][:5]} {a}"), (x_tr[b], f"e={e:.2f} {t}")) for a, b, s, t, e in pick[i * 6:(i + 1) * 6]])
           for i in range(int(np.ceil(len(pick) / 6)))],
          f"Same-label pairs near the edge threshold ({EDGE}): below = not counted, above = duplicate",
          "dedup_borderline.png")
print(f"\ndone in {time.time() - t0:.0f}s")
