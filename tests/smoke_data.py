"""Smoke test for the data layer (CPU is fine, ~1 min): shapes, value ranges, seeding, common random
numbers across arms, every Transform variant, TTA views, shift-mimic, and the submission validator.

Usage: python -m tests.smoke_data
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from src import inference as I
from src import load as L
from src.config import DATA


def first_batches(cfg, n=2, fold=0, seed=0, epochs=3, epoch=0):
    d = L.load_cache()
    tr, _ = L.split(fold)
    s = L.TrainStream(d["x_train"], d["y_train"], tr[:2048], cfg, 128, epochs, seed, fold, "cpu", workers=2)
    out = []
    for i, b in enumerate(s.epoch(epoch)):
        out.append(b)
        if i + 1 == n:
            break
    return out


def main():
    torch.manual_seed(0)
    variants = {
        "default (jitter + cutmix)": {},
        "none + cutout": dict(photometric="none", occlusion="cutout"),
        "jitter+pca + gray": dict(photometric="jitter+pca", grayscale_p=0.1, occlusion="none"),
        "trivialaugment": dict(photometric="trivialaugment", occlusion="none"),
        "repeats=2": dict(aug_repeats=2),
        "annealed epoch": dict(strong_aug_off_last=1),
    }
    for name, over in variants.items():
        cfg = {**DATA, **over}
        epoch = 2 if over.get("strong_aug_off_last") else 0
        (x, ya, yb, lam), *_ = first_batches(cfg, epoch=epoch)
        exp_b = 128 * cfg["aug_repeats"]
        assert x.shape == (exp_b, 3, 32, 32) and x.dtype == torch.float32, (name, x.shape)
        assert torch.isfinite(x).all() and ya.shape == (exp_b,) and 0 < lam <= 1
        print(f"  ok  {name:28s} batch {tuple(x.shape)}  mean {x.mean():+.3f} std {x.std():.3f}  lam {lam:.2f}")

    # reproducibility (L4) and common random numbers across arms (L5)
    a = first_batches({**DATA, "occlusion": "none"})
    b = first_batches({**DATA, "occlusion": "none"})
    assert all(torch.equal(p[0], q[0]) for p, q in zip(a, b)), "same config + seed must give identical batches"
    c = first_batches({**DATA, "occlusion": "cutout"})
    assert all(torch.equal(p[1], q[1]) for p, q in zip(a, c)), "arms in the same fold must see the same labels/order"
    d = first_batches({**DATA, "occlusion": "none"}, seed=1)
    assert not torch.equal(a[0][1], d[0][1]), "a different seed must change the order"
    print("  ok  reproducible; arms share data order; seeds differ")

    # inference: views, TTA, shift-mimic, submission
    cache = L.load_cache()
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    probs = I.predict_views(model, cache["x_test"][:50], "cpu")
    assert probs.shape == (50, len(I.VIEWS), 10) and np.allclose(probs.astype(np.float32).sum(-1), 1, atol=1e-2)
    for mode in I.TTA:
        assert I.tta(probs, mode).shape == (50, 10)
    tr, va = L.split(0)
    sm = I.shift_mimic(cache["x_train"][va[:500]], va[:500])
    assert sm.dtype == np.uint8 and sm.shape == (500, 32, 32, 3)
    gray = (np.abs(sm[..., 0].astype(int) - sm[..., 1]) <= 1).all((1, 2))
    print(f"  ok  views {probs.shape}; shift-mimic grayscale fraction in sample {gray.mean():.3f}")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "sub.csv"
        I.write_submission(cache["id_test"], np.zeros(len(cache["id_test"]), int), path)
        print("  ok  submission written and validated")
    assert len(tr) == 40_000 and len(va) == 10_000 and len(L.split("final")[0]) == 50_000
    assert L.config_hash({**DATA}) == L.config_hash(dict(reversed(list(DATA.items())))), "hash must ignore key order"
    print("all data-layer smoke tests passed")


if __name__ == "__main__":
    main()
