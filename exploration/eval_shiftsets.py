"""Accuracy of trained models on the robustness sets (src/shiftsets.py), CPU.

Uses a stratified 1000-image subset of fold 0 (all weights here are fold-0 models, so these are held-out),
single view (no TTA) to keep CPU time under ~2 min per model. Weights: weights/<hash>_<name>_model.pt, config re-derived from experiment.py by hash.
  python exploration/eval_shiftsets.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import inference as I  # noqa: E402
from src.launch import expand  # noqa: E402
from src.load import config_hash  # noqa: E402
from src.model import build  # noqa: E402
from src.shiftsets import build as build_sets  # noqa: E402
import experiment as E  # noqa: E402

torch.set_num_threads(max(1, torch.get_num_threads()))
cfgs = {config_hash(c): c for e in E.EXPERIMENTS for _, c in expand(e)}

d = np.load(ROOT / "data" / "cache_uint8.npz")
x, y = d["x_train"], d["y_train"]
folds = np.load(ROOT / "data" / "folds.npy")
rng = np.random.default_rng(0)
idx = np.sort(np.concatenate([rng.choice(np.where((folds == 0) & (y == c))[0], 100, replace=False) for c in range(10)]))
sets = build_sets(x[idx], idx, y[idx])
yy = y[idx]

rows = []
for w in sorted((ROOT / "weights").glob("*_model.pt")):
    h, name = w.name.split("_", 1)
    model = build(cfgs[h]["model"])
    model.load_state_dict(torch.load(w, map_location="cpu"))
    t0 = time.time()
    acc = {}
    for s, xs in sets.items():
        p = I.predict_views_subset(model, xs, "cpu", views=[0, 1]) if hasattr(I, "predict_views_subset") else None
        if p is None:
            model.eval()
            with torch.no_grad():
                xt = I.T.normalize(I.T.to_float(torch.from_numpy(xs)))
                p = sum(torch.softmax(model(I.view(xt, 0, 0, f)), 1) for f in (False,)).numpy()
        acc[s] = (p.argmax(1) == yy).mean()
    rows.append((name.replace("_model.pt", ""), acc))
    print(f"{name:40s} {time.time() - t0:5.0f}s", flush=True)

names = list(sets)
print("\n" + f"{'model':28s}" + "".join(f"{n:>9s}" for n in names))
for name, acc in rows:
    print(f"{name:28s}" + "".join(f"{100 * acc[n]:9.1f}" for n in names))
print(f"{'drop vs clean (mean)':28s}" + "".join(f"{100 * np.mean([a['clean'] - a[n] for _, a in rows]):9.1f}" for n in names))
