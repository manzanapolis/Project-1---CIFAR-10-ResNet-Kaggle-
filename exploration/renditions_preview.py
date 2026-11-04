"""Preview of the T7C training renditions (src/transform.py) next to the L6 evaluation versions (src/shiftsets.py).
  python exploration/renditions_preview.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import shiftsets as S  # noqa: E402
from src import transform as T  # noqa: E402

d = np.load(ROOT / "data" / "cache_uint8.npz")
idx = np.array([29, 4, 6, 9, 3, 27, 0, 7, 8, 1])          # one image per class (first train images)
x = d["x_train"][idx]
g = torch.Generator().manual_seed(0)
xt = T.to_float(torch.from_numpy(x))
cols = [("clean", x)]
for k in T.RENDITIONS:
    r = T.render(xt, k, g).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    cols.append((f"train: {k}", (r * 255 + 0.5).astype(np.uint8)))
for k in ("sketch", "poster", "paint", "framed", "lineart"):
    cols.append((f"eval: {k}", S.DEPICTION[k](x, idx)))

fig, axes = plt.subplots(len(idx), len(cols), figsize=(len(cols) * 1.3, len(idx) * 1.35))
for c, (name, imgs) in enumerate(cols):
    for r in range(len(idx)):
        axes[r, c].imshow(imgs[r], interpolation="nearest")
        axes[r, c].axis("off")
    axes[0, c].set_title(name, fontsize=8, fontweight="bold")
plt.tight_layout()
out = ROOT / "exploration" / "diagnostics" / "renditions_preview.png"
fig.savefig(out, dpi=110)
print(out)
