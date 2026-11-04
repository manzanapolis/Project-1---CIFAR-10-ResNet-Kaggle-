"""Visual check of every Transform variant: the same 10 training images through each pipeline.

Usage: python -m exploration.show_augmentations   (CPU, a few seconds; writes exploration/diagnostics/augmentations.png)
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import load as L
from src import transform as T
from src.config import DATA, MEAN, STD, ROOT

N = 10
VARIANTS = {
    "original": None,
    "T1+T5 flip, crop": dict(photometric="none", occlusion="none"),
    "T3A jitter": dict(occlusion="none"),
    "T3B jitter+PCA": dict(photometric="jitter+pca", occlusion="none"),
    "T3C TrivialAugment": dict(photometric="trivialaugment", occlusion="none"),
    "T6A CutMix": dict(photometric="none", occlusion="cutmix", cutmix_p=1.0),
    "T6B Cutout": dict(photometric="none", occlusion="cutout"),
    "T7B grayscale p=0.5*": dict(photometric="none", occlusion="none", grayscale_p=0.5),
}


def denorm(x):
    x = x * torch.tensor(STD).view(1, 3, 1, 1) + torch.tensor(MEAN).view(1, 3, 1, 1)
    return x.clamp(0, 1).permute(0, 2, 3, 1).numpy()


def main():
    d = L.load_cache()
    idx = np.random.default_rng(3).choice(len(d["y_train"]), 512, replace=False)
    fig, axes = plt.subplots(len(VARIANTS), N, figsize=(N * 1.1, len(VARIANTS) * 1.25))
    for r, (name, over) in enumerate(VARIANTS.items()):
        if over is None:
            imgs, lam = d["x_train"][idx[:N]] / 255, None
        else:
            s = L.TrainStream(d["x_train"], d["y_train"], idx, {**DATA, **over}, 512, 10, 0, 0, "cpu", workers=0)
            s._batches = lambda: torch.arange(512).view(1, 512)          # fixed order: show the same images
            x, _, _, lam = next(iter(s.epoch(0)))
            imgs = denorm(x[:N])
        for k in range(N):
            axes[r, k].imshow(imgs[k], interpolation="nearest"); axes[r, k].axis("off")
        axes[r, 0].set_title(name + (f"  (kept {lam:.2f})" if lam not in (None, 1.0) else ""),
                             fontsize=8, loc="left")
    fig.suptitle("Same 10 training images through each Transform variant (* p raised from 0.1 for visibility)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "exploration" / "diagnostics" / "augmentations.png", dpi=120)


if __name__ == "__main__":
    main()
