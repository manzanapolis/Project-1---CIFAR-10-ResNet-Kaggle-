"""Inference (I1-I3): deterministic prediction over fixed TTA views, the shift-mimic evaluation set,
and submission writing / validation / logging.

  I1  test images: normalization only, model.eval(), no training augmentation
  I2  fixed views only: "flip" = identity + mirror; "flip+shift" adds +-2 px reflect-padded shifts
  I3  every submission validated against sample_submission.csv and logged with the run behind it
"""
import csv
import datetime
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from . import transform as T
from .config import DATA_DIR, GRAY, ROOT

SHIFTS = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]
VIEWS = [(dy, dx, flip) for dy, dx in SHIFTS for flip in (False, True)]     # view 0 = plain, 1 = mirror
TTA = {"none": [0], "flip": [0, 1], "flip+shift": list(range(len(VIEWS)))}


def view(x, dy, dx, flip, pad=2):
    if dy or dx:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")[:, :, pad + dy:pad + dy + 32, pad + dx:pad + dx + 32]
    return x.flip(3) if flip else x


@torch.no_grad()
def predict_views(model, x_u8, device, batch=1000):
    """Class probabilities for every image under every view: (N, len(VIEWS), 10) float16 numpy."""
    model.eval()
    device = torch.device(device)
    out = []
    for i in range(0, len(x_u8), batch):
        x = T.normalize(T.to_float(torch.from_numpy(x_u8[i:i + batch]).to(device)))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            probs = [F.softmax(model(view(x, *v)).float(), 1) for v in VIEWS]
        out.append(torch.stack(probs, 1).half().cpu())
    return torch.cat(out).numpy()


def tta(probs, mode):
    """Average stored per-view probabilities over the views of a TTA mode -> (N, 10)."""
    return probs[:, TTA[mode]].astype(np.float32).mean(1)


# --- Shift-mimic evaluation set --------------------------------------------------
# Held-out images transformed toward the measured test-set pixel shift (data_provenance.md):
# median brightness +2%, contrast +5%, ~3.5% grayscale. Designed from unlabeled test statistics;
# it measures robustness to the pixel-level part of the shift only, not the full test distribution.
SHIFT_BRIGHTNESS, SHIFT_CONTRAST, SHIFT_GRAY_FRAC = 1.02, 1.05, 0.035


def shift_mimic(x_u8, global_idx):
    """Deterministic per image: the same training image gets the same treatment in every fold/run."""
    x = x_u8.astype(np.float32) / 255
    g = (x * np.array(GRAY, np.float32)).sum(-1, keepdims=True)
    m = g.mean((1, 2), keepdims=True)
    x = np.clip(((x * SHIFT_BRIGHTNESS) - m * SHIFT_BRIGHTNESS) * SHIFT_CONTRAST + m * SHIFT_BRIGHTNESS, 0, 1)
    gray = np.random.default_rng(12345).random(50_000)[global_idx] < SHIFT_GRAY_FRAC
    gx = (x * np.array(GRAY, np.float32)).sum(-1, keepdims=True)
    x = np.where(gray[:, None, None, None], np.repeat(gx, 3, -1), x)
    return (x * 255 + 0.5).astype(np.uint8)


# --- Submission (I3) -------------------------------------------------------------
def write_submission(ids, labels, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(zip(map(int, ids), map(int, labels)))
    validate_submission(path)


def validate_submission(path):
    with open(DATA_DIR / "sample_submission.csv", newline="") as f:
        expected = [r["id"] for r in csv.DictReader(f)]
    with open(path, newline="") as f:
        reader = csv.reader(f)
        assert next(reader) == ["id", "label"], "header must be id,label"
        rows = list(reader)
    assert len(rows) == len(expected), f"{len(rows)} rows, expected {len(expected)}"
    assert [r[0] for r in rows] == expected, "ids differ from sample_submission.csv order"
    assert all(r[1].isdigit() and 0 <= int(r[1]) <= 9 for r in rows), "labels must be integers 0-9"
    return True


def log_submission(path, run_hash, fold, oof_acc, public_score=""):
    """Append to submissions.csv; fill public_score in by hand after uploading."""
    log = ROOT / "submissions.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "file", "file_sha1", "run_hash", "fold", "oof_acc", "public_score"])
        w.writerow([datetime.date.today().isoformat(), path.name,
                    hashlib.sha1(path.read_bytes()).hexdigest()[:12], run_hash, fold, oof_acc, public_score])
