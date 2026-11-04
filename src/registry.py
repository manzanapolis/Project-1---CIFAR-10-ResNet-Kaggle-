"""Results table of every finished run (L5 registry): results/runs.csv, one row per run directory.

  python -m src.registry            pull finished runs from Drive into runs/, then rebuild the table
  python -m src.registry --no-pull  rebuild from runs/ only

This machine's runs/ is the home of all run outputs (weights included). Drive cifar/runs/ is only Colab's
working area and the transport back here; checkpoints of unfinished runs are not copied. The table is small
and committed to git, so results are versioned with the code.
"""
import csv
import json
import shutil
import subprocess
from pathlib import Path
import sys

from .config import ROOT, RUNS_DIR

DRIVE_RUNS = "gdrive:cifar/runs"
RCLONE = shutil.which("rclone") or str(Path.home() / "anaconda3" / "envs" / "d2l" / "bin" / "rclone.exe")
COLUMNS = ["hash", "arms", "fold", "params", "heldout_acc_flip", "heldout_acc_flip+shift", "shift_acc_flip",
           "robust_acc_sketch", "robust_acc_lineart", "robust_acc_poster", "robust_acc_paint", "robust_acc_gray",
           "robust_acc_framed", "robust_acc_distract", "train_seconds", "gpu", "git_commit", "git_dirty"]


def pull():
    cmd = [RCLONE, "copy", DRIVE_RUNS, str(RUNS_DIR), "--exclude", "ckpt.pt"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def rows():
    for d in sorted(p for p in RUNS_DIR.iterdir() if (p / "metrics.json").exists()):
        m = json.loads((d / "metrics.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        arms = (d / "arms.txt").read_text().split() if (d / "arms.txt").exists() else []
        yield {"hash": d.name, "arms": " ".join(arms), "fold": cfg.get("fold"),
               **{k: m.get(k, "") for k in COLUMNS if k not in ("hash", "arms", "fold")}}


def main():
    if "--no-pull" not in sys.argv:
        pull()
    out = ROOT / "results" / "runs.csv"
    out.parent.mkdir(exist_ok=True)
    table = list(rows())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} finished runs -> {out}")


if __name__ == "__main__":
    main()
