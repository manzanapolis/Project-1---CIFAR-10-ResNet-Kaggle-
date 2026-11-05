"""Build, validate, log and submit a Kaggle submission from one run (I1-I3).

  python -m src.submit <run_hash> -m "message"            write + validate + log + submit via the kaggle CLI
  python -m src.submit <run_hash> -m "message" --dry-run  write + validate only

Predictions come from runs/<hash>/preds.npz (test probabilities for all 10 TTA views); the TTA mode is
flip+shift (I2B). The kaggle CLI needs ~/.kaggle/kaggle.json. After submitting, the public score is read
back with `kaggle competitions submissions` and written into submissions.csv.
"""
import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from . import inference as I
from .config import DATA_DIR, ROOT, RUNS_DIR

COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
# kaggle CLI >= 2 (reads KGAT_ tokens) lives in the conda env "tools" (Python 3.12); see README one-time setup
KAGGLE = str(Path.home() / "anaconda3" / "envs" / "tools" / "Scripts" / "kaggle.exe")


def kaggle_env():
    """Process env plus KAGGLE_API_TOKEN from the Windows user environment (setx) if this process predates it."""
    env = dict(os.environ)
    if not env.get("KAGGLE_API_TOKEN") and os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            try:
                env["KAGGLE_API_TOKEN"] = winreg.QueryValueEx(k, "KAGGLE_API_TOKEN")[0]
            except FileNotFoundError:
                pass
    return env


def build(run_hash, tta="flip+shift", dry_run=False):
    d = RUNS_DIR / run_hash
    probs = np.load(d / "preds.npz")["test"]
    labels = I.tta(probs, tta).argmax(1)
    ids = np.load(DATA_DIR / "cache_uint8.npz")["id_test"]
    arms = (d / "arms.txt").read_text().split()[0].replace("/", "_") if (d / "arms.txt").exists() else run_hash
    n = max([int(p.name[3:5]) for p in (ROOT / "submissions").glob("sub*.csv")], default=0) + 1
    name = f"sub{n:02d}_{arms}_{tta.replace('+', '')}.csv"
    path = Path(tempfile.gettempdir()) / name if dry_run else ROOT / "submissions" / name   # dry runs take no number
    I.write_submission(ids, labels, path)                     # validates against sample_submission.csv
    m = json.loads((d / "metrics.json").read_text())
    return path, m, labels


def kaggle(*args):
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True, check=True, env=kaggle_env()).stdout


def public_score(file_name, tries=20):
    for _ in range(tries):
        out = kaggle("competitions", "submissions", "-c", COMPETITION, "--csv")
        for row in csv.DictReader(out.splitlines()):
            if row.get("fileName") == file_name and row.get("publicScore"):
                return row["publicScore"]
        time.sleep(15)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_hash")
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--tta", default="flip+shift", choices=list(I.TTA))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    path, m, labels = build(a.run_hash, tta=a.tta, dry_run=a.dry_run)
    fold = json.loads((RUNS_DIR / a.run_hash / "config.json").read_text())["fold"]
    print(f"{path.name}: counts {np.bincount(labels, minlength=10).tolist()}, held-out {m.get(f'heldout_acc_{a.tta}')}")
    if a.dry_run:
        return
    print(kaggle("competitions", "submit", "-c", COMPETITION, "-f", str(path), "-m", a.message))
    score = public_score(path.name)
    print("public score:", score or "not available yet")
    I.log_submission(path, a.run_hash, fold, m.get(f"heldout_acc_{a.tta}", ""), score)


if __name__ == "__main__":
    main()
