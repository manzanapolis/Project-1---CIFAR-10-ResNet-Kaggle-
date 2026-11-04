"""Expand an experiment from experiment.py into runs (L5): one config per arm x fold x seed.

Usage:
  python -m src.launch <experiment> [--pack N] [--local]

Writes runs/<hash>/config.json for every run (the hash ignores arm names, so identical configs in
different experiments are trained once), runs/<hash>/arms.txt naming the arms that use it, and
jobs/<experiment>.txt listing the runs not yet done. Prints the sbatch command; with --local it
runs them here instead.
"""
import json
import math
import sys

from . import load as L
from .config import ROOT, RUNS_DIR
from .train import default_config, run

sys.path.insert(0, str(ROOT))
import experiment as E  # noqa: E402


def expand(name):
    runs = []
    for arm, over in E.EXPERIMENTS[name].items():
        for fold in getattr(E, "EXPERIMENT_FOLDS", {}).get(name, E.FOLDS):
            for seed in E.SEEDS:
                cfg = default_config(fold=fold, seed=seed, **json.loads(json.dumps(over)))
                runs.append((arm, cfg))
    return runs


def main():
    name = sys.argv[1]
    pack = int(sys.argv[sys.argv.index("--pack") + 1]) if "--pack" in sys.argv else 2
    pending = []
    for arm, cfg in expand(name):
        d = L.run_dir(cfg)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        tag = f"{name}/{arm}/fold{cfg['fold']}/seed{cfg['seed']}"
        arms = (d / "arms.txt").read_text().split() if (d / "arms.txt").exists() else []
        if tag not in arms:
            (d / "arms.txt").write_text("\n".join(arms + [tag]) + "\n")
        if not L.is_done(cfg) and d.name not in pending:
            pending.append(d.name)
    jobs = ROOT / "jobs"
    jobs.mkdir(exist_ok=True)
    (jobs / f"{name}.txt").write_text("\n".join(pending) + ("\n" if pending else ""))
    print(f"{name}: {len(pending)} runs to do ({len(expand(name))} total) -> jobs/{name}.txt")
    if not pending:
        return
    if "--local" in sys.argv:
        for h in pending:
            run(json.loads((RUNS_DIR / h / "config.json").read_text()), RUNS_DIR / h)
    else:
        tasks = math.ceil(len(pending) / pack)
        print(f"sbatch --array=0-{tasks - 1} slurm/array.sbatch jobs/{name}.txt {pack}")


if __name__ == "__main__":
    main()
