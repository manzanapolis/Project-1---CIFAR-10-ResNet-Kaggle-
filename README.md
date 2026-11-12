# CIFAR-10 ResNet under 5M parameters (ECE7123 Project 1)

Design decisions, with evidence and sources, are in `decisions.md`; the test-set provenance notes are in `data_provenance.md`.

## Layout

| Path | Stage | Contents |
|---|---|---|
| `src/config.py` | all | constants and data-layer defaults (row ids refer to `decisions.md`) |
| `src/extract.py` | E1, E3 | verify the Kaggle files, decode them into `data/cache_uint8.npz` |
| `exploration/dedup_transforms.py` | E2 | transform-invariant duplicate search → `data/dup_groups*.npy` |
| `src/folds.py` | L1 | stratified, grouped 5-fold split → `data/folds.npy` |
| `src/transform.py` | T1–T7 | GPU augmentation (incl. renditions), normalization, Cutout, CutMix |
| `src/load.py` | L1–L5 | splits, seeded generators, training batch stream, run registry |
| `src/shiftsets.py` | L6 | robustness evaluation sets (depiction + distractor) from held-out images |
| `src/model.py` | M1–M3 | configurable ResNet in the spec's design variables |
| `src/train.py` | R1 | one training run: logs, checkpoints/resume, final evaluation, metrics with provenance |
| `src/inference.py` | I1–I3 | TTA prediction, shift-mimic evaluation set, submission writing/validation/log |
| `src/launch.py`, `slurm/` | L5 | expand an experiment into runs; Slurm array script for NYU Burst |
| `src/registry.py` | L5 | pull run outputs from Drive, build `results/runs.csv` |
| `src/submit.py` | I3 | build + validate + log + submit a Kaggle submission from one run |
| `experiment.py` | — | every experiment (arms × folds) ever run |
| `colab/run.ipynb` | — | run an experiment on a Colab GPU from a pinned commit |
| `exploration/` | — | data analysis scripts and figures |
| `kd_pipeline/` | — | distilled M3E student (public 0.912): self-contained `train.py`, checkpoint, recipe (see its README) |

## Where things live

This project folder (on the laptop) is the home base: every file of record lives here.

| What | Home (source of truth) | Other copies |
|---|---|---|
| Code, configs, decisions, notebooks | this folder, versioned in git | GitHub = backup + what compute machines clone (pinned commit) |
| Data cache `cache_uint8.npz` (162 MB, immutable) | `data/` here | Drive `cifar/data/` for Colab; checksum in `data/SHA256SUMS` |
| Run outputs `runs/<config-hash>/` (weights, preds, metrics, logs) | `runs/` here | Drive `cifar/runs/` = Colab's working area; Burst `/scratch` = Burst's; both copied here by `src/registry.py` |
| Results table | `results/runs.csv` (git) | regenerated from `runs/` |
| Submissions | `submissions/*.csv` + `submissions.csv` (git) | every submission tied to a run hash |

Every run directory is named by its config hash and its `metrics.json` records the git commit, GPU and host,
so any number traces back to exact code + config + hardware.

## Standard operating procedure (every experiment)

1. **Decide.** Agree the decision row in chat; commit `decisions.md`.
2. **Define.** Add the experiment to `experiment.py` (arms, and folds via `EXPERIMENT_FOLDS` if not 0–4). Commit and push.
3. **Run.** Open `colab/run.ipynb` in Cursor, set `EXPERIMENT` and `COMMIT`, select the Colab A100 kernel, Run All.
   (Burst alternative: `python -m src.launch <experiment>`, then `sbatch` the printed array plus a retry array with
   `--dependency=afterany:<id>`; the job copies its outputs to Drive.) Never train from uncommitted code.
4. **Monitor.** Notebook output in Cursor (autosaved), or pull the latest `log.csv` with rclone.
5. **Collect.** `python -m src.registry` (copies finished runs from Drive into `runs/` here, rebuilds `results/runs.csv`); analysis scripts live in
   `exploration/`. Commit the table.
6. **Submit.** `python -m src.submit <run_hash> -m "what changed"` (validates, submits with the kaggle CLI, reads the
   public score back, logs it). Commit `submissions/`.
7. **Tear down.** Run the notebook's last cell (releases the Colab runtime); `scancel` any Burst session; close any
   web notebook tabs (they can reconnect and start a new paid runtime).

## One-time setup per machine

- GitHub: `gh auth login` (local); Colab secret `GITHUB_TOKEN` = fine-grained, read-only token for this repo.
- Google Drive: `rclone config create gdrive drive` (write access; used by `src/registry.py`).
- Kaggle: `pip install kaggle`; API token at `~/.kaggle/kaggle.json`.

## Data setup

1. Download the competition data from Kaggle and unzip it into `data/` (`train_labels.csv`, `test.csv`, `sample_submission.csv`, `train_images/`, `test_images/`).
2. `python -m src.extract` verifies the files and builds the cache (~5 min on a laptop CPU).
3. `data/dup_groups*.npy` and `data/folds.npy` are committed; to regenerate them: `python exploration/dedup_transforms.py`, then `python -m src.folds`.
