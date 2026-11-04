# CIFAR-10 ResNet under 5M parameters (ECE7123 Project 1)

Design decisions, with evidence and sources, are in `decisions.md`; the test-set provenance notes are in `data_provenance.md`.

## Layout

| Path | Stage | Contents |
|---|---|---|
| `src/config.py` | all | constants and data-layer defaults (row ids refer to `decisions.md`) |
| `src/extract.py` | E1, E3 | verify the Kaggle files, decode them into `data/cache_uint8.npz` |
| `exploration/dedup_transforms.py` | E2 | transform-invariant duplicate search → `data/dup_groups*.npy` |
| `src/folds.py` | L1 | stratified, grouped 5-fold split → `data/folds.npy` |
| `src/transform.py` | T1–T7 | GPU augmentation, normalization, Cutout, CutMix |
| `src/load.py` | L1–L5 | splits, seeded generators, training batch stream, run registry |
| `src/inference.py` | I1–I3 | TTA prediction, shift-mimic evaluation set, submission writing/validation/log |
| `exploration/` | — | data analysis scripts and figures |

## Data setup

1. Download the competition data from Kaggle and unzip it into `data/` (`train_labels.csv`, `test.csv`, `sample_submission.csv`, `train_images/`, `test_images/`).
2. `python -m src.extract` verifies the files and builds the cache (~5 min on a laptop CPU).
3. `data/dup_groups*.npy` and `data/folds.npy` are committed; to regenerate them: `python exploration/dedup_transforms.py`, then `python -m src.folds`.
