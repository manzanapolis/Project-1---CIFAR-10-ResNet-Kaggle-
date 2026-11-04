# Working theory: where this dataset comes from

Running list. Statistics come from `exploration/data_diagnostics.py` and `exploration/integrity_check.py` (statistics only, no trained models).

## Theory

1. **Train = the standard CIFAR-10 training set, unmodified.**
2. **Test = a different, CIFAR-like held-out collection**, not a sample of the standard CIFAR-10 test set. It may be a re-collection in the style of CIFAR-10.1 (Recht et al. 2018). That is unverified, and it must not be verified against external datasets: the rules forbid inferring evaluation labels from external sources.

## Evidence

| # | Observation | Supports | Strength |
|---|---|---|---|
| 1 | Train: 50,000 images, exactly 5,000 per class; per-channel mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616) match the published CIFAR-10 train constants to 4 decimals. | Theory 1 | Strong against large modifications; can't rule out a few hundred swapped images (not verified image-by-image, by choice) |
| 2 | Test brightness/contrast differ from train: median brightness +2%, 95th pct +9%, contrast +4–6%; KS 0.06–0.07 vs a 5% threshold of ~0.031. A random 2,000-image draw from the standard test set (whose stats nearly equal train's) would sit within ~±0.003 of its mean; ours is ~0.02 off (~7 SE). | Theory 2 | Strong |
| 3 | Train↔test power-spectrum ratio is nearly flat (1.07–1.15): a gain change, not a filtering/resampling change; sharpness only +2–3%. | Theory 2 (same capture pipeline, different photos/exposure) | Moderate |
| 4 | Transform-invariant search (flip, zoom, corner crop; brightness/contrast/tint-invariant; verified by horizon-insensitive edge correlation, then by eye): **13 test images (0.65%) have a duplicate in train, confirmed by eye (`exploration/diagnostics/manual_labels.csv`), all mirrored and/or zoomed/cropped, none a plain copy**; a 14th (test 1395) looks like a meme edit of a training airplane image (background cut out, text added). 3 of the 12 automatically flagged pairs were false (different ship, drawing vs photo of a horse, red vs yellow truck), and 4 true pairs were below the automatic cutoff. About 3% of the standard CIFAR-10 test set has near-duplicates in train (Barz & Denzler 2020). | Theory 2. The all-transformed pattern fits a curator who removed plain near-copies against CIFAR-10 but not mirrored/zoomed ones, suggesting the test images come from the same source pool (80 Million Tiny Images), as CIFAR-10.1's did (speculative) | Strong for "not the standard test set"; weak for the source-pool detail |
| 5 | Test set size is exactly 2,000, which matches CIFAR-10.1 (v6). | Theory 2 (re-collection variant) | Weak, circumstantial |
| 6 | Train contains 759 verified duplicate pairs (same label; coarse ≥ 0.95; edge ≥ 0.50; top-5 matches per image) forming 468 groups covering 1,086 images (2.17%; 618 redundant copies, 1.24%): the same photo re-tinted, grayscaled, mirrored, zoomed, cropped, framed or captioned. Largest: **one white Audi ×54** (1.1% of the automobile class), one silver hatchback ×29, an emu face ×11. The rule agrees with 23 of 24 hand-labeled borderline pairs; it still misses some true copies, so these counts are a lower bound. **Mechanism:** CIFAR-10 was assembled by web keyword search (80 Million Tiny Images); popular images recur as reposts. The emu face is an old internet meme, and several copies still show blurry caption text or frames; the repeated cars look like stock photos reused across listings. | Neutral: expected in the standard CIFAR-10 training set | — |
| 7 | Grayscale images (R ≈ G ≈ B everywhere, max channel difference < 0.02): train 582 (1.16%), test 69 (**3.45%**), ~3× the train rate. At the train rate, ~23 of 2,000 would be expected; 69 is ~9.6 SD above that. | Theory 2 (a different collection with more black-and-white photos) | Strong |
| 7b | The test set contains **plain duplicates of itself**: 9 pairs (18 images, 0.9%) confirmed by eye, all untransformed copies (one re-framed). | Theory 2 (the curator deduplicated against CIFAR-10 but not within the new set) | Weak |
| 8 | File-level format is clean: 52,000 PNGs, all RGB mode, all 32×32, none unreadable; CSV ids are contiguous (train 0–49,999, test 0–1,999), filenames match ids, files on disk match the CSV rows, `sample_submission.csv` ids are in `test.csv` order. | Neither (rules out a corrupted or re-encoded file pipeline) | — |

## Implications

- Validation accuracy (drawn from train) will **overstate** leaderboard accuracy, but should **rank** variants correctly (Recht et al.'s finding for re-collected test sets).
- The leaderboard measures genuine generalization: no evidence of train→test leakage so far.
- Photometric augmentation should cover the measured gain shift (decision T3).
- Literature numbers measured on the standard CIFAR-10 test set transfer in ranking, not necessarily in size.
- About 3.5% of test images carry no color at all, ~3× the training rate, so the color cue fails on them. This contradicts the reason random grayscale was rejected (decision T3A: "the color cue holds on test") and reopens that decision.
- The validation split must keep duplicate groups together, including transformed copies, or validation partly scores memorized images.

## Open checks

- [x] Transform-invariant duplicate search (flip, zoom/crop, shift, color/tint): within train and test↔train (`exploration/dedup_transforms.py`, verification by edge correlation; groups in `data/dup_groups.npy`).
- [x] Near-duplicates within the test set: 9 pairs confirmed by eye (evidence 7b).
