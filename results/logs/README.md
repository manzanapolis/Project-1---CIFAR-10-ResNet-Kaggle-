# Per-epoch training logs

One CSV per run, named `<model>_<recipe>_<seed>_<split>[_<submission>]__<config hash>.csv`.
The hash is the run directory in `runs/` / Drive `cifar/runs/`; its `config.json` has every setting.

Columns:
- `epoch`: 1-based.
- `lr`: learning rate at the end of the epoch (cosine, stepped per batch).
- `loss`: mean training loss over the epoch, on augmented images (CutMix batches use the mixed loss; long runs include label smoothing).
- `train_acc`: training accuracy against each image's primary label. CutMix batches count only the first label, so this understates real training accuracy.
- `heldout_acc`: accuracy on the held-out fold, single view (no TTA), recorded every 10 epochs and at the last epoch. Long runs evaluate the EMA weights. Empty for all-50k runs, which have no held-out data.
- `seconds`: cumulative training time.

Notes:
- Runs resumed from a checkpoint re-logged a few epochs; the files keep the last entry per epoch (the one the final model followed).
- `M3E_long_seed0_fold0` has no row for epoch 410 (lost in the crash before its resume).
- "long" = 600-epoch recipe; "T1B" = 200-epoch run with repeated augmentation. Fold-0 runs share held-out images, so their curves are directly comparable.
