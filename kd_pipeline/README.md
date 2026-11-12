# Distilled M3E student (public 0.912)

Single ResNet, **4,930,530 trainable parameters** (limit 5M), trained from scratch on all 50,000 training images.
It uses the M3E shape from this repo (channels 88/128/216, blocks 4/4/4) with the distillation recipe of the other pipeline.
Kaggle public score 0.91200 (`submissions/sub12_kd_M3E_wrn28-10_sam600_foldfinal_seed0_flip.csv`, logged in `submissions.csv`).
It is the second-best of our final candidates; the final submission is `sub10`.

| | |
|---|---|
| Student | ResNet, channels (88,128,216), blocks (4,4,4), basic blocks, 4,930,530 params |
| Data | all 50k training images (no validation split), TrivialAugment + CutMix (p 0.5, alpha 1.0) + Cutout 16, repeated augmentation off |
| Optimiser | SGD lr 0.1 (cosine), weight decay 5e-4, batch 128, label smoothing 0.1, SAM rho 0.05, EMA 0.999, 600 epochs, seed 0 |
| Distillation | KL to a WRN-28-10 teacher, temperature 4, weight 0.7 on the distillation term |
| Inference | EMA weights, horizontal-flip TTA |
| Teacher | WRN-28-10 (36.5M params), trained from scratch on the same 50k, 300 epochs, SAM 0.05, EMA 0.999, TrivialAugment + CutMix. It is only used during training and is not part of the submitted model. |

No pretrained weights or external data are used. Test images are only used for the final prediction.

## Files

- `train.py`: self-contained training / distillation / prediction script (this pipeline is separate from `src/`).
- `train.sbatch`: Slurm wrapper.
- `checkpoint/model.pt`: weights of the submitted student (19.8 MB, strict-loadable into `ResNet` in `train.py`); `checkpoint/history.json`: arguments and per-epoch log.
  The folder is not called `weights/` because that name is in `.gitignore`.
  The teacher checkpoint (146 MB) exceeds GitHub's 100 MB limit and is not included; it is reproduced by step 1.

## Reproduce

Place the Kaggle files in `data/` (`train_labels.csv`, `test.csv`, `train_images/`, `test_images/`) and run from this folder:

```
# 1. teacher (writes runs/teacher_wrn_full_sam/model.pt)
python train.py --tag teacher_wrn_full_sam --arch wrn28_10 --epochs 300 --val-size 0 \
    --aug trivial --mix cutmix --ema 0.999 --sam 0.05

# 2. student (writes runs/full_kd2_m3e_sam600/{model.pt,submission.csv,history.json})
python train.py --tag full_kd2_m3e_sam600 --channels 88 128 216 --blocks 4 4 4 --epochs 600 --val-size 0 \
    --aug trivial --mix cutmix --ema 0.999 --sam 0.05 \
    --teacher runs/teacher_wrn_full_sam/model.pt --kd-alpha 0.7 --kd-T 4
```

`train.py` prints the parameter count and, with a validation split, the final accuracy. GPU kernels are not bit-exact, so a rerun matches
the submission to within the seed-to-seed range, not exactly.
