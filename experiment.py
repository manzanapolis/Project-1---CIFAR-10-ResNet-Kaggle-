"""Experiments: edit this file to define arms. Everything else is boilerplate in src/.

Each experiment is a dict of arm name -> overrides of the defaults in src/config.py
(MODEL, TRAIN, DATA). Every arm runs on every fold in FOLDS and every seed in SEEDS.

  python -m src.launch arch_screen          write configs + job list, print the sbatch command
  python -m src.launch arch_screen --local  run them here, one after another
"""
FOLDS = [0, 1, 2, 3, 4]
EXPERIMENT_FOLDS = {"final_screen": ["final", 0], "kfold_T1B": [1, 2, 3, 4], "long_screen": ["final", 0],
                    "kfold_T1B_L4": [1, 2, 3, 4], "long_screen_L4": ["final", 0],
                    "long_screen_s1": ["final"],
                    "teacher": ["final", 0], "kd_final": ["final"], "kd_fold0": [0]}   # per-experiment override; "final" = all 50k (L2)
SEEDS = [0]
EXPERIMENT_SEEDS = {"kfold_T1B_L4": [1], "long_screen_L4": [1], "long_screen_s1": [1]}   # per-experiment override

# M3: candidate shapes, all ~4.74-4.93M parameters (decisions.md)
SHAPES = {
    "M3A_wide_shallow":   dict(widths=[80, 160, 344],      blocks=[2, 2, 2]),
    "M3B_balanced":       dict(widths=[56, 112, 232],      blocks=[4, 4, 4]),
    "M3C_deep_front":     dict(widths=[48, 96, 216],       blocks=[8, 8, 4]),
    "M3D_four_layers":    dict(widths=[64, 128, 192, 368], blocks=[3, 3, 3, 1]),
    "M3E_high_res_heavy": dict(widths=[88, 128, 216],      blocks=[4, 4, 4]),
    "M3F_deep_two_stage": dict(widths=[64, 128, 256],      blocks=[8, 8, 2]),   # teammate's 91.00 shape
}
# Distillation teacher only (never submitted): WRN-28-10-sized, 36.7M parameters, exempt from the 5M check.
TEACHER = dict(widths=[160, 320, 640], blocks=[4, 4, 4], teacher=True)

EXPERIMENTS = {
    # Network shape, with the data-layer defaults and the R1 recipe held fixed.
    "arch_screen": {name: {"model": shape} for name, shape in SHAPES.items()},

    # Core data factors on the balanced shape: photometric (T3) x occlusion/mixing (T6).
    # Grayscale (T7) and "no occlusion" are left for a later screen.
    "data_screen": {
        f"{ph}-{occ}": {"model": SHAPES["M3B_balanced"], "data": {"photometric": ph, "occlusion": occ}}
        for ph in ["none", "jitter", "trivialaugment"]
        for occ in ["cutout", "cutmix"]
    },

    # Best data arm (trivialaugment-cutmix) on the cheapest tied shape and the best-on-shift shape.
    "combo_screen": {
        f"{s}-trivialaugment-cutmix": {"model": SHAPES[s], "data": {"photometric": "trivialaugment", "occlusion": "cutmix"}}
        for s in ["M3A_wide_shallow", "M3E_high_res_heavy"]
    },

    # Grayscale (T7B) and repeated augmentation (T1B) on top of trivialaugment-cutmix, balanced shape.
    # The baseline arm shares its configs with data_screen, so only its missing folds are trained.
    "gray_repeat_screen": {
        name: {"model": SHAPES["M3B_balanced"], "data": {"photometric": "trivialaugment", "occlusion": "cutmix", **extra}}
        for name, extra in [("baseline", {}), ("gray0.1", {"grayscale_p": 0.1}), ("repeats2", {"aug_repeats": 2})]
    },

    # T7 depiction shift, A/B/C on the submitted recipe (M3E, trivialaugment-cutmix): none / grayscale / renditions.
    # Judged on clean held-out accuracy plus the L6 robustness sets (robust_acc_* in metrics.json).
    # The A arm shares its configs with combo_screen, so only its missing folds are trained.
    "depiction_screen": {
        name: {"model": SHAPES["M3E_high_res_heavy"], "data": {"photometric": "trivialaugment", "occlusion": "cutmix", **extra}}
        for name, extra in [("T7A_none", {}), ("T7B_gray0.1", {"grayscale_p": 0.1}), ("T7C_rendition0.25", {"rendition_p": 0.25})]
    },

    # Round 2 (fold 0 only, compared against the existing M3E trivialaugment-cutmix fold-0 run):
    # T7C, T7B, T1B on the data side; M1B (pre-activation) and M2B (avg-pool shortcut) on the model side.
    # Ordered for early answers: the T7 pair trains first; T1B (2x compute) trains alone last.
    "round2": {
        name: {"model": {**SHAPES["M3E_high_res_heavy"], **model},
               "data": {"photometric": "trivialaugment", "occlusion": "cutmix", **data}}
        for name, model, data in [
            ("T7C_rendition0.25", {}, {"rendition_p": 0.25}),
            ("T7B_gray0.1", {}, {"grayscale_p": 0.1}),
            ("M1B_preact", {"block": "pre"}, {}),
            ("M2B_avgpool", {"shortcut": "avgpool"}, {}),
            ("T1B_repeats2", {}, {"aug_repeats": 2}),
        ]
    },

    # Final fit (L2) on all 50k, plus the same recipes on fold 0 to check the combination on held-out data.
    # T1B (repeated augmentation) was the best clean-accuracy arm; renditions (T7C) were neutral on the leaderboard.
    "final_screen": {
        name: {"model": SHAPES["M3E_high_res_heavy"],
               "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2, **extra}}
        for name, extra in [("T1B", {}), ("T1B_T7C", {"rendition_p": 0.25})]
    },

    # 5-fold CV of the best round-2 arm (fold 0 = round2/T1B_repeats2, same config hash).
    "kfold_T1B": {
        "T1B": {"model": SHAPES["M3E_high_res_heavy"],
                "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2}},
    },

    # R2 long recipe: T1B data + EMA + SAM + label smoothing, 600 epochs, on our M3E and on M3F (8/8/2).
    "long_screen": {
        name: {"model": SHAPES[shape],
               "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2},
               "train": {"epochs": 600, "ema": 0.999, "sam": 0.05, "label_smoothing": 0.1}}
        for name, shape in [("M3E_R2", "M3E_high_res_heavy"), ("M3F_R2", "M3F_deep_two_stage")]
    },

    # L4 copies of the two overnight experiments (A100 nodes kept failing to boot on 11-05), seed 1 so they are
    # separate runs; long recipe cut to 300 epochs so an L4 finishes inside the 12 h limit without resuming.
    "kfold_T1B_L4": {"T1B": {"model": SHAPES["M3E_high_res_heavy"],
                             "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2}}},
    "long_screen_L4": {
        name: {"model": SHAPES[shape],
               "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2},
               "train": {"epochs": 300, "ema": 0.999, "sam": 0.05, "label_smoothing": 0.1}}
        for name, shape in [("M3E_R2_300", "M3E_high_res_heavy"), ("M3F_R2_300", "M3F_deep_two_stage")]
    },

    # Second seed of the 600-epoch long recipe on all 50k (a second final model; same configs as long_screen otherwise).
    "long_screen_s1": {
        name: {"model": SHAPES[shape],
               "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2},
               "train": {"epochs": 600, "ema": 0.999, "sam": 0.05, "label_smoothing": 0.1}}
        for name, shape in [("M3E_R2", "M3E_high_res_heavy"), ("M3F_R2", "M3F_deep_two_stage")]
    },

    # Large teacher for knowledge distillation (allowed: trained from scratch on the permitted data). T1B data + EMA,
    # 200 epochs; no SAM (cost) and no label smoothing (smoothed teachers distil worse, Mueller et al. 2019).
    "teacher": {
        "WRN28_10": {"model": TEACHER,
                     "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2},
                     "train": {"epochs": 200, "ema": 0.999}},
    },

    # Ensemble distillation (R3): M3F student, long recipe, plus a KL term to the averaged soft predictions of our two
    # best models (M3E + M3F, same recipe). alpha 0.7, T 4 (teammate's settings). kd_fold0 uses the fold-0 teachers so
    # the gain can be checked on held-out data; kd_final uses the all-50k teachers (sub07 and sub09).
    "kd_final": {"M3F_KD_ens": {"model": SHAPES["M3F_deep_two_stage"],
                                "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2},
                                "train": {"epochs": 600, "ema": 0.999, "sam": 0.05, "label_smoothing": 0.1,
                                          "kd": {"teachers": ["db5220449ec5", "1a453b9d969a"], "alpha": 0.7, "T": 4.0}}}},
    "kd_fold0": {"M3F_KD_ens": {"model": SHAPES["M3F_deep_two_stage"],
                                "data": {"photometric": "trivialaugment", "occlusion": "cutmix", "aug_repeats": 2},
                                "train": {"epochs": 600, "ema": 0.999, "sam": 0.05, "label_smoothing": 0.1,
                                          "kd": {"teachers": ["558aaaa4e6d7", "3c9415e41118"], "alpha": 0.7, "T": 4.0}}}},
}
