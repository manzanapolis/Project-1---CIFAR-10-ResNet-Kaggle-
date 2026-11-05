"""Train one run (R1 recipe) and save everything later analysis needs.

Usage:
  python -m src.train runs/<hash>/config.json     one run (written by src.launch)
  python -m src.train --smoke                     1 epoch on a small subset, CPU-friendly

A run directory holds:
  config.json   full config (model / train / data / fold / seed); its hash is the directory name
  log.csv       per-epoch lr, loss, train accuracy, time (+ held-out accuracy every monitor_every epochs)
  ckpt.pt       latest checkpoint (resumes automatically if the job is restarted)
  model.pt      final weights
  preds.npz     probabilities for every TTA view: held-out fold, shift-mimic held-out, test set
  metrics.json  written last; its presence marks the run as done
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import inference as I
from . import load as L
from . import transform as T
from .config import DATA, MODEL, RUNS_DIR, TRAIN
from .model import build, summary
from .shiftsets import build as build_sets


def default_config(fold=0, seed=0, **over):
    cfg = {"model": dict(MODEL), "train": dict(TRAIN), "data": dict(DATA), "fold": fold, "seed": seed}
    for k, v in over.items():
        cfg[k].update(v) if isinstance(v, dict) else cfg.__setitem__(k, v)
    return cfg


def provenance(device):
    """Which code and hardware produced a run: git commit (or $CIFAR_COMMIT), dirty flag, GPU, host."""
    import os, platform, subprocess
    root = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()
    try:
        commit, dirty = git("rev-parse", "--short=12", "HEAD"), bool(git("status", "--porcelain", "--", "src", "experiment.py"))
    except OSError:
        commit, dirty = "", None
    return {"git_commit": commit or os.environ.get("CIFAR_COMMIT", "unknown"), "git_dirty": dirty,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "host": platform.node()}


class EMA:
    """Exponential moving average of all weights and BN buffers (R2); evaluation and the saved model use it."""
    def __init__(self, model, decay):
        self.decay, self.shadow = decay, copy.deepcopy(model).eval()
        for q in self.shadow.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src, dst = model.state_dict(), self.shadow.state_dict()
        f_dst = [dst[k] for k in dst if dst[k].dtype.is_floating_point]
        f_src = [src[k].detach() for k in dst if dst[k].dtype.is_floating_point]
        torch._foreach_mul_(f_dst, self.decay)
        torch._foreach_add_(f_dst, f_src, alpha=1 - self.decay)
        for k in dst:
            if not dst[k].dtype.is_floating_point:
                dst[k].copy_(src[k])


def accuracy(probs, y):
    return float((probs.argmax(1) == y).mean()) if len(y) else float("nan")


def run(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    tc, fold, seed = cfg["train"], cfg["fold"], cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    L.seed_everything(L.derive_seed(seed, fold, 9))

    d = L.load_cache()
    tr, va = L.split(fold)
    if tc.get("limit"):                                           # smoke test only
        tr, va = tr[: tc["limit"]], va[: tc["limit"] // 4]
    stream = L.TrainStream(d["x_train"], d["y_train"], tr, cfg["data"], tc["batch"], tc["epochs"],
                           seed, fold, device, workers=tc["workers"])

    model = build(cfg["model"]).to(device).to(memory_format=torch.channels_last)
    n_params = summary(model, device) if not (out / "ckpt.pt").exists() else sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=tc["lr"], momentum=tc["momentum"],
                          nesterov=tc["nesterov"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"] * stream.steps)
    bf16 = tc["bf16"] and device.type == "cuda"
    ema = EMA(model, tc["ema"]) if tc.get("ema") else None        # R2; absent from older configs (hash unchanged)
    rho = tc.get("sam", 0.0)                                      # R2: SAM neighbourhood radius

    start, t_used = 0, 0.0
    if (out / "ckpt.pt").exists():                                # resume after preemption / time limit
        ck = torch.load(out / "ckpt.pt", map_location=device, weights_only=False)   # our own file
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        stream.order_g.set_state(ck["order_g"]); stream.aug_g.set_state(ck["aug_g"])
        stream.aug_np.bit_generator.state = ck["aug_np"]; stream.ta_g.set_state(ck["ta_g"])
        start, t_used = ck["epoch"], ck["time"]
        if ema is not None:
            ema.shadow.load_state_dict(ck["ema"])
        print(f"resumed at epoch {start}")

    log = open(out / "log.csv", "a")
    if start == 0:
        log.write("epoch,lr,loss,train_acc,heldout_acc,seconds\n")
    x_va, y_va = d["x_train"][va], d["y_train"][va]
    for epoch in range(start, tc["epochs"]):
        t0 = time.time()
        model.train()
        tot_loss = tot_correct = tot_n = 0.0
        for x, ya, yb, lam in stream.epoch(epoch):
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                logits = model(x)
                loss = T.mixed_cross_entropy(logits.float(), ya, yb, lam, tc["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if rho:                                               # SAM: step to the worst nearby point, re-take the gradient there
                with torch.no_grad():
                    params = [q for q in model.parameters() if q.grad is not None]
                    norm = torch.norm(torch.stack([q.grad.norm() for q in params])) + 1e-12
                    eps = [q.grad * (rho / norm) for q in params]
                    torch._foreach_add_(params, eps)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                    loss2 = T.mixed_cross_entropy(model(x).float(), ya, yb, lam, tc["label_smoothing"])
                loss2.backward()
                with torch.no_grad():
                    torch._foreach_sub_(params, eps)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)
            tot_loss += loss.item() * len(x)
            tot_correct += (logits.argmax(1) == ya).sum().item()
            tot_n += len(x)
        t_used += time.time() - t0
        heldout = ""
        if len(va) and ((epoch + 1) % tc["monitor_every"] == 0 or epoch + 1 == tc["epochs"]):
            heldout = f"{accuracy(I.tta(I.predict_views(ema.shadow if ema else model, x_va, device, views=[0]), 'none'), y_va):.4f}"
        log.write(f"{epoch + 1},{sched.get_last_lr()[0]:.5f},{tot_loss / tot_n:.4f},{tot_correct / tot_n:.4f},{heldout},{t_used:.0f}\n")
        log.flush()
        print(f"epoch {epoch + 1}/{tc['epochs']}  loss {tot_loss / tot_n:.4f}  train acc {tot_correct / tot_n:.4f}"
              f"  {'held-out ' + heldout if heldout else ''}  {t_used:.0f}s", flush=True)
        if (epoch + 1) % 10 == 0 and epoch + 1 < tc["epochs"]:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "order_g": stream.order_g.get_state(), "aug_g": stream.aug_g.get_state().cpu(),
                        "aug_np": stream.aug_np.bit_generator.state, "ta_g": stream.ta_g.get_state(),
                        "epoch": epoch + 1, "time": t_used,
                        **({"ema": ema.shadow.state_dict()} if ema is not None else {})}, out / "ckpt.pt")
    log.close()

    # --- final evaluation: last epoch only (no checkpoint selection on held-out data) ---
    if ema is not None:
        model = ema.shadow                                        # evaluate and save the averaged weights
    torch.save(model.state_dict(), out / "model.pt")
    p_va = I.predict_views(model, x_va, device) if len(va) else np.zeros((0, len(I.VIEWS), 10), np.float16)
    p_sh = I.predict_views(model, I.shift_mimic(x_va, va), device) if len(va) else p_va
    p_te = I.predict_views(model, d["x_test"][: tc.get("limit") or None], device)
    # L6 robustness sets: flip-TTA predicted class per held-out image (compact; enough for paired tests)
    robust = {}
    if len(va):
        for name, xs in build_sets(x_va, va, y_va).items():
            if name != "clean":
                p = I.predict_views(model, xs, device, views=I.TTA["flip"]).astype(np.float32).mean(1)
                robust[name] = p.argmax(1).astype(np.uint8)
    np.savez_compressed(out / "preds.npz", val_idx=va, val=p_va, shift=p_sh, test=p_te,
                        **{f"robust_{k}": v for k, v in robust.items()})
    metrics = {"params": n_params, "epochs": tc["epochs"], "train_seconds": round(t_used),
               "final_train_loss": tot_loss / tot_n, "final_train_acc": tot_correct / tot_n, "ema": tc.get("ema", 0), "sam": rho,
               **provenance(device)}
    for mode in I.TTA:
        metrics[f"heldout_acc_{mode}"] = accuracy(I.tta(p_va, mode), y_va)
        metrics[f"shift_acc_{mode}"] = accuracy(I.tta(p_sh, mode), y_va)
    for name, pred in robust.items():
        metrics[f"robust_acc_{name}"] = float((pred == y_va).mean())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "ckpt.pt").unlink(missing_ok=True)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        cfg = default_config(train={"epochs": 1, "limit": 1024, "workers": 0}, data={"photometric": "trivialaugment", "rendition_p": 0.25})
        run(cfg, RUNS_DIR / "smoke")
    else:
        path = Path(sys.argv[1])
        if (path.parent / "metrics.json").exists():
            print(f"{path.parent.name}: already done")
        elif (path.parent / "claimed_elsewhere").exists():          # trained on another machine (e.g. Colab)
            print(f"{path.parent.name}: claimed elsewhere, skipping")
        else:
            run(json.loads(path.read_text()), path.parent)
