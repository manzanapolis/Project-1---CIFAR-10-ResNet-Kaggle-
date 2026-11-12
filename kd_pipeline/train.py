"""
Train a < 5M-parameter ResNet on the competition's CIFAR-10 data and write a Kaggle submission.

Expected layout (from the Kaggle "Data" tab):
    data/train_images/*.png   data/train_labels.csv
    data/test_images/*.png    data/test.csv   data/sample_submission.csv

Usage:
    python train.py                       # 45k train / 5k val, 100 epochs
    python train.py --val-size 0 --epochs 200 --tag full   # final run on all 50k

Data augmentation runs on the GPU, applied to whole batches, so the DataLoader never becomes the bottleneck.

Optional extras (all off by default, so the defaults reproduce the v1 baseline):
    --aug trivial        per-sample TrivialAugmentWide-style ops on the GPU (Müller & Hutter, 2021)
    --mix cutmix|mixup   CutMix (Yun et al., 2019) / Mixup (Zhang et al., 2018) with probability --mix-prob
    --ema 0.999          evaluate/submit an exponential moving average of the weights
    --sam 0.05           Sharpness-Aware Minimization (Foret et al., 2021) with radius rho
    --se 16              squeeze-and-excitation (Hu et al., 2018) in every residual block, reduction ratio 16
    --arch wrn28_10      WideResNet-28-10 teacher (36M params, NOT a valid submission; for distillation)
    --teacher runs/teacher_wrn/model.pt   knowledge distillation (Hinton et al., 2015) from a trained WRN-28-10:
                         loss = (1 - kd_alpha) * CE + kd_alpha * T^2 * KL(teacher_T || student_T) on the same augmented batch
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    """ReLU(S(x) + F(x)),  F = conv -> BN -> ReLU -> conv -> BN,  S = identity or KxK conv + BN."""

    def __init__(self, c_in, c_out, stride, f=3, k=1, se=0):
        super().__init__()
        self.drop_prob = 0.0      # stochastic depth (Huang et al., 2016): drop F(x) per sample during training only
        self.reuse_mask = False   # SAM's 2nd pass reuses the 1st pass's mask so both gradients see the same network
        self._mask = None
        self.conv1 = nn.Conv2d(c_in, c_out, f, stride, f // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, f, 1, f // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, k, stride, k // 2, bias=False), nn.BatchNorm2d(c_out))
        # squeeze-and-excitation (Hu et al., 2018) on F(x), reduction ratio se; created last so se=0 leaves init/RNG unchanged
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_out, c_out // se, 1), nn.ReLU(),
                                nn.Conv2d(c_out // se, c_out, 1), nn.Sigmoid()) if se else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = out * self.se(out)
        if self.training and self.drop_prob > 0:
            if not self.reuse_mask:
                keep = 1.0 - self.drop_prob
                self._mask = torch.bernoulli(torch.full((x.shape[0], 1, 1, 1), keep, device=x.device)) / keep
            out = out * self._mask.to(out.dtype)
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """Configurable ResNet: channels C_i, blocks B_i, conv kernel F, skip kernel K (see project PDF)."""

    def __init__(self, channels=(64, 128, 256), blocks=(3, 3, 3), f=3, k=1, num_classes=10, se=0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU())
        layers, c_in = [], channels[0]
        for i, (c, b) in enumerate(zip(channels, blocks)):
            for j in range(b):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(BasicBlock(c_in, c, stride, f, k, se))
                c_in = c
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in, num_classes)
        self.register_buffer('mean', MEAN.clone(), persistent=False)
        self.register_buffer('std', STD.clone(), persistent=False)

    def forward(self, x):  # x in [0, 1]
        x = self.layers(self.stem((x - self.mean) / self.std))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class WideBlock(nn.Module):
    """Pre-activation wide residual block (Zagoruyko & Komodakis, 2016)."""

    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(c_in), nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(c_out), nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.shortcut = None if (stride == 1 and c_in == c_out) else nn.Conv2d(c_in, c_out, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        s = x if self.shortcut is None else self.shortcut(o)
        o = self.conv1(o)
        o = self.conv2(F.relu(self.bn2(o)))
        return o + s


class WideResNet(nn.Module):
    """WRN-depth-widen, used only as a distillation teacher (exceeds the 5M budget)."""

    def __init__(self, depth=28, widen=10, num_classes=10):
        super().__init__()
        n, w = (depth - 4) // 6, [16, 16 * widen, 32 * widen, 64 * widen]
        self.stem = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        layers, c_in = [], w[0]
        for i, c in enumerate(w[1:]):
            for j in range(n):
                layers.append(WideBlock(c_in, c, 2 if (i > 0 and j == 0) else 1))
                c_in = c
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c_in)
        self.fc = nn.Linear(c_in, num_classes)
        self.register_buffer('mean', MEAN.clone(), persistent=False)
        self.register_buffer('std', STD.clone(), persistent=False)

    def forward(self, x):
        x = F.relu(self.bn(self.layers(self.stem((x - self.mean) / self.std))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- data
def load_images(folder, filenames):
    return np.stack([np.array(Image.open(folder / f).convert('RGB')) for f in filenames])  # N,32,32,3 uint8


def load_data(data_dir):
    cache = data_dir / 'cache.npz'
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d['x_train'], d['y_train'], d['x_test'], d['test_ids']

    labels = pd.read_csv(data_dir / 'train_labels.csv')
    print('train_labels.csv columns:', list(labels.columns))
    files, y = labels['filename'], labels['label']  # columns: id, filename, label (0-9)
    x_train = load_images(data_dir / 'train_images', files)

    test = pd.read_csv(data_dir / 'test.csv')
    x_test = load_images(data_dir / 'test_images', test['filename'])
    np.savez(cache, x_train=x_train, y_train=y.to_numpy(np.int64), x_test=x_test, test_ids=test['id'].to_numpy())
    return x_train, y.to_numpy(np.int64), x_test, test['id'].to_numpy()


def to_tensor(x_uint8, device):
    """uint8 NHWC -> float NCHW in [0, 1]; normalization happens inside the model."""
    return torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255)


# ---- TrivialAugmentWide-style per-sample ops on [0, 1] images, m in [0, 1] is the magnitude
GRAY = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def _signed(m):
    return m * torch.where(torch.rand_like(m) < 0.5, -1.0, 1.0)


def _blend(img, other, factor):
    return (other + factor.view(-1, 1, 1, 1) * (img - other)).clamp(0, 1)


def _gray(x):
    return (x * GRAY.to(x.device)).sum(1, keepdim=True)


def _sharp_blur(x):
    k = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]], device=x.device) / 13
    out = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), k.expand(3, 1, 3, 3), groups=3)
    return out


def _equalize(x):
    B, C, H, W = x.shape
    v = (x * 255).round().long().reshape(B * C, H * W)
    hist = torch.zeros(B * C, 256, device=x.device).scatter_add_(1, v, torch.ones_like(v, dtype=torch.float))
    cdf = hist.cumsum(1)
    cdf_min = torch.where(hist > 0, cdf, torch.full_like(cdf, float('inf'))).min(1, keepdim=True).values
    lut = ((cdf - cdf_min) / (H * W - cdf_min).clamp(min=1)).clamp(0, 1)
    return lut.gather(1, v).view(B, C, H, W)


def _autocontrast(x):
    lo, hi = x.amin((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
    return torch.where(hi > lo, (x - lo) / (hi - lo).clamp(min=1e-6), x)


def _affine(x, theta):
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def _geo(kind):
    def f(x, m):
        B = x.size(0)
        th = torch.zeros(B, 2, 3, device=x.device)
        th[:, 0, 0] = th[:, 1, 1] = 1
        s = _signed(m)
        if kind == 'rotate':
            a = s * math.radians(135)
            th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1] = a.cos(), -a.sin(), a.sin(), a.cos()
        elif kind == 'shear_x':
            th[:, 0, 1] = s * 0.99
        elif kind == 'shear_y':
            th[:, 1, 0] = s * 0.99
        elif kind == 'translate_x':
            th[:, 0, 2] = s * 2 * 32 / 32  # up to 32 px, as in torchvision's TrivialAugmentWide
        elif kind == 'translate_y':
            th[:, 1, 2] = s * 2 * 32 / 32
        return _affine(x, th)
    return f


TA_OPS = [
    lambda x, m: x,                                                          # identity
    lambda x, m: _blend(x, torch.zeros_like(x), 1 + _signed(m) * 0.99),      # brightness
    lambda x, m: _blend(x, _gray(x).mean((2, 3), keepdim=True).expand_as(x), 1 + _signed(m) * 0.99),  # contrast
    lambda x, m: _blend(x, _gray(x).expand_as(x), 1 + _signed(m) * 0.99),    # color (saturation)
    lambda x, m: _blend(x, _sharp_blur(x), 1 + _signed(m) * 0.99),           # sharpness
    lambda x, m: (((x * 255).floor().long() >> (6 * m).long().view(-1, 1, 1, 1))
                  << (6 * m).long().view(-1, 1, 1, 1)).float() / 255,       # posterize 8..2 bits
    lambda x, m: torch.where(x >= (1 - m).view(-1, 1, 1, 1), 1 - x, x),      # solarize
    lambda x, m: _autocontrast(x),
    lambda x, m: _equalize(x),
    _geo('rotate'), _geo('shear_x'), _geo('shear_y'), _geo('translate_x'), _geo('translate_y'),
]


def trivial_augment(x):
    """Each image gets one uniformly chosen op at a uniformly chosen magnitude (31 bins)."""
    B = x.size(0)
    op = torch.randint(0, len(TA_OPS), (B,), device=x.device)
    mag = torch.randint(0, 31, (B,), device=x.device).float() / 30
    out = x.clone()
    for k, f in enumerate(TA_OPS[1:], start=1):
        sel = (op == k).nonzero().squeeze(1)
        if sel.numel():
            out[sel] = f(x[sel], mag[sel])
    return out


def mix_batch(x, y, mode, alpha=1.0):
    """CutMix / Mixup. Returns mixed x, the two label sets and the weight of the first."""
    perm = torch.randperm(x.size(0), device=x.device)
    lam = float(np.random.beta(alpha, alpha))
    if mode == 'mixup':
        return lam * x + (1 - lam) * x[perm], y, y[perm], lam
    H, W = x.shape[2:]
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, y, y[perm], lam


def gpu_augment(x, cutout=16, aug='basic'):
    """Random crop (4px reflect pad) + horizontal flip [+ TrivialAugment] + Cutout, on the whole batch on the GPU."""
    B, dev = x.size(0), x.device
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    i, j = torch.randint(0, 9, (B,), device=dev), torch.randint(0, 9, (B,), device=dev)
    r = torch.arange(32, device=dev)
    rows = (i[:, None] + r)[:, :, None]
    cols = (j[:, None] + r)[:, None, :]
    x = xp[torch.arange(B, device=dev)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if aug == 'trivial':
        x = trivial_augment(x)
    if cutout > 0:
        cy, cx = torch.randint(0, 32, (B,), device=dev), torch.randint(0, 32, (B,), device=dev)
        my = (r[None, :] - cy[:, None]).abs() < cutout // 2
        mx = (r[None, :] - cx[:, None]).abs() < cutout // 2
        x = x * (~(my[:, :, None] & mx[:, None, :]))[:, None].float()
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, x, bs=1000, tta=True):
    model.eval()
    out = []
    for k in range(0, len(x), bs):
        xb = x[k:k + bs].contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(xb).float().softmax(1)
            if tta:
                logits += model(xb.flip(3)).float().softmax(1)
        out.append(logits)
    return torch.cat(out)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='runs')
    p.add_argument('--tag', default='baseline')
    p.add_argument('--channels', type=int, nargs='+', default=[64, 128, 256])
    p.add_argument('--blocks', type=int, nargs='+', default=[3, 3, 3])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--cutout', type=int, default=16)
    p.add_argument('--val-size', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--repeat-aug', type=int, default=1, help='repeated augmentation: each step uses bs distinct images, each repeat_aug times (1 = off)')
    p.add_argument('--arch', default='resnet', choices=['resnet', 'wrn28_10'])
    p.add_argument('--aug', default='basic', choices=['basic', 'trivial'])
    p.add_argument('--mix', default='none', choices=['none', 'cutmix', 'mixup'])
    p.add_argument('--mix-prob', type=float, default=0.5)
    p.add_argument('--mix-alpha', type=float, default=1.0)
    p.add_argument('--ema', type=float, default=0.0, help='EMA decay per step (0 = off)')
    p.add_argument('--sam', type=float, default=0.0, help='SAM radius rho (0 = plain SGD)')
    p.add_argument('--drop-path', type=float, default=0.0,
                   help='stochastic depth: max drop prob, linearly increasing over blocks (0 = off; inference unaffected)')
    p.add_argument('--se', type=int, default=0, help='squeeze-and-excitation reduction ratio in every block (0 = off)')
    p.add_argument('--teacher', default='', help='path to a WRN-28-10 model.pt to distill from, or comma-separated paths (soft targets averaged); empty = no KD')
    p.add_argument('--kd-alpha', type=float, default=0.7, help='weight of the distillation term')
    p.add_argument('--kd-T', type=float, default=4.0, help='distillation temperature')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda'
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    x_np, y_np, xt_np, test_ids = load_data(Path(args.data))
    print(f'train {x_np.shape}  test {xt_np.shape}  class counts {np.bincount(y_np)}')
    perm = np.random.permutation(len(x_np))
    val_idx, tr_idx = perm[:args.val_size], perm[args.val_size:]
    x_tr, y_tr = to_tensor(x_np[tr_idx], dev), torch.from_numpy(y_np[tr_idx]).to(dev)
    x_val, y_val = to_tensor(x_np[val_idx], dev), torch.from_numpy(y_np[val_idx]).to(dev)
    x_test = to_tensor(xt_np, dev)

    if args.arch == 'resnet':
        model = ResNet(args.channels, args.blocks, se=args.se)
        if args.drop_path > 0:
            blocks = [m for m in model.modules() if isinstance(m, BasicBlock)]
            for i, b in enumerate(blocks):  # linear decay rule: p_l = p_max * l / L
                b.drop_prob = args.drop_path * (i + 1) / len(blocks)
    else:
        model = WideResNet(28, 10)
    model = model.to(dev).to(memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f'Trainable parameters: {n_params:,}')
    if args.arch == 'resnet':
        assert n_params <= 5_000_000, 'model exceeds the 5M parameter limit'
    else:
        print('NOTE: teacher model above the 5M limit -- for distillation only, not a valid submission')
    teachers = []  # --teacher a.pt[,b.pt,...]: several teachers -> soft targets = mean of their temperature-T probabilities
    for tp in filter(None, args.teacher.split(',')):
        t = WideResNet(28, 10).to(dev).to(memory_format=torch.channels_last)
        t.load_state_dict(torch.load(tp, map_location=dev))
        t.eval().requires_grad_(False)
        if args.val_size:
            print(f'Teacher {tp} validation accuracy (flip TTA): {(predict(t, x_val).argmax(1) == y_val).float().mean().item():.4f}')
        teachers.append(t)
    teacher = teachers[0] if teachers else None
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    sd_blocks = [m for m in model.modules() if isinstance(m, BasicBlock) and m.drop_prob > 0]

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    steps_per_epoch = len(x_tr) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                                                pct_start=0.15, anneal_strategy='cos')
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history = []
    for epoch in range(args.epochs):
        model.train()
        t0, tot_loss, correct, seen = time.time(), 0.0, 0, 0
        order = torch.randperm(len(x_tr), device=dev)
        for s in range(steps_per_epoch):
            idx = order[s * args.bs:(s + 1) * args.bs]
            if args.repeat_aug > 1:  # repeated augmentation (Hoffer et al., 2020): each image appears repeat_aug times, augmented independently
                idx = idx.repeat(args.repeat_aug)
            xb, yb = gpu_augment(x_tr[idx], args.cutout, args.aug), y_tr[idx]
            yb2, lam = yb, 1.0
            if args.mix != 'none' and np.random.rand() < args.mix_prob:
                xb, yb, yb2, lam = mix_batch(xb, yb, args.mix, args.mix_alpha)
            if teacher is not None:  # soft targets on the same (augmented, mixed) images, reused by both SAM passes
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    if len(teachers) == 1:
                        t_logp = F.log_softmax(teacher(xb).float() / args.kd_T, 1)
                    else:
                        t_logp = torch.stack([F.log_softmax(t(xb).float() / args.kd_T, 1) for t in teachers]).logsumexp(0) - math.log(len(teachers))

            def forward_loss():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(xb)
                    loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb2)
                if teacher is not None:
                    s_logp = F.log_softmax(out.float() / args.kd_T, 1)
                    kd = F.kl_div(s_logp, t_logp, log_target=True, reduction='batchmean') * args.kd_T ** 2
                    loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
                return out, loss

            opt.zero_grad(set_to_none=True)
            logits, loss = forward_loss()
            loss.backward()
            if args.sam:
                # SAM: step to the worst-case point within radius rho, take the gradient there, step back
                params = [p for p in model.parameters() if p.grad is not None]
                with torch.no_grad():
                    grads = [p.grad for p in params]
                    scale = args.sam / (torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])) + 1e-12)
                    eps = torch._foreach_mul(grads, scale)
                    torch._foreach_add_(params, eps)
                opt.zero_grad(set_to_none=True)
                for m in bns:  # don't update BN running stats twice per step
                    m.backup_momentum, m.momentum = m.momentum, 0.0
                for b in sd_blocks:
                    b.reuse_mask = True
                forward_loss()[1].backward()
                for b in sd_blocks:
                    b.reuse_mask = False
                for m in bns:
                    m.momentum = m.backup_momentum
                with torch.no_grad():
                    torch._foreach_sub_(params, eps)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
        rec = {'epoch': epoch + 1, 'train_loss': tot_loss / seen, 'train_acc': correct / seen,
               'lr': sched.get_last_lr()[0], 'time': time.time() - t0}
        if args.val_size:
            rec['val_acc'] = (predict(model, x_val, tta=False).argmax(1) == y_val).float().mean().item()
            if ema is not None:
                rec['val_acc_ema'] = (predict(ema.module, x_val, tta=False).argmax(1) == y_val).float().mean().item()
        history.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)

    if args.val_size:
        val_tta = (predict(model, x_val).argmax(1) == y_val).float().mean().item()
        print(f'Final validation accuracy (flip TTA): {val_tta:.4f}')
        if ema is not None:
            val_tta_ema = (predict(ema.module, x_val).argmax(1) == y_val).float().mean().item()
            print(f'Final validation accuracy, EMA weights (flip TTA): {val_tta_ema:.4f}')
    final = ema.module if ema is not None else model  # submit the EMA weights when EMA is on
    probs = predict(final, x_test)
    preds = probs.argmax(1).cpu().numpy()
    pd.DataFrame({'id': test_ids, 'label': preds}).to_csv(out / 'submission.csv', index=False)
    np.save(out / 'test_probs.npy', probs.cpu().numpy())
    torch.save(final.state_dict(), out / 'model.pt')
    json.dump({'args': vars(args), 'n_params': n_params, 'history': history}, open(out / 'history.json', 'w'), indent=1)
    print(f'Trainable parameters: {n_params:,}')
    print(f'Wrote {out / "submission.csv"}  (pred class counts {np.bincount(preds, minlength=10)})')


if __name__ == '__main__':
    main()
