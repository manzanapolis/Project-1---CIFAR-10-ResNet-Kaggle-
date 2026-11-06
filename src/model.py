"""Model (M1-M3): a configurable residual network in the project spec's design variables.

  N = len(widths) residual layers; layer i has B_i = blocks[i] blocks at C_i = widths[i] channels,
  3x3 convs (F_i = 3), 1x1 projection skips (K_i = 1), global average pooling (P = H_N), FC to 10.
  The stem width C_0 equals C_1.

  M1  block: "post" (spec reference, ReLU after the sum) or "pre" (pre-activation)
  M2  shortcut: "conv" (1x1 conv, stride 2) or "avgpool" (2x2 average pool, then 1x1 conv)
  M3  shape: widths / blocks

Block code adapted from kuangliu/pytorch-cifar (https://github.com/kuangliu/pytorch-cifar),
MIT License, Copyright (c) 2017 liukuang. Changes: configurable widths and block counts,
optional anti-aliased (avg-pool) projection shortcut, 3-or-more-layer networks.
"""
import torch.nn as nn
import torch.nn.functional as F

LIMIT = 5_000_000


def projection(cin, cout, stride, kind, bn=True):
    """Skip path when the shape changes (M2)."""
    layers = []
    if kind == "avgpool" and stride > 1:
        layers.append(nn.AvgPool2d(stride))
        stride = 1
    layers.append(nn.Conv2d(cin, cout, 1, stride=stride, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """Post-activation: y = ReLU(S(x) + BN(conv(ReLU(BN(conv(x))))))."""
    def __init__(self, cin, cout, stride, shortcut):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = projection(cin, cout, stride, shortcut)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class PreActBlock(nn.Module):
    """Pre-activation: y = S(x) + conv(ReLU(BN(conv(ReLU(BN(x))))))."""
    def __init__(self, cin, cout, stride, shortcut):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = projection(cin, cout, stride, shortcut, bn=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        skip = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + skip


class ResNet(nn.Module):
    def __init__(self, widths, blocks, block="post", shortcut="conv", num_classes=10):
        super().__init__()
        self.pre = block == "pre"
        Block = PreActBlock if self.pre else BasicBlock
        c0 = widths[0]
        self.stem = nn.Sequential(nn.Conv2d(3, c0, 3, padding=1, bias=False),
                                  *([] if self.pre else [nn.BatchNorm2d(c0), nn.ReLU(inplace=True)]))
        layers, cin = [], c0
        for i, (c, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(Block(cin, c, 2 if (i > 0 and j == 0) else 1, shortcut))
                cin = c
        self.layers = nn.Sequential(*layers)
        self.final = nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(inplace=True)) if self.pre else nn.Identity()
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        out = self.final(self.layers(self.stem(x)))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))


def build(model_cfg):
    return ResNet(model_cfg["widths"], model_cfg["blocks"], model_cfg["block"], model_cfg["shortcut"])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summary(model, device="cpu", limit=LIMIT):
    """Print the layer table the spec asks for (torchsummary if installed) and enforce the 5M limit.
    limit=None only for distillation teachers, which are never submitted."""
    n = count_params(model)
    try:
        from torchsummary import summary as ts
        ts(model, (3, 32, 32), device=str(device).split(":")[0])
    except ImportError:
        print(model)
    print(f"trainable parameters: {n:,} (limit {limit:,})" if limit else f"trainable parameters: {n:,} (teacher, no limit)")
    assert limit is None or n <= limit, f"model has {n:,} parameters, over the {limit:,} limit"
    return n
