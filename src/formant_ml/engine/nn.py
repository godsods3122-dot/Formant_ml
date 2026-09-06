"""Shifted-softplus 신경망 블록.

ssp(x) = softplus(x) − ln 2.  ssp(0) = 0, 어디서나 매끄럽고(C∞) 기울기가 0 이
되는 구간이 없다(ReLU 의 dead unit 없음). 물리 파라미터처럼 **작은 값 근처에서
매끄럽게 움직여야 하는 출력**에 맞다(SchNet 이 같은 이유로 채택).

이 모듈의 모든 망은 **파형을 만들지 않는다.** 출력은 물리 손잡이(또는 그 델타)
뿐이며, `Bounded` 헤드가 물리 범위를 구조적으로 강제한다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2 = math.log(2.0)


def shifted_softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return F.softplus(x, beta=beta) - LOG2 / beta


class ShiftedSoftplus(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return shifted_softplus(x, self.beta)


class SSPMLP(nn.Module):
    """Linear → ssp → … → Linear. 마지막 층은 0 으로 초기화(처음엔 아무것도 안 바꾼다)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 3,
                 zero_last: bool = True):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(ShiftedSoftplus())
        self.net = nn.Sequential(*layers)
        if zero_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class SSPTCN(nn.Module):
    """인과(causal) 시간 컨볼루션 망. 입력 (B, T, C) → (B, T, out).

    프레임률 제어열을 다루는 잔차망/토큰 인코더의 몸통. 인과라서 스트리밍에서
    그대로 쓸 수 있다(미래 프레임을 보지 않는다).
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, kernel: int = 5,
                 n_blocks: int = 3, dilation_base: int = 2):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList()
        self.pads: list[int] = []
        for i in range(n_blocks):
            d = dilation_base ** i
            self.blocks.append(nn.Conv1d(hidden, hidden, kernel, dilation=d))
            self.pads.append((kernel - 1) * d)
        self.out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(self.pads)

    def forward(self, x):
        h = shifted_softplus(self.inp(x)).transpose(1, 2)        # (B, H, T)
        for conv, p in zip(self.blocks, self.pads):
            h = h + shifted_softplus(conv(F.pad(h, (p, 0))))       # 인과 패딩
        return self.out(h.transpose(1, 2))


class Bounded(nn.Module):
    """실수 → [lo, hi]. 물리 범위를 구조로 강제한다 (softplus 기반, 매끄럽다).

    tanh 대신 두 개의 ssp 로 만든 매끄러운 클램프:
        y = lo + (hi − lo)·σ(x)  대신
        y = lo + ssp(x + c) − ssp(x − (hi−lo) + c) 꼴은 경계 근처에서 기울기가
    남아 있어 학습이 경계에서 멈추지 않는다. 여기서는 단순·검증된 sigmoid 를
    쓰되 경계 도달을 막는 여유(margin)를 둔다.
    """

    def __init__(self, lo: float, hi: float, margin: float = 0.0):
        super().__init__()
        self.lo, self.hi, self.margin = float(lo), float(hi), float(margin)

    def forward(self, x):
        lo, hi = self.lo + self.margin, self.hi - self.margin
        return lo + (hi - lo) * torch.sigmoid(x)
