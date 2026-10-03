"""성대 떨림 진폭의 로지스틱 재귀 — numba 순전파 + 수반 역전파 (MEASUREMENTS §52.468).

`glottis.GlottalSource.physiology` 는 틀마다 닫힌 해
    A(t+dt) = A*·A / (A*·q + A·(1−q)),  q = e^{−max(dt·σ, 0)}          (A* > seed)
    A(t+dt) = max(A·e^{−dt·f0/N_decay}, seed)                           (그 밖)
를 torch 로 돌려 1.32 s 발화 한 번에 틀 1320 개 × 작은 연산 여럿을 순·역전파로 쌓았다. 같은 식을 numba 로 돌리고 기울기는 손으로 쓴 수반 재귀로 낸다.
윗가지: ∂A'/∂A = A*²q/D², ∂A'/∂A* = A²(1−q)/D², ∂A'/∂q = −A*·A(A*−A)/D², ∂q/∂σ = −dt·q (dt·σ > 0 일 때), D = A*q + A(1−q).
아랫가지: ∂A'/∂A = e, ∂A'/∂e = A (seed 에 안 가둬졌을 때). 수치는 torch 판과 대조해 고정한다 (tests/engine/test_oral_ode_fast.py).
"""
from __future__ import annotations

import math

import numpy as np
import torch

try:
    import numba as _nb

    @_nb.njit(cache=True, fastmath=False)
    def _fwd(astar, grow, dfac, a0, seed, dt, r32, A):
        B, T = astar.shape
        for b in range(B):
            a = a0[b]
            for i in range(T):
                s = astar[b, i]
                if s > seed:
                    g = dt * grow[b, i]
                    q = math.exp(-g) if g > 0.0 else 1.0
                    D = s * q + a * (1.0 - q)
                    if D < 1e-6:
                        D = 1e-6
                    a = s * a / D
                else:
                    a2 = a * dfac[b, i]
                    a = a2 if a2 > seed else seed
                if r32:
                    a = float(np.float32(a))         # 청크 상태가 float32 로 저장되므로 틀마다 같게 (스트리밍 = 오프라인)
                A[b, i] = a

    @_nb.njit(cache=True, fastmath=False)
    def _bwd(gA, A, astar, grow, dfac, a0, seed, dt, gs, gg, gd):
        B, T = astar.shape
        for b in range(B):
            carry = 0.0
            for i in range(T - 1, -1, -1):
                g = gA[b, i] + carry
                ap = A[b, i - 1] if i > 0 else a0[b]
                s = astar[b, i]
                gs[b, i] = 0.0
                gg[b, i] = 0.0
                gd[b, i] = 0.0
                carry = 0.0
                if s > seed:
                    x = dt * grow[b, i]
                    q = math.exp(-x) if x > 0.0 else 1.0
                    D = s * q + ap * (1.0 - q)
                    if D < 1e-6:
                        continue
                    D2 = D * D
                    gs[b, i] = g * ap * ap * (1.0 - q) / D2
                    if x > 0.0:
                        gg[b, i] = g * (-s * ap * (s - ap) / D2) * (-dt * q)
                    carry = g * s * s * q / D2
                else:
                    if ap * dfac[b, i] > seed:
                        gd[b, i] = g * ap
                        carry = g * dfac[b, i]

    HAVE_FAST = True
except Exception:                                     # pragma: no cover
    HAVE_FAST = False


class LogisticAmpFn(torch.autograd.Function):
    """(A*, σ, e) (B,T) → A (B,T). a0 (B,) 는 상수(청크 상태)."""

    @staticmethod
    def forward(ctx, astar, grow, dfac, a0, seed: float, dt: float):
        np_ = lambda t: np.ascontiguousarray(t.detach().double().cpu().numpy())
        s, g, d, x0 = np_(astar), np_(grow), np_(dfac), np_(a0)
        A = np.empty_like(s)
        _fwd(s, g, d, x0, float(seed), float(dt), astar.dtype == torch.float32, A)
        ctx.save_for_backward(*(torch.from_numpy(v) for v in (A, s, g, d, x0)))
        ctx.seed, ctx.dt, ctx.dev, ctx.dtype = float(seed), float(dt), astar.device, astar.dtype
        return torch.from_numpy(A).to(astar.device, astar.dtype)

    @staticmethod
    def backward(ctx, gA):
        A, s, g, d, x0 = (v.numpy() for v in ctx.saved_tensors)
        ga = np.ascontiguousarray(gA.detach().double().cpu().numpy())
        gs, gg, gd = np.empty_like(ga), np.empty_like(ga), np.empty_like(ga)
        _bwd(ga, A, s, g, d, x0, ctx.seed, ctx.dt, gs, gg, gd)
        t = lambda a: torch.from_numpy(a).to(ctx.dev, ctx.dtype)
        return t(gs), t(gg), t(gd), None, None, None


def logistic_amp(astar, grow, dfac, a0, seed: float, dt: float) -> torch.Tensor:
    return LogisticAmpFn.apply(astar, grow, dfac, a0, seed, dt)
