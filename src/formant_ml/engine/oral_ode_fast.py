"""구강압 상미분방정식의 빠른 경로 — numba 순전파 + 수반(adjoint) 역전파 (MEASUREMENTS §52.468).

`glottis.GlottalSource._oral_ode` 는 틀(1 ms)마다 뉴턴 8 회를 torch 로 돌려, 1.32 s 발화 한 번에 작은 연산이 약 16 만 개 쌓였다. 프로파일(W8b 조리법,
8 회 적합): 합성 순전파 1.35 s 중 이 함수가 0.82 s, 역전파는 한 번에 2.9 s. `tviir` 의 빠른 경로와 같은 방법으로 고친다.

식 (x = √Po, 틀 i):  F(x_i) = C (x_i² − x_{i−1}²)/dt − k_g √(P − x_i²) + k_o x_i = 0,  x 에 단조 증가.
순전파는 torch 판과 **같은 뉴턴 8 회·같은 가두기**(0 ≤ x ≤ √(P + 1e-9), r ≥ 1e-9)를 float64 로 돈다.
역전파는 음함수 정리: F_x = 2C x/dt + k_g x/√r + k_o,
  dx/dP = k_g/(2√r)/F_x,  dx/dk_g = √r/F_x,  dx/dk_o = −x/F_x,  dx_i/dx_{i−1} = (2C x_{i−1}/dt)/F_x.
위에서 가둬진 틀(x = √P)은 dx/dP = 1/(2√P) 만, 0 에서 가둬진 틀은 기울기 0 — torch 의 clamp/minimum 과 같은 가지.
수치는 torch 판과 대조해 고정한다 (tests/engine/test_oral_ode_fast.py).
"""
from __future__ import annotations

import math

import numpy as np
import torch

try:
    import numba as _nb

    @_nb.njit(cache=True, fastmath=False)
    def _fwd(P, ko, kg, x0, C, dt, r32, X):
        B, T = P.shape
        for b in range(B):
            x = x0[b]
            for i in range(T):
                Pi = P[b, i]
                koi = ko[b, i]
                kgi = kg[b, i]
                xp2 = x * x
                sP = math.sqrt(Pi + 1e-9)
                if x > sP:
                    x = sP
                for _ in range(8):
                    r = Pi - x * x
                    if r < 1e-9:
                        r = 1e-9
                    sr = math.sqrt(r)
                    f = C * (x * x - xp2) / dt - kgi * sr + koi * x
                    df = 2.0 * C * x / dt + kgi * x / sr + koi
                    if df < 1e-12:
                        df = 1e-12
                    x = x - f / df
                    if x < 0.0:
                        x = 0.0
                    if x > sP:
                        x = sP
                if r32:
                    x = float(np.float32(x))         # 청크 상태가 float32 로 저장되므로 틀마다 같게 (스트리밍 = 오프라인)
                X[b, i] = x

    @_nb.njit(cache=True, fastmath=False)
    def _bwd(gX, X, P, ko, kg, x0, C, dt, gP, gko, gkg):
        B, T = P.shape
        for b in range(B):
            carry = 0.0
            for i in range(T - 1, -1, -1):
                a = gX[b, i] + carry
                x = X[b, i]
                xprev = X[b, i - 1] if i > 0 else x0[b]
                Pi = P[b, i]
                sP = math.sqrt(Pi + 1e-9)
                gP[b, i] = 0.0
                gko[b, i] = 0.0
                gkg[b, i] = 0.0
                carry = 0.0
                if x <= 0.0:
                    continue
                if x >= sP * (1.0 - 1e-12):
                    gP[b, i] = a * 0.5 / sP
                    continue
                r = Pi - x * x
                if r < 1e-9:
                    r = 1e-9
                sr = math.sqrt(r)
                Fx = 2.0 * C * x / dt + kg[b, i] * x / sr + ko[b, i]
                if Fx < 1e-12:
                    Fx = 1e-12
                gP[b, i] = a * kg[b, i] / (2.0 * sr) / Fx
                gkg[b, i] = a * sr / Fx
                gko[b, i] = -a * x / Fx
                carry = a * (2.0 * C * xprev / dt) / Fx

    HAVE_FAST = True
except Exception:                                     # pragma: no cover
    HAVE_FAST = False


class OralODEFn(torch.autograd.Function):
    """(P, k_o, k_g) (B,T) → x = √Po (B,T). x0 (B,) 는 상수로 본다(청크 상태)."""

    @staticmethod
    def forward(ctx, P, ko, kg, x0, C: float, dt: float):
        dev, dt_ = P.device, P.dtype
        np_ = lambda t: np.ascontiguousarray(t.detach().double().cpu().numpy())
        Pn, kon, kgn, x0n = np_(P), np_(ko), np_(kg), np_(x0)
        X = np.empty_like(Pn)
        _fwd(Pn, kon, kgn, x0n, float(C), float(dt), P.dtype == torch.float32, X)
        ctx.save_for_backward(torch.from_numpy(X), torch.from_numpy(Pn), torch.from_numpy(kon), torch.from_numpy(kgn),
                              torch.from_numpy(x0n))
        ctx.C, ctx.dt, ctx.dev, ctx.dtype = float(C), float(dt), dev, dt_
        return torch.from_numpy(X).to(dev, dt_)

    @staticmethod
    def backward(ctx, gX):
        X, Pn, kon, kgn, x0n = ctx.saved_tensors
        g = np.ascontiguousarray(gX.detach().double().cpu().numpy())
        gP, gko, gkg = np.empty_like(g), np.empty_like(g), np.empty_like(g)
        _bwd(g, X.numpy(), Pn.numpy(), kon.numpy(), kgn.numpy(), x0n.numpy(), ctx.C, ctx.dt, gP, gko, gkg)
        t = lambda a: torch.from_numpy(a).to(ctx.dev, ctx.dtype)
        return t(gP), t(gko), t(gkg), None, None, None


def oral_ode_x(P: torch.Tensor, ko: torch.Tensor, kg: torch.Tensor, x0: torch.Tensor, C: float, dt: float) -> torch.Tensor:
    return OralODEFn.apply(P, ko, kg, x0, C, dt)
