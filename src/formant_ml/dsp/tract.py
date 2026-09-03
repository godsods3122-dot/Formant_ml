"""Kelly-Lochbaum 도파관 성도 모델의 미분가능 전달함수.

등간격 원통관 N개의 연접으로 성도를 근사하면, 반사계수 k_m 으로부터
Levinson-Durbin '스텝업' 재귀로 전극(all-pole) 다항식 A(z)를 얻을 수 있다.
시간영역 산란 루프를 돌지 않고 주파수축에서 병렬로 계산되므로 학습에 적합하고,
|k_m| < 1 (면적이 양수이면 자동)이면 A(z)는 최소위상 = 필터가 항상 안정하다.
"""
from __future__ import annotations

import torch

from .core import TWO_PI, freq_grid


def area_to_reflection(area: torch.Tensor, lip_reflection: float = 0.9) -> torch.Tensor:
    """단면적 (B, T, N) -> 반사계수 (B, T, N).

    공개 API 규약: area[..., 0] = 성문쪽, area[..., -1] = 입술쪽.
    내부 래티스 재귀는 입술쪽에서 시작하므로 여기서 한 번 뒤집는다.
    검증: 균일관 -> 500/1500/2500 Hz, /a/형 -> F1 상승, /i/·/u/형 -> F1 하강.
    """
    a = torch.flip(area, dims=[-1]).clamp_min(1e-4)
    k = (a[..., :-1] - a[..., 1:]) / (a[..., :-1] + a[..., 1:])
    lip = torch.full_like(k[..., :1], lip_reflection)
    return torch.cat([k, lip], dim=-1)


def reflection_to_lpc(k: torch.Tensor) -> torch.Tensor:
    """반사계수 (B, T, N) -> LPC 다항식 계수 a (B, T, N+1), a[0] = 1."""
    b, t, n = k.shape
    a = torch.ones(b, t, 1, device=k.device, dtype=k.dtype)
    for m in range(n):
        km = k[..., m : m + 1]
        rev = torch.flip(a, dims=[-1])
        a = torch.cat([a, torch.zeros_like(km)], dim=-1) \
            + km * torch.cat([torch.zeros_like(km), rev], dim=-1)
    return a


def bandwidth_expansion(a: torch.Tensor, rho: float = 0.99) -> torch.Tensor:
    """a_j <- a_j * rho^j : 극점 반지름을 rho배로 줄여 대역폭(손실)을 준다."""
    j = torch.arange(a.shape[-1], device=a.device, dtype=a.dtype)
    return a * rho ** j


def lpc_response(a: torch.Tensor, sample_rate: float, n_freq: int,
                 gain: torch.Tensor | None = None) -> torch.Tensor:
    """H(w) = g / A(w). a: (B, T, P+1) -> (B, T, n_freq) 복소."""
    f = freq_grid(n_freq, sample_rate, device=a.device, dtype=a.dtype)
    w = TWO_PI * f / sample_rate                                   # (n_freq,)
    j = torch.arange(a.shape[-1], device=a.device, dtype=a.dtype)   # (P+1,)
    basis = torch.exp(-1j * (w[:, None] * j[None, :]).to(torch.float32))  # (n_freq,P+1)
    A = torch.einsum("btp,fp->btf", a.to(torch.complex64), basis)
    H = 1.0 / A
    if gain is not None:
        H = H * gain.to(torch.complex64)
    return H


def tract_response(area: torch.Tensor, sample_rate: float, n_freq: int,
                   rho: float = 0.99, lip_reflection: float = 0.9,
                   gain: torch.Tensor | None = None) -> torch.Tensor:
    """단면적 함수 -> 성도 전달함수(복소)."""
    k = area_to_reflection(area, lip_reflection)
    a = bandwidth_expansion(reflection_to_lpc(k), rho)
    return lpc_response(a, sample_rate, n_freq, gain)


def formants_from_response(H: torch.Tensor, sample_rate: float, n_peaks: int = 5):
    """응답에서 국소 최대(포먼트) 주파수 추정. 분석/로깅용(미분 불필요)."""
    mag = H.abs()
    left = mag[..., 1:-1] > mag[..., :-2]
    right = mag[..., 1:-1] > mag[..., 2:]
    peak = (left & right)
    f = freq_grid(H.shape[-1], sample_rate, device=H.device)[1:-1]
    scores = torch.where(peak, mag[..., 1:-1], torch.zeros_like(mag[..., 1:-1]))
    idx = scores.topk(n_peaks, dim=-1).indices
    return f[idx].sort(dim=-1).values
