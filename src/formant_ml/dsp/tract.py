"""Kelly-Lochbaum 도파관 성도 모델의 미분가능 전달함수.

등간격 원통관 N개의 연접으로 성도를 근사하면, 반사계수 k_m 으로부터
Levinson-Durbin '스텝업' 재귀로 전극(all-pole) 다항식 A(z)를 얻을 수 있다.
시간영역 산란 루프를 돌지 않고 주파수축에서 병렬로 계산되므로 학습에 적합하고,
|k_m| < 1 (면적이 양수이면 자동)이면 A(z)는 최소위상 = 필터가 항상 안정하다.
"""
from __future__ import annotations

import math

import torch

from .core import TWO_PI, freq_grid

# 포먼트 대역폭은 주파수에 따라 자란다. 방사 저항이 f^2 로 커지기 때문이고,
# 실측 대역폭(Fant 1972)이 그 모양이다: F1 ~50, F2 ~65, F3(2.8k) ~170,
# F4(4k) ~290, F5(5k) ~430 Hz. 아래 계수는 그 값들에 맞춘 것이다.
#
#     BW_excess(f) = BANDWIDTH_F2_HZ * (f/1000)^2   [Hz]
#
# 직류에서 0 이므로 저역(F1·F2)의 극 위치와 대역폭은 건드리지 않는다 —
# 이미 풀어 둔 면적함수 프리셋이 그대로 유효하다는 뜻이다.
BANDWIDTH_F2_HZ = 15.0


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


def bandwidth_expansion(a: torch.Tensor, rho=0.99) -> torch.Tensor:
    """a_j <- a_j * rho^j : 극점 반지름을 rho배로 줄여 대역폭(손실)을 준다.

    rho 는 스칼라 또는 (B, T, 1) 텐서다. 프레임별로 줄 수 있어야 하는 이유:
    설측음은 측지(side branch)가 에너지를 빼가서 실제 대역폭이 모음보다 훨씬
    넓다. 손실을 발화 내내 고정하면 자음 구간이 모음처럼 쨍하게 울려
    "혀가 입천장에 닿지 않은" 소리가 된다.

    대역폭 환산: BW = -ln(rho) * fs / pi. 24 kHz 에서 rho=0.99 -> 77 Hz,
    rho=0.975 -> 193 Hz.
    """
    j = torch.arange(a.shape[-1], device=a.device, dtype=a.dtype)
    if torch.is_tensor(rho):
        return a * rho.to(a.dtype) ** j
    return a * rho ** j


def frequency_loss(sample_rate: float, n_freq: int, device=None,
                   dtype=torch.float32) -> torch.Tensor:
    """한 단(z^-1)당 **주파수 의존** 손실 G(f) in (0, 1]. (n_freq,)

    `bandwidth_expansion` 의 rho 는 상수라 모든 극에 같은 대역폭을 준다
    (24 kHz 에서 rho=0.99 -> 77 Hz). 그러면 9 kHz 의 공명도 Q=117 로 울려
    응답의 최대점이 늘 최상단 대역에 선다 — 측정: 설측 자세가 11.46 kHz 에서
    +25.5 dB, 모음이 9.16 kHz 에서 +17.6 dB (RIEUL.md §6.1).

    실제로는 방사 저항이 f^2 로 커져서 대역폭이 f^2 로 자란다. 지연 연산자를
    z^-1 -> G(f) z^-1 로 두면 래티스 재귀를 건드리지 않고 그 감쇠를 넣을 수
    있다(rho 를 상수 대신 주파수의 함수로 두는 것과 같다):

        BW(f) = -ln(rho G(f)) fs/pi  =>  G(f) = exp(-pi BW_excess(f)/fs)

    G(0) = 1 이라 저역은 손대지 않는다.
    """
    f = freq_grid(n_freq, sample_rate, device=device, dtype=dtype)
    bw = BANDWIDTH_F2_HZ * (f / 1000.0) ** 2
    return torch.exp(-math.pi * bw / sample_rate)


def lpc_response(a: torch.Tensor, sample_rate: float, n_freq: int,
                 gain: torch.Tensor | None = None,
                 loss: torch.Tensor | None = None) -> torch.Tensor:
    """H(w) = g / A(w). a: (B, T, P+1) -> (B, T, n_freq) 복소.

    `loss` 를 주면 지연 연산자가 z^-1 대신 G(f) z^-1 이 된다 (frequency_loss).
    """
    f = freq_grid(n_freq, sample_rate, device=a.device, dtype=a.dtype)
    w = TWO_PI * f / sample_rate                                   # (n_freq,)
    j = torch.arange(a.shape[-1], device=a.device, dtype=a.dtype)   # (P+1,)
    basis = torch.exp(-1j * (w[:, None] * j[None, :]).to(torch.float32))  # (n_freq,P+1)
    if loss is not None:
        basis = basis * loss.reshape(-1, 1).to(torch.float32) ** j
    A = torch.einsum("btp,fp->btf", a.to(torch.complex64), basis)
    H = 1.0 / A
    if gain is not None:
        H = H * gain.to(torch.complex64)
    return H


def tract_response(area: torch.Tensor, sample_rate: float, n_freq: int,
                   rho=0.99, lip_reflection: float = 0.9,
                   gain: torch.Tensor | None = None,
                   frequency_dependent_loss: bool = True) -> torch.Tensor:
    """단면적 함수 -> 성도 전달함수(복소).

    손실이 두 갈래다. `rho` 는 주파수 무관한 바닥(프레임별로 줄 수 있다),
    `frequency_loss` 는 f^2 로 자라는 방사 손실이다. 후자를 끄면 예전 응답과
    **정확히 같다**(회귀 검사용).
    """
    k = area_to_reflection(area, lip_reflection)
    a = bandwidth_expansion(reflection_to_lpc(k), rho)
    loss = (frequency_loss(sample_rate, n_freq, area.device)
            if frequency_dependent_loss else None)
    return lpc_response(a, sample_rate, n_freq, gain, loss)


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
