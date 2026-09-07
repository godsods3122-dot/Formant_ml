"""위상차(phase dispersion) 파라미터.

왜 별도 모듈인가
----------------
같은 크기 스펙트럼을 갖는 두 소리도 **하모닉 간 위상 관계**가 다르면 다르게 들린다.
성문파의 폐쇄가 날카로우면 모든 하모닉이 폐쇄 순간에 정렬되어(위상 결맞음) 소리가
'단단하게' 들리고, 정렬이 흐트러지면 같은 스펙트럼인데도 부드럽고 넓게 들린다.
이것이 화자 개성의 한 축이며, 크기 스펙트럼만 맞추는 모델이 놓치는 부분이다.

여기서는 위상을 자유롭게 예측하지 **않는다**. 자유 위상은 우리가 없애려던
phasiness 를 되돌린다. 대신 2차 올패스 체인의 위상응답을 하모닉 주파수에서
평가해 위상 오프셋으로 쓴다:

    φ_k += Σ_s arg H_ap(k·f0 ; F_s, r_s)

* 크기응답은 정확히 1 이므로 스펙트럼 포락선이 절대 변하지 않는다.
* 최소위상/안정 구조라 물리적으로 실현 가능한 위상만 낸다.
* 파라미터가 (F_s, r_s) 몇 쌍뿐이라 실제 음성에서 역추정할 수 있다.

측정량: 상대위상(RPS, relative phase shift)
    RPS_k = ∠X_k − k·∠X_1
는 분석창의 시작 위치에 무관한 화자 고유량이고, 위 파라미터가 바로 이것을 만든다.
"""
from __future__ import annotations

import math

import torch

from .core import TWO_PI

PI = math.pi


def allpass_phase(freq_hz: torch.Tensor, ap_freq: torch.Tensor,
                  ap_radius: torch.Tensor, sample_rate: float) -> torch.Tensor:
    """올패스 체인의 위상응답을 임의 주파수에서 평가.

    freq_hz  : (B, T, K)  평가할 주파수 (예: 하모닉 k·f0)
    ap_freq  : (B, T, S)  올패스 단의 중심주파수
    ap_radius: (B, T, S)  극점 반지름 (0..0.995)
    반환      : (B, T, K) 라디안 위상 (누적, anti-wrap 하지 않음)
    """
    w = (TWO_PI * freq_hz / sample_rate).unsqueeze(-1)          # (B, T, K, 1)
    r = ap_radius.clamp(0.0, 0.995).unsqueeze(-2)               # (B, T, 1, S)
    th = (TWO_PI * ap_freq / sample_rate).unsqueeze(-2)
    a1 = -2.0 * r * torch.cos(th)
    a2 = r * r
    c1, s1 = torch.cos(w), -torch.sin(w)                        # e^{-jw}
    c2, s2 = torch.cos(2 * w), -torch.sin(2 * w)
    # num = a2 + a1 e^{-jw} + e^{-2jw},  den = 1 + a1 e^{-jw} + a2 e^{-2jw}
    nr, ni = a2 + a1 * c1 + c2, a1 * s1 + s2
    dr, di = 1.0 + a1 * c1 + a2 * c2, a1 * s1 + a2 * s2
    return (torch.atan2(ni, nr) - torch.atan2(di, dr)).sum(-1)


def relative_phase_shift(spec: torch.Tensor, n_harmonics: int = 12) -> torch.Tensor:
    """하모닉 복소진폭 (…, K) -> 상대위상 RPS_k = ∠X_k − k·∠X_1 (…, K).

    분석 시작 시점(창 위치)에 무관하므로 화자 비교/학습 타깃으로 쓸 수 있다.
    """
    ang = torch.angle(spec[..., :n_harmonics])
    k = torch.arange(1, ang.shape[-1] + 1, device=ang.device, dtype=ang.dtype)
    rps = ang - k * ang[..., :1]
    return rps - TWO_PI * torch.round(rps / TWO_PI)


def circular_mean(angles: torch.Tensor, dim: int = 0):
    """위상의 원형 평균과 집중도(0=완전분산, 1=완전정렬). 반환 (mean, R)."""
    c, s = torch.cos(angles).mean(dim), torch.sin(angles).mean(dim)
    return torch.atan2(s, c), torch.sqrt(c * c + s * s)


def phase_distortion_deviation(rps: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """프레임 간 RPS 의 원형 표준편차(PDD). 값이 크면 '거칠고 기식적'이다."""
    _, r = circular_mean(rps, dim=dim)
    return torch.sqrt(-2.0 * torch.log(r.clamp(1e-6, 1.0)))


def resonator_phase(freq_hz: torch.Tensor, f_form: torch.Tensor,
                    bw_form: torch.Tensor, sample_rate: float) -> torch.Tensor:
    """포먼트 캐스케이드의 위상응답을 임의 주파수에서 평가 (B, T, K).

    H = D(1)/D(z) 이고 D(1) 은 실수 양수이므로 위상은 −∠D(e^{jw}) 이다.
    성도(최소위상)가 만드는 위상을 소스의 위상차와 분리하는 데 쓴다.
    """
    w = (TWO_PI * freq_hz / sample_rate).unsqueeze(-1)          # (B, T, K, 1)
    r = torch.exp(-PI * bw_form / sample_rate).unsqueeze(-2)    # (B, T, 1, S)
    th = (TWO_PI * f_form / sample_rate).unsqueeze(-2)
    b1 = -2.0 * r * torch.cos(th)
    b2 = r * r
    c1, s1 = torch.cos(w), -torch.sin(w)
    c2, s2 = torch.cos(2 * w), -torch.sin(2 * w)
    dr = 1.0 + b1 * c1 + b2 * c2
    di = b1 * s1 + b2 * s2
    return (-torch.atan2(di, dr)).sum(-1)
