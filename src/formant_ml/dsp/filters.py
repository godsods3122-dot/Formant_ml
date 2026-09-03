"""공명/반공명/올패스/방사 필터의 미분가능 주파수응답.

모든 함수는 프레임별 파라미터 (B, T, K)를 받아 복소응답 (B, T, n_freq)을 돌려준다.
극점 반지름을 r = exp(-pi*BW/fs) < 1 로 두므로 설계상 항상 안정하다.
"""
from __future__ import annotations

import math

import torch

from .core import TWO_PI, freq_grid


def _z_powers(n_freq: int, sample_rate: float, device, dtype):
    """z^-1, z^-2 를 rfft 격자 위에서 평가."""
    f = freq_grid(n_freq, sample_rate, device=device, dtype=dtype)
    w = TWO_PI * f / sample_rate                       # (n_freq,)
    z1 = torch.exp(-1j * w.to(torch.float32))          # e^{-jw}
    return z1, z1 * z1


def _pole_pair(freq, bw, sample_rate, n_freq):
    """공명 극점쌍의 분모 D(w)와 DC 정규화 상수.

    Klatt(1980) 관례대로 DC 이득이 1이 되도록 정규화한다(A = 1 - 2r cos(t) + r^2).
    캐스케이드로 쌓아도 전체 이득이 발산/소멸하지 않으며, 공진점에서는 Q 만큼
    솟아오르는 '진짜 포먼트'가 된다. (피크 정규화로 하면 대역통과 필터들의 곱이
    되어 캐스케이드 전체가 0 으로 죽는다.)
    """
    r = torch.exp(-math.pi * bw / sample_rate)         # (B, T, K)
    theta = TWO_PI * freq / sample_rate
    z1, z2 = _z_powers(n_freq, sample_rate, freq.device, freq.dtype)
    b1 = (-2.0 * r * torch.cos(theta)).unsqueeze(-1)   # (B, T, K, 1)
    b2 = (r * r).unsqueeze(-1)
    D = 1.0 + b1 * z1 + b2 * z2                        # (B, T, K, n_freq) complex
    Ddc = 1.0 + b1 + b2                                # z = 1 (DC)
    return D, Ddc


def resonator_response(freq, bw, gain, sample_rate: float, n_freq: int) -> torch.Tensor:
    """포먼트 공명기 캐스케이드. freq/bw/gain: (B, T, K) -> (B, T, n_freq)."""
    D, Ddc = _pole_pair(freq, bw, sample_rate, n_freq)
    return (Ddc / D) * gain.unsqueeze(-1).to(D.dtype)


def resonator_stage_responses(freq, bw, gain, sample_rate: float, n_freq: int):
    """단(stage)별 응답을 그대로 반환. (B, T, K, n_freq)

    노이즈를 성도 중간(협착 지점)에서 주입하기 위해 필요하다.
    """
    D, Ddc = _pole_pair(freq, bw, sample_rate, n_freq)
    return (Ddc / D) * gain.unsqueeze(-1).to(D.dtype)


def antiresonator_response(freq, bw, sample_rate: float, n_freq: int) -> torch.Tensor:
    """반공명(영점) 캐스케이드: 비음의 안티포먼트, 마찰음의 스펙트럼 골."""
    D, Ddc = _pole_pair(freq, bw, sample_rate, n_freq)
    return (D / Ddc).prod(dim=2)


def allpass_response(freq, radius, sample_rate: float, n_freq: int) -> torch.Tensor:
    """2차 올패스 캐스케이드: 크기응답은 1, 군지연(위상차)만 바꾼다.

    H(z) = (r^2 - 2r cos(t) z^-1 + z^-2) / (1 - 2r cos(t) z^-1 + r^2 z^-2)
    """
    r = radius.clamp(0.0, 0.995)
    theta = TWO_PI * freq / sample_rate
    z1, z2 = _z_powers(n_freq, sample_rate, freq.device, freq.dtype)
    a1 = (-2.0 * r * torch.cos(theta)).unsqueeze(-1)
    a2 = (r * r).unsqueeze(-1)
    num = a2 + a1 * z1 + z2
    den = 1.0 + a1 * z1 + a2 * z2
    return (num / den).prod(dim=2)


def one_pole_tilt(tilt_db: torch.Tensor, sample_rate: float, n_freq: int) -> torch.Tensor:
    """스펙트럼 기울기(성문 소스의 spectral tilt). tilt_db: (B, T) 나이퀴스트 감쇠량."""
    a = torch.exp(-torch.nn.functional.softplus(tilt_db) * 0.05).unsqueeze(-1)
    z1, _ = _z_powers(n_freq, sample_rate, tilt_db.device, tilt_db.dtype)
    return ((1.0 - a) / (1.0 - a * z1))


def lip_radiation_response(sample_rate: float, n_freq: int, alpha: float = 0.98,
                           device=None) -> torch.Tensor:
    """입술 방사 = 미분기 근사 H(z) = 1 - alpha z^-1. (1, 1, n_freq)

    주의: LF 모델은 '유량의 미분'을 직접 생성하므로 기본 체인에서는 꺼 둔다.
    """
    z1, _ = _z_powers(n_freq, sample_rate, device, torch.float32)
    return (1.0 - alpha * z1).reshape(1, 1, -1)


def bands_to_response(band_gains: torch.Tensor, n_freq: int,
                      min_phase: bool = True) -> torch.Tensor:
    """대역 게인 (B, T, n_bands) -> 매끄러운 복소응답 (B, T, n_freq).

    min_phase=True 면 힐베르트 변환으로 최소위상화하여 프리링잉(pre-echo)을 없앤다.
    """
    b, t, n_bands = band_gains.shape
    mag = torch.nn.functional.interpolate(
        band_gains.reshape(b * t, 1, n_bands), size=n_freq,
        mode="linear", align_corners=True
    ).reshape(b, t, n_freq).clamp_min(1e-7)
    if not min_phase:
        return mag.to(torch.complex64)
    # log|H| 의 실수 캡스트럼 -> 최소위상 재구성
    log_mag = torch.log(mag)
    n_fft = 2 * (n_freq - 1)
    cep = torch.fft.irfft(log_mag.to(torch.complex64), n_fft)
    w = torch.zeros(n_fft, device=mag.device, dtype=mag.dtype)
    w[0] = 1.0
    w[1 : n_fft // 2] = 2.0
    w[n_fft // 2] = 1.0
    spec = torch.fft.rfft(cep * w, n_fft)
    return torch.exp(spec)
