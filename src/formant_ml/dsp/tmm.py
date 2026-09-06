"""전달행렬(transfer-matrix) 성도 — **방사 임피던스가 있는** 음향 전달함수.

왜 `tract.py` 로 안 되는가
--------------------------
`tract.py` 는 Kelly-Lochbaum 반사계수를 Levinson 스텝업으로 전극 다항식으로
바꾼다. 빠르고 미분가능하지만, 그 재귀는 입술 반사가 **실수 상수**여야만
성립한다. 실제 방사 임피던스는 복소수이고 주파수에 따라 변한다 —
저역에서 Γ≈−1(거의 전반사), 고역에서 Γ→0(다 나간다).

그 결과가 어떻게 나타나는지 측정했다. 균일관에서 입술만 좁혀 가면:

| 입술 면적 [cm²] | `tract.py` | 이 모듈 |
|---|---|---|
| 1.00 | +2.7 dB | −2.8 dB |
| 0.10 | **+10.0 dB** | −14.6 dB |
| 0.05 | **+13.1 dB** | −19.4 dB |

**입술을 좁힐수록 출력이 커진다.** 실제 물리는 반대다(닫으면 0 으로 간다).
협착이 곧 자음이므로 자음이 모음보다 크게 나오고, 그걸 덧대려고
`presets.POSTURE_GAIN_20`(자세별 출력의 역수)이 생겼다. 근본 원인은 이것이다.

전달행렬은 그 제약이 없다. 주파수마다 2×2 행렬을 곱하고 끝에 복소 방사
임피던스를 걸면 된다. 대신 Levinson 만큼 싸지는 않다(단면 수 × 주파수).

무엇을 계산하는가
-----------------
성문 체적유량 U_g 에서 **원거리 음압** p 까지의 전달함수다. 단극 방사이므로
p ∝ jω·U_lips 이고, 그 jω 가 흔히 말하는 "입술 방사 +6 dB/oct" 다.
`dsp/glottal.py` 의 LF 모델은 유량의 **미분**을 내므로 그쪽 경로에는 이 jω 가
이미 들어 있다 — 두 번 걸지 않도록 `radiation_derivative` 로 끌 수 있다.

단위는 CGS (cm, g, s). 이 레포의 다른 모듈과 같다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .core import freq_grid

# --- 매질 상수 (체온 37 °C 습공기) -----------------------------------------
RHO = 1.14e-3          # 공기 밀도 [g/cm^3]
SOUND_SPEED = 35000.0  # 음속 [cm/s]  (config.sound_speed_cm_s 와 같은 값)
MU = 1.86e-4           # 점성계수 [g/(cm*s)]
GAMMA = 1.4            # 비열비
PRANDTL = 0.72         # 프란틀 수

#: 벽면 진동 손실. 성도 벽은 단단하지 않아서 저역 포먼트의 대역폭 대부분이
#: 여기서 온다(점성·열전도 손실만으로는 F1 대역폭이 실측의 1/5 밖에 안 된다).
#: 값은 Fant(1972)/Flanagan(1972)이 쓰는 벽 임피던스 단위면적당 상수다.
WALL_RESISTANCE = 1600.0     # [dyn*s/cm^3]
WALL_MASS = 1.5              # [g/cm^2]

_SERIES_TERMS = 24           # J1 / StruveH1 정급수 항 수 (§ _bessel_j1 주석)


def _bessel_j1(x: torch.Tensor) -> torch.Tensor:
    """J1(x) 를 정급수로. torch 에 없어서 직접 쓴다(미분가능해야 하므로).

    J1(x) = Σ_m (−1)^m / (m! (m+1)!) · (x/2)^(2m+1)

    24 항이면 x ≤ 10 에서 float64 오차 1.3e-13, float32 5.6e-5 다(scipy 대조).
    실제로 필요한 x 의 상한은 6.9 뿐이다 — 12 kHz, 면적 8 cm² 일 때 x=2ka=6.88.
    """
    h = 0.5 * x
    term = h.clone()
    total = torch.zeros_like(x)
    for m in range(_SERIES_TERMS):
        total = total + term
        term = -term * h * h / ((m + 1) * (m + 2))
    return total


def _struve_h1(x: torch.Tensor) -> torch.Tensor:
    """Struve H1(x) 를 정급수로.

    H1(x) = (x/2)^2 · Σ_k (−1)^k (x/2)^(2k) / [Γ(k+3/2) Γ(k+5/2)]
    """
    h = 0.5 * x
    p = h * h
    total = torch.zeros_like(x)
    for k in range(_SERIES_TERMS):
        c = math.exp(-math.lgamma(k + 1.5) - math.lgamma(k + 2.5)) * ((-1) ** k)
        total = total + c * p
        p = p * h * h
    return total


def radiation_impedance(freq: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """무한 배플 피스톤의 방사 임피던스. 관 특성임피던스 ρc/A 로 **정규화**된 값.

    freq: (F,) [Hz],  area: (..., 1) [cm^2]  ->  (..., F) 복소

    정확식을 쓴다(근사가 아니다):

        z(x) = [1 − 2·J1(x)/x] + j·[2·H1(x)/x],   x = 2ka,  a = sqrt(A/π)

    집중정수 근사(Flanagan 의 병렬 R-L)도 재 봤는데 반사계수 오차가 면적에 따라
    최대 |ΔΓ| = 0.20 이었다. 나중에 결과가 이상할 때 "근사 탓인가" 를 되묻게
    되는 크기라, 정확식을 급수로 계산하는 쪽을 택했다.

    극한이 맞는지: ω→0 이면 z→0(개방단, Γ→−1), ω→∞ 이면 z→1(무반사, Γ→0).
    """
    a = (area.clamp_min(1e-9) / math.pi).sqrt()                  # 등가 반지름
    k = (2.0 * math.pi / SOUND_SPEED) * freq                     # (F,)
    x = (2.0 * k) * a                                            # (..., F)
    x = x.clamp_min(1e-9)
    r = 1.0 - 2.0 * _bessel_j1(x) / x
    i = 2.0 * _struve_h1(x) / x
    return torch.complex(r, i)


@dataclass
class TubeLosses:
    """벽면 손실을 켜고 끈다. 끄면 이상적인 강체벽 무손실 관이다.

    두 종류가 있고 **지배하는 대역이 다르다**:

    * 점성·열전도 경계층 (`viscothermal`) — sqrt(f) 로 커진다. 고역에서 지배적.
    * 벽 진동 (`yielding_wall`) — 성도 벽이 무겁지만 단단하지는 않다. 저역에서
      지배적이고, **F1 대역폭의 대부분이 여기서 온다.** 이걸 빼면 F1 이
      실측(50~90 Hz)보다 훨씬 좁게 나온다.
    """
    viscothermal: bool = True
    yielding_wall: bool = True


def _propagation(freq: torch.Tensor, area: torch.Tensor, losses: TubeLosses):
    """(복소 파수 k, 복소 특성임피던스 Z) 를 낸다. 손실이 여기 들어간다.

    freq: (F,), area: (..., 1)  ->  둘 다 (..., F) 복소
    """
    w = 2.0 * math.pi * freq
    a = (area.clamp_min(1e-9) / math.pi).sqrt()                  # 반지름
    k0 = w / SOUND_SPEED
    kc = torch.complex(k0.expand(a.shape[:-1] + k0.shape).clone(),
                       torch.zeros_like(k0).expand(a.shape[:-1] + k0.shape).clone())
    z0 = RHO * SOUND_SPEED / area.clamp_min(1e-9)                # (..., 1)
    zc = torch.complex(z0.expand(a.shape[:-1] + k0.shape).clone(),
                       torch.zeros_like(z0).expand(a.shape[:-1] + k0.shape).clone())

    if losses.viscothermal:
        # Kirchhoff 경계층 감쇠 [Np/cm]:
        #   α = (1/(r·c))·sqrt(ω·μ/(2ρ))·(1 + (γ−1)/sqrt(Pr))
        alpha = (torch.sqrt(w * MU / (2.0 * RHO))
                 * (1.0 + (GAMMA - 1.0) / math.sqrt(PRANDTL))
                 / (a * SOUND_SPEED))
        kc = kc - 1j * alpha.to(kc.dtype)

    if losses.yielding_wall:
        # 벽 어드미턴스 Y_w = S_wall / (R_w + jωM_w) 를 관 벽에 분포시킨다.
        # 전파상수에 미치는 영향(1차): k^2 -> k^2 − jωρ·Y_w/A  (Flanagan 1972)
        circumference = 2.0 * math.pi * a
        yw = circumference / torch.complex(
            torch.full_like(w, WALL_RESISTANCE).expand(a.shape[:-1] + w.shape).clone(),
            (w * WALL_MASS).expand(a.shape[:-1] + w.shape).clone())
        k2 = kc * kc - 1j * (RHO * w).to(kc.dtype) * yw / area.clamp_min(1e-9)
        kc = torch.sqrt(k2)
        kc = torch.where(kc.real < 0, -kc, kc)        # 물리적 분지 선택
    return kc, zc


def tract_transfer(area: torch.Tensor, sample_rate: float, n_freq: int,
                   length_cm: float = 17.5,
                   losses: TubeLosses | None = None,
                   radiate: bool = True,
                   radiation_derivative: bool = True) -> torch.Tensor:
    """면적함수 -> 성문 유량에서 원거리 음압까지의 전달함수 (복소).

    area: (B, T, N) [cm^2].  **규약: area[..., 0] = 성문쪽, [..., -1] = 입술쪽**
    (`tract.area_to_reflection` 과 같다).  반환 (B, T, n_freq).

    입술에서 성문 쪽으로 거꾸로 전파시킨다. 입술에서 U=1 로 두면
    P = Z_rad 이고, 각 단면의 전달행렬을 곱해 올라가면 성문에서의 U_g 가
    나온다. 구하려는 것은 U_lips/U_g = 1/U_g 다. 2×2 행렬을 통째로 곱하는
    것보다 2-벡터 하나를 옮기는 쪽이 싸다.

    `radiation_derivative=True` 면 단극 방사의 jω 를 곱한다. 성문 소스가
    유량의 **미분**을 내는 경로(`dsp/glottal.py` 의 LF)에서는 **꺼야 한다** —
    안 그러면 +6 dB/oct 가 두 번 걸린다.
    """
    losses = losses or TubeLosses()
    b, t, n = area.shape
    f = freq_grid(n_freq, sample_rate, device=area.device, dtype=area.dtype)
    # **DC 를 0 으로 두면 안 된다.** f=0 에서 파수 k 가 0 이 되고, 벽 손실이
    # 켜져 있으면 k = sqrt(k²) 의 미분이 거기서 발산해서 **기울기 전체가
    # NaN** 이 된다(역추정에 쓸 것이므로 치명적이다). DC 빈의 값은 어차피
    # 방사 미분 jω 가 0 으로 만들므로, 아주 작은 값으로 밀어도 무해하다.
    f = f.clamp_min(1e-3)
    seg = length_cm / n

    a_l = area[..., -1:]                                        # 입술 단면
    if radiate:
        zr = radiation_impedance(f, a_l) * (RHO * SOUND_SPEED
                                            / a_l.clamp_min(1e-9)).to(torch.complex64)
    else:
        zr = torch.zeros(b, t, n_freq, dtype=torch.complex64, device=area.device)

    p = zr                                                       # 입술: U=1, P=Z_r
    u = torch.ones_like(p)
    for i in range(n - 1, -1, -1):                               # 입술 -> 성문
        kc, zc = _propagation(f, area[..., i:i + 1], losses)
        kl = kc * seg
        cs, sn = torch.cos(kl), torch.sin(kl)
        p, u = cs * p + 1j * zc * sn * u, 1j * sn / zc * p + cs * u

    h = 1.0 / u                                                  # U_lips / U_glottis
    if radiation_derivative:
        h = h * (1j * 2.0 * math.pi * f).to(h.dtype)
    return h


def formant_peaks(h: torch.Tensor, sample_rate: float, n_peaks: int = 5):
    """전달함수의 **낮은 쪽부터** n 개 국소 최대(포먼트) [Hz]. (B, T, n_peaks)

    **크기 상위 n 개를 고르면 안 된다.** `tract.formants_from_response` 가 그렇게
    하는데, 무손실 관처럼 모든 첨두가 다 클 때는 어느 것이 뽑힐지가 수치오차로
    정해진다(실제로 균일관 검증에서 500/1500/2500 대신 1500/4500/7500 이
    나왔다). 포먼트는 정의상 **낮은 쪽부터**다.
    """
    mag = h.abs()
    peak = (mag[..., 1:-1] > mag[..., :-2]) & (mag[..., 1:-1] > mag[..., 2:])
    f = freq_grid(h.shape[-1], sample_rate, device=h.device)[1:-1]
    # DC 근처의 잔물결은 포먼트가 아니다. 성인 F1 하한(약 200 Hz)보다 한참
    # 아래인 50 Hz 미만은 버린다 — 안 버리면 폐쇄 구간에서 8 Hz 짜리 첨두가
    # 'F1' 으로 뽑힌다.
    peak = peak & (f > 50.0)
    big = f.max() * 2.0
    pos = torch.where(peak, f.expand_as(peak), torch.full_like(peak, big,
                                                              dtype=f.dtype))
    return pos.topk(min(n_peaks, pos.shape[-1]), dim=-1, largest=False).values


def formant_bandwidths(h: torch.Tensor, sample_rate: float, n_peaks: int = 5):
    """각 포먼트의 −3 dB 대역폭 [Hz]. 손실 모델이 맞는지 보는 데 쓴다.

    실측 기준(Fant 1972, 남성 모음): F1 40~90, F2 50~110, F3 70~140 Hz.
    """
    mag = h.abs()[0, 0].numpy()
    import numpy as np
    f = np.linspace(0.0, sample_rate / 2, h.shape[-1])
    out = []
    idx = [i for i in range(1, len(mag) - 1)
           if mag[i] > mag[i - 1] and mag[i] > mag[i + 1]][:n_peaks]
    for i in idx:
        half = mag[i] / math.sqrt(2.0)
        lo = i
        while lo > 0 and mag[lo] > half:
            lo -= 1
        hi = i
        while hi < len(mag) - 1 and mag[hi] > half:
            hi += 1
        out.append(float(f[hi] - f[lo]))
    return out
