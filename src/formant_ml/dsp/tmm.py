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

전송선 형식으로 쓴다
-------------------
단면마다 단위길이당 **직렬 임피던스 Z'** 와 **병렬 어드미턴스 Y'** 를 만들고,
전달행렬을 그 둘로만 쓴다. 파수는 k² = −Z'Y' 로만 나타나고 **k 를 만들지
않는다**(`_entire_cos_sinc` 주석). 두 가지가 동시에 해결된다:

1. k = √(k²) 를 만들면 손실이 있을 때 ω=0 에 분지점이 생겨 **기울기가 NaN**
   이 된다. 한때 주파수를 1e-3 Hz 로 클램프해서 피했는데, 그건 특이점을 안
   밟은 것이지 없앤 게 아니다(모델을 DC 근처에서 조용히 바꾼다).
2. **더 중요한 것**: 예전 판은 손실 k 와 함께 특성임피던스를 무손실값
   ρc/A 로 썼다. 손실 관의 올바른 값은 Z_c = ωρ/(A·k) = √(Z'/Y') 다.
   전송선 형식으로 쓰면 이 불일치가 원천적으로 생길 수 없다.
   전송선 ODE 를 RK4 로 직접 적분한 독립 기준과 대조하면, 현재 식은
   **0.0000 dB**, 예전 식은 최대 **9.5 dB** 틀렸다 — 그것도 첨두 −40 dB 이내,
   즉 들리는 대역에서다. (`test_matches_an_independent_ode_integration`)

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
#: cos(√z)/sinc(√z) 정급수 항 수. 실제 |z| 상한이 0.9 라 8 항이면 이미 1e-14 인데,
#: 단면을 아주 잘게 쪼개거나 손실을 키울 여지를 두어 12 로 둔다.
#: `test_tmm.test_series_argument_stays_in_the_validated_range` 가 상한을 지킨다.
_COS_TERMS = 12


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
    k = ((2.0 * math.pi / SOUND_SPEED) * freq).to(area.dtype)    # (F,)
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


def _entire_cos_sinc(z: torch.Tensor, n_terms: int = _COS_TERMS):
    """(cos(√z), sinc(√z)) 를 z 의 **정급수**로. √z 를 만들지 않는다.

    왜 이렇게 하는가 — 이게 이 모듈에서 제일 중요한 수치 결정이다.

    전달행렬에 파수 k 는 `cos(kl)` 과 `sin(kl)` 로만 들어가고, 둘 다 k 의
    **짝함수**로 쓸 수 있다(cos 는 그 자체로, sin 은 sinc = sin(kl)/(kl) 로).
    즉 답은 k² 에만 의존하는데, k = √(k²) 를 굳이 만들면 **없던 분지점이
    생긴다.** 손실이 있으면 k² ∝ ω 라 ω=0 에서 dk/dω ∝ 1/√ω 로 발산하고,
    자동미분이 그걸 타고 내려와 **기울기 전체가 NaN** 이 된다.

    한때 주파수 격자를 1e-3 Hz 로 클램프해서 피했는데, 그건 특이점을 없앤 게
    아니라 안 밟은 것이다(모델을 DC 근처에서 조용히 바꾼다). 급수로 쓰면
    특이점이 **식에서 사라진다** — 클램프가 필요 없다.

        cos(√z)  = Σ (−1)^n z^n / (2n)!
        sinc(√z) = Σ (−1)^n z^n / (2n+1)!

    둘 다 정함수라 z 전체에서 수렴한다. 실제 |z| 는 (|k|·단면길이)² 이고
    이 레포의 범위에서 1 을 넘지 않는다(테스트가 확인한다). 16 항이면
    |z|=25 에서도 항 크기가 1e-10 이다.
    """
    c = torch.ones_like(z)
    s = torch.ones_like(z)
    tc = torch.ones_like(z)
    ts = torch.ones_like(z)
    for n in range(1, n_terms):
        tc = -tc * z / ((2 * n - 1) * (2 * n))
        ts = -ts * z / ((2 * n) * (2 * n + 1))
        c = c + tc
        s = s + ts
    return c, s


def _series_shunt(freq: torch.Tensor, area: torch.Tensor, losses: TubeLosses):
    """단위길이당 (직렬 임피던스 Z', 병렬 어드미턴스 Y'). freq: (F,), area: (...,1)

    전송선 형식으로 쓰면 손실이 전부 이 둘 안에 들어가고, 파수는 k² = −Z'Y'
    로만 나타난다. **k 를 만들 필요가 없다** (`_entire_cos_sinc` 주석).

        Z' = jωρ/A + R_v          (관성 + 점성 경계층)
        Y' = jωA/(ρc²) + G_t + Y_w (압축성 + 열전도 경계층 + 벽 진동)

    이 형태의 감쇠상수는 α = ½(R_v/Z_c + G_t·Z_c) = (1/(a·c))·√(ωμ/2ρ)·
    (1 + (γ−1)/√Pr) 로, Kirchhoff 경계층 감쇠와 같다(같은 물리를 다르게 쓴 것).
    """
    # **dtype 을 면적 쪽으로 통일한다.** freq 와 area 의 dtype 이 다르면
    # torch.complex 가 그대로 터진다(float64 면적 + float32 주파수에서 실제로
    # 겪었다). 면적이 최적화 변수이므로 그쪽을 기준으로 맞추는 것이 맞다.
    w = (2.0 * math.pi * freq).to(area.dtype)                    # (F,)
    a = (area.clamp_min(1e-9) / math.pi).sqrt()                  # 반지름 (...,1)
    circ = 2.0 * math.pi * a                                     # 둘레
    zero = torch.zeros(a.shape[:-1] + w.shape, dtype=a.dtype, device=a.device)

    z_re = zero.clone()
    z_im = (w * RHO / area.clamp_min(1e-9)).expand_as(zero).clone()
    y_re = zero.clone()
    y_im = (w * area.clamp_min(1e-9) / (RHO * SOUND_SPEED ** 2)).expand_as(zero).clone()

    if losses.viscothermal:
        root = torch.sqrt(w * RHO * MU / 2.0)                    # (F,)
        z_re = z_re + (circ / area.clamp_min(1e-9) ** 2) * root
        y_re = y_re + (circ / (RHO * SOUND_SPEED ** 2)) * (GAMMA - 1.0) \
            * torch.sqrt(w * MU / (2.0 * RHO * PRANDTL))

    zp = torch.complex(z_re, z_im)
    yp = torch.complex(y_re, y_im)

    if losses.yielding_wall:
        # 벽 어드미턴스 Y_w = 둘레 / (R_w + jωM_w) 를 병렬 가지에 더한다.
        yw = circ.to(zp.dtype) / torch.complex(
            torch.full_like(zero, WALL_RESISTANCE),
            (w * WALL_MASS).expand_as(zero).clone())
        yp = yp + yw
    return zp, yp


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
    seg = length_cm / n

    a_l = area[..., -1:]                                        # 입술 단면
    if radiate:
        zr = radiation_impedance(f, a_l) * (RHO * SOUND_SPEED
                                            / a_l.clamp_min(1e-9)).to(torch.complex64)
    else:
        zr = torch.zeros(b, t, n_freq, dtype=torch.complex64, device=area.device)

    # 입술에서 U=1 로 두면 P=Z_rad. 거기서 성문 쪽으로 2-벡터를 옮긴다.
    #   [P_in]   [ cos(kl)      Z'·l·sinc(kl) ] [P_out]
    #   [U_in] = [ Y'·l·sinc(kl)   cos(kl)    ] [U_out]
    # k² = −Z'Y' 이므로 (kl)² = −Z'Y'l² 이고, cos·sinc 는 그 값의 정함수다.
    p = zr
    u = torch.ones_like(p)
    for i in range(n - 1, -1, -1):
        zp, yp = _series_shunt(f, area[..., i:i + 1], losses)
        zl, yl = zp * seg, yp * seg
        cs, sc = _entire_cos_sinc(-zl * yl)                      # (kl)² = −Z'Y'l²
        p, u = cs * p + zl * sc * u, yl * sc * p + cs * u

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
