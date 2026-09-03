"""치찰음(sibilant) 필터: /s/ /ʃ/ /z/ 를 6개의 물리 파라미터로 요약한다.

왜 대역게인만으로는 부족한가
----------------------------
난류 노이즈의 스펙트럼을 자유로운 대역게인(n_bands=40)으로만 두면, 학습 모델은
그 40차원을 '이 화자의 /s/ 를 재현하는 어떤 벡터'로 외운다. 외운 벡터는 프레임마다
같은 값이 되기 쉽고, 그 결과 마찰음이 **미세하게 주기적인 텍스처**로 들린다
(사용자가 지적한 그 현상이다: 패턴을 인식하면 패턴을 반복한다).

물리적으로 마찰음 스펙트럼을 결정하는 것은 몇 개 안 된다.

* 협착 앞쪽 공동(front cavity)의 1/4 파장 공진 -> **극(pole)**.
  /s/ 는 앞공동이 1.5 cm 안팎이라 5~8 kHz, /ʃ/ 는 2.5~4 kHz.
  **대역폭이 넓다.** 앞공동은 짧고, 입술로 열려 있어 방사 손실이 크고, 난류원이
  한 점이 아니라 협착 하류에 퍼져 있다. 그래서 사람의 /s/ 는 뾰족한 봉우리가
  아니라 4~10 kHz 의 **넓은 고원**이다(1~11 kHz 스펙트럼 평탄도 대략 0.2~0.4).
  Q 를 8 쯤으로 두면(대역폭 800 Hz) 잡음이 그 공진에서 울려 **음조가 들린다** —
  측정: 평탄도 0.059, 위상을 무작위로 돌려도 같은 값이므로 시간영역 아티팩트가
  아니라 순전히 스펙트럼이 뾰족해서 생기는 소리다.
* 협착 뒤쪽 공동과 설하공(sublingual cavity)의 반공진 -> **영점(zero)**.
* 난류 소스 자체의 기울기 -> **tilt** (dB/oct).

이 셋(6개 숫자)이 치찰음의 정체성이고, 사람마다 다른 것도 정확히 이 숫자들이다
(치열 간격, 혀끝 위치, 앞공동 길이). 그래서

* 화자 지문으로 **추출**할 수 있고 (`analysis/sibilant.py`),
* 스크립트에서 **직접 조종**할 수 있고,
* 남은 자유도(대역게인)는 좁아져서 외워버리기 어렵다.

주기성 문제는 별도로 `roughness`(난류의 시간 변조)와 손실쪽
`unvoiced_periodicity_penalty` 로 막는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .filters import pole_zero_response, rms_normalize, tilt_response


@dataclass
class SibilantParams:
    """모두 (B, T, 1) 텐서 또는 float. 화자 지문이자 스크립트 손잡이."""
    pole_f: torch.Tensor          # 앞공동 공진 [Hz]  (/s/ 5~8k, /ʃ/ 2.5~4k)
    pole_bw: torch.Tensor         # 그 대역폭 [Hz]. 사람은 1500~3500 이 보통이고,
    #                             800 아래로 내리면 잡음이 그 공진에서 울려 음조가 들린다
    zero_f: torch.Tensor          # 반공진 [Hz]
    zero_bw: torch.Tensor
    tilt: torch.Tensor            # 난류 기울기 [dB/oct]
    mix: torch.Tensor             # 0=이 필터 미적용, 1=완전 적용
    roughness: torch.Tensor | None = None   # 난류 시간변조 깊이(주기성 방지)

    @staticmethod
    def constant(shape, pole_f=6500.0, pole_bw=2200.0, zero_f=2600.0, zero_bw=2600.0,
                 tilt=0.0, mix=1.0, roughness=0.12, device=None,
                 dtype=torch.float32) -> "SibilantParams":
        def c(v):
            return torch.full(shape, float(v), device=device, dtype=dtype)
        return SibilantParams(c(pole_f), c(pole_bw), c(zero_f), c(zero_bw),
                              c(tilt), c(mix), c(roughness))

    def to(self, device) -> "SibilantParams":
        f = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in self.__dict__.items()}
        return SibilantParams(**f)


# 관용적 출발점. 실제 화자 값은 analysis/sibilant.py 로 추출한다.
PRESETS = {
    #        pole_f  pole_bw  zero_f  zero_bw  tilt
    "s":    (6600.0, 2200.0,  2900.0, 2600.0,  1.0),
    "sh":   (3300.0, 1600.0,  1600.0, 1800.0,  0.0),
    "z":    (6400.0, 2400.0,  2900.0, 2600.0,  0.0),
    "f":    (7500.0, 3500.0,  1200.0, 2500.0,  1.5),
    "th":   (7000.0, 4000.0,  1000.0, 2500.0,  1.0),
    "h":    (1400.0, 2000.0,   400.0, 1200.0, -1.5),
    "ss":   (7200.0, 1700.0,  3200.0, 2200.0,  2.0),   # 된소리 ㅆ: 더 높고 조금 좁게
}


def preset(name: str, shape, device=None, dtype=torch.float32,
           mix: float = 1.0, roughness: float = 0.12) -> SibilantParams:
    pf, pb, zf, zb, ti = PRESETS[name]
    return SibilantParams.constant(shape, pf, pb, zf, zb, ti, mix, roughness,
                                   device=device, dtype=dtype)


def sibilant_response(p: SibilantParams, sample_rate: float,
                      n_freq: int) -> torch.Tensor:
    """치찰음 필터의 복소 응답 (B, T, n_freq). RMS 정규화되어 있어 게인 중립적.

    `mix` 로 항등응답과 보간하므로 마찰음이 아닌 프레임(mix=0)에서는 아무 일도
    일어나지 않는다.
    """
    H = pole_zero_response(p.pole_f, p.pole_bw, p.zero_f, p.zero_bw,
                           sample_rate, n_freq)
    H = H * tilt_response(p.tilt, sample_rate, n_freq)
    H = rms_normalize(H)
    m = p.mix.clamp(0.0, 1.0).to(H.dtype)
    return (1.0 - m) + m * H


def spectral_moments(mag: torch.Tensor, freqs: torch.Tensor, eps: float = 1e-9):
    """마찰음 스펙트럼의 1~4차 모멘트 (centroid, spread, skew, kurtosis).

    마찰음 음성학에서 화자/음소를 가르는 표준 기술자다. mag: (..., F)
    """
    p = mag.clamp_min(eps)
    p = p / p.sum(-1, keepdim=True)
    m1 = (p * freqs).sum(-1)
    d = freqs - m1.unsqueeze(-1)
    m2 = (p * d.pow(2)).sum(-1)
    sd = m2.clamp_min(eps).sqrt()
    m3 = (p * d.pow(3)).sum(-1) / sd.pow(3).clamp_min(eps)
    m4 = (p * d.pow(4)).sum(-1) / sd.pow(4).clamp_min(eps) - 3.0
    return m1, sd, m3, m4
