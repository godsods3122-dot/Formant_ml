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

from .filters import (pole_zero_response, resonator_response, rms_normalize,
                      skirt_response, tilt_response)


# 실측 기반 출발점. /s/ 값은 실제 녹음(한국어 "스" 2회)의 장기 스펙트럼에
# 이 모형을 경사하강으로 맞춰 얻었다(rmse 2.25 dB). 손으로 찍은 값이 아니다.
#
# 그 결과가 알려 준 것:
#   * 저역 스커트는 거의 평평하다(+3.4 dB/oct). 가파른 삼각형이 아니다.
#   * 고역은 가파르게 떨어진다(-13.5 dB/oct).
#   * 봉우리의 주역은 앞공동 극이 아니라 **앞니 공명 7.2 kHz** 다.
#   * 바닥이 피크 대비 -11 dB 밖에 안 된다 -> 스펙트럼이 전역적으로 깔린다.
#
# **dict 로 둔다.** 튜플 위치로 두었더니 항목을 추가할 때마다 값이 밀리는 사고가
# 반복됐다(roughness 에 -5 가 들어가는 식으로).
PRESETS = {
    "s":  dict(pole_f=3750.0, pole_bw=3800.0, zero_f=2030.0, zero_bw=1150.0,
               tilt=-5.0, slope_lo=3.5, slope_hi=-13.5,
               teeth_f=7200.0, teeth_bw=1020.0, floor_db=-11.0),
    "ss": dict(pole_f=4100.0, pole_bw=3400.0, zero_f=2200.0, zero_bw=1000.0,
               tilt=-4.0, slope_lo=5.0, slope_hi=-15.0,
               teeth_f=7800.0, teeth_bw=800.0, floor_db=-13.0),   # 된소리: 더 날카롭게
    "z":  dict(pole_f=3750.0, pole_bw=3800.0, zero_f=2030.0, zero_bw=1150.0,
               tilt=-5.0, slope_lo=3.5, slope_hi=-13.5,
               teeth_f=7100.0, teeth_bw=1100.0, floor_db=-10.0),
    "sh": dict(pole_f=2300.0, pole_bw=2600.0, zero_f=1250.0, zero_bw=1000.0,
               tilt=-4.0, slope_lo=4.0, slope_hi=-11.0,
               teeth_f=3900.0, teeth_bw=1300.0, floor_db=-12.0),
    "f":  dict(pole_f=5000.0, pole_bw=4000.0, zero_f=1500.0, zero_bw=2000.0,
               tilt=-3.0, slope_lo=2.0, slope_hi=-8.0,
               teeth_f=8000.0, teeth_bw=3000.0, floor_db=-8.0),   # 순치음: 공진이 약하다
    "th": dict(pole_f=5500.0, pole_bw=4500.0, zero_f=1500.0, zero_bw=2200.0,
               tilt=-3.0, slope_lo=2.0, slope_hi=-7.0,
               teeth_f=8500.0, teeth_bw=3500.0, floor_db=-8.0),
    "h":  dict(pole_f=1400.0, pole_bw=2500.0, zero_f=500.0, zero_bw=1200.0,
               tilt=-4.0, slope_lo=2.0, slope_hi=-6.0,
               teeth_f=3000.0, teeth_bw=3000.0, floor_db=-6.0),   # 성문 마찰: 거의 평평
}


@dataclass
class SibilantParams:
    """모두 (B, T, 1) 텐서 또는 float. 화자 지문이자 스크립트 손잡이."""
    pole_f: torch.Tensor          # 앞공동 공진 [Hz]  (/s/ 5~8k, /ʃ/ 2.5~4k)
    pole_bw: torch.Tensor         # 그 대역폭 [Hz]. 사람은 1500~3500 이 보통이고,
    #                             800 아래로 내리면 잡음이 그 공진에서 울려 음조가 들린다
    zero_f: torch.Tensor          # 반공진 [Hz]
    zero_bw: torch.Tensor
    tilt: torch.Tensor            # 난류 기울기 [dB/oct] (전체를 기울인다)
    mix: torch.Tensor             # 0=이 필터 미적용, 1=완전 적용
    # 봉우리 양옆의 직선 스커트 [dB/oct]. 극 하나로는 둥근 돔밖에 안 나온다.
    slope_lo: torch.Tensor | None = None    # 봉우리 아래 상승 기울기 (양수)
    slope_hi: torch.Tensor | None = None    # 봉우리 위 하강 기울기 (음수)
    # 앞니 사이 좁은 틈의 공명. 실제 /s/ 를 재 보면 앞공동 봉우리(6.5 kHz) 말고도
    # 8.5 kHz 부근에 봉우리가 하나 더 있다(측정: 추세 대비 +7.4 dB). 혀끝-앞니
    # 사이로 얕게 빠져나가는 제트가 만드는 휘파람 같은 성분이다.
    teeth_f: torch.Tensor | None = None
    teeth_bw: torch.Tensor | None = None
    #: 앞니(장애물) 공진의 **세기** 0~1. 1 이면 지문 그대로, 0 이면 앞니 공진이
    #: 없고 앞공동 극만 남는다.
    #:
    #: 왜 손잡이가 필요한가: 장애물 다이폴은 제트가 앞니를 때려야 생긴다
    #: (Shadle 1985/1990). 협착이 덜 극단적이면 제트가 느려 다이폴이 약해지고,
    #: 그러면 **앞공동 공진이 드러난다**. 실측에서 같은 화자의 /s/ 가 문맥에
    #: 따라 한 옥타브 다른 게 이것이다 — 짧은 CV 의 '사' 는 4.7~5.6 kHz(앞공동
    #: 극 5274 Hz)에서, 길게 끈 /s/ 는 9.7 kHz(앞니 공명 10015 Hz)에서 봉우리가
    #: 선다. 세기가 고정(ones_like)이면 이 대조가 아예 안 생긴다.
    teeth_gain: torch.Tensor | None = None
    # 직접 방사 바닥 [dB, 피크 대비]. 난류원은 앞공동 공진만 통해 나오는 게
    # 아니라 입 구멍에서 그대로도 방사된다(단극 방사). 이게 없으면 봉우리 밖이
    # 통째로 비어서, 실제 녹음처럼 **스펙트럼이 전역적으로** 깔리지 않는다.
    floor_db: torch.Tensor | None = None
    roughness: torch.Tensor | None = None   # 난류 시간변조 깊이(주기성 방지)

    @staticmethod
    @staticmethod
    def constant(shape, mix: float = 1.0, roughness: float = 0.12,
                 device=None, dtype=torch.float32, **overrides) -> "SibilantParams":
        """상수 파라미터 묶음. 기본값은 실측 /s/ 프리셋에서 가져온다.

        시그니처에 기본값을 또 적어 두지 않는다 — 프리셋과 어긋나기 시작하면
        어느 쪽이 진짜인지 알 수 없게 된다(실제로 그렇게 됐었다).
        """
        vals = dict(PRESETS["s"])
        unknown = set(overrides) - set(vals)
        if unknown:
            raise TypeError(f"모르는 치찰음 파라미터: {sorted(unknown)}")
        vals.update(overrides)

        def c(v):
            return torch.full(shape, float(v), device=device, dtype=dtype)
        # 반드시 키워드로 만든다(위치인자면 필드를 끼워 넣을 때 값이 밀린다).
        return SibilantParams(mix=c(mix), roughness=c(roughness),
                              **{k: c(v) for k, v in vals.items()})

    def to(self, device) -> "SibilantParams":
        f = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in self.__dict__.items()}
        return SibilantParams(**f)


def preset(name: str, shape, device=None, dtype=torch.float32,
           mix: float = 1.0, roughness: float = 0.12) -> SibilantParams:
    return SibilantParams.constant(shape, mix=mix, roughness=roughness,
                                   device=device, dtype=dtype, **PRESETS[name])


def sibilant_response(p: SibilantParams, sample_rate: float,
                      n_freq: int) -> torch.Tensor:
    """치찰음 필터의 복소 응답 (B, T, n_freq). RMS 정규화되어 있어 게인 중립적.

    `mix` 로 항등응답과 보간하므로 마찰음이 아닌 프레임(mix=0)에서는 아무 일도
    일어나지 않는다.
    """
    H = pole_zero_response(p.pole_f, p.pole_bw, p.zero_f, p.zero_bw,
                           sample_rate, n_freq)
    if p.slope_lo is not None and p.slope_hi is not None:
        H = H * skirt_response(p.pole_f, p.slope_lo, p.slope_hi,
                               sample_rate, n_freq)
    H = H * tilt_response(p.tilt, sample_rate, n_freq)
    if p.teeth_f is not None and p.teeth_bw is not None:
        # resonator_response 는 단(stage)축을 남긴 (B,T,K,F) 를 돌려준다. 곱해서 접는다.
        # 혼합 가중치는 **단 축을 접은 뒤** 곱하므로 (B,T,1) 이어야 한다.
        # `ones_like(teeth_f)` 로 두면 앞니 공진을 두 개 이상 줄 때 모양이
        # 어긋난다(K=1 을 가정한 잠재 버그였다).
        tg = (torch.ones_like(p.teeth_f[..., :1]) if p.teeth_gain is None
              else p.teeth_gain[..., :1].clamp(0.0, 1.0))
        # 공진을 **병렬로** 섞는다: (1-g)·평탄 + g·공진. 게인을 공진기 자체에
        # 곱하면 세기를 줄일 때 그 대역이 통째로 파여서 스펙트럼에 구멍이 난다.
        # 여기서 원하는 건 "앞니 공진이 덜 도드라진다" 이지 "그 대역이 없다" 가
        # 아니다.
        teeth = resonator_response(p.teeth_f, p.teeth_bw,
                                   torch.ones_like(p.teeth_f), sample_rate,
                                   n_freq).prod(dim=2)
        H = H * ((1.0 - tg).to(teeth.dtype) + tg.to(teeth.dtype) * teeth)
    if p.floor_db is not None:
        # 입 구멍에서 직접 방사되는 광대역 성분을 **병렬로** 더한다.
        # (max 로 자르지 않는다 — 병렬 경로의 합이 물리적으로 맞고 미분도 매끄럽다.)
        #
        # **이 바닥도 소스 기울기(tilt)를 받는다.** 같은 난류원이 앞공동을 안 거치고
        # 나오는 경로라, 소스의 색은 똑같이 입혀져야 한다. 평탄하게 두면 고역에서
        # 필터를 뚫고 나와 스펙트럼이 floor_db 에서 **하강을 멈춘다**.
        # 측정(긴 /s/): 18~22 kHz 가 실측 대비 12~23 dB 과다했고, 더 나쁜 건
        # 제트가 느린 양 끝에서 바닥이 상대적으로 커져 **무게중심이 거꾸로
        # 올라간다**는 것이다(합성 8682 -> 8387 -> 10308 Hz, 실측은
        # 6521 -> 8770 -> 7339 의 아치다).
        #
        # 기울기를 **바닥에만** 건다(필터 전체가 아니라). 앞공동 필터 경로는
        # 예전과 똑같이 두어야 추출기가 적합한 pole_f/zero_f 가 그대로 복원된다
        # (tests/test_voice.py::test_extracted_profile_recovers_what_was_synthesized).
        peak = H.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
        H = H / peak.to(H.dtype)
        H = H + (10.0 ** (p.floor_db.clamp(-60.0, -3.0) / 20.0)).to(H.dtype) \
            * tilt_response(p.tilt, sample_rate, n_freq)
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
