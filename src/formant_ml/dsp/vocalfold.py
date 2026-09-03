"""성대 자가진동 물리모델 (2-mass / body-cover 계열).

Steinecke & Herzel (1995) 의 비대칭 2질량 모델을 기반으로 한다. 이 모델은
소스를 '주기파형 사전'이 아니라 진짜 미분방정식의 극한주기해로 만들기 때문에
다음이 자동으로 따라온다.

* 진동 모드(mode) 제어 : 강성 파라미터 q 와 좌우 비대칭 a 를 움직이면
  1:1 모드락 -> 서브하모닉(2:1, 3:1) -> 비주기(biphonic) 로 분기(bifurcation)가 생긴다.
  성구(chest/falsetto), 성대 결절성 음성, 이중음(diplophonia)이 같은 방정식에서 나온다.
* 지터/시머 : 인위적 난수 없이 방정식 자체의 과도상태에서 자연스럽게 생긴다.

주의: 샘플 단위 시간루프이므로 느리다. 기본 학습 경로는 LF 사전(glottal.py)이고,
이 모듈은 (a) 데이터 증강/사전학습용 시뮬레이터, (b) LF 파라미터의 물리적 해석,
(c) 짧은 구간의 truncated BPTT 미세조정에 쓰도록 설계했다.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FoldParams:
    """모두 CGS 단위 (g, cm, s). 기본값은 성인 남성 성대 근사."""
    m1: float = 0.125       # 하부(cover) 질량 [g]
    m2: float = 0.025       # 상부(cover) 질량 [g]
    k1: float = 80_000.0    # 강성 [dyn/cm]
    k2: float = 8_000.0
    kc: float = 25_000.0    # 두 질량 간 결합 강성
    r1: float = 0.02        # 감쇠비
    r2: float = 0.02
    a01: float = 0.05       # 정지 시 성문 면적 [cm^2] (내전 정도)
    a02: float = 0.05
    length: float = 1.4     # 성대 길이 [cm]
    d1: float = 0.25        # 질량의 수직 두께 [cm]
    d2: float = 0.05
    rho: float = 1.14e-3    # 공기밀도 [g/cm^3]
    ps: float = 8_000.0     # 성문하압 [dyn/cm^2] (~8 cmH2O)
    q: float = 1.0          # 전체 긴장도 (성구/음높이 제어). q>1 -> 고음/가성쪽
    asym: float = 1.0       # 좌우 비대칭 (1.0 = 대칭). 낮추면 서브하모닉/이중음
    collision: float = 3.0  # 접촉 시 강성 배수


def simulate(params: FoldParams, n_samples: int, sample_rate: int = 24000,
             oversample: int = 4, device=None, dtype=torch.float64):
    """성대 진동을 적분해 성문 유량 U(t) [cm^3/s] 를 반환. (n_samples,)

    반환: (flow, x1, x2) — 유량과 두 질량의 변위 궤적(모드 분석용).
    """
    p = params
    dt = 1.0 / (sample_rate * oversample)
    q, asym = p.q, p.asym

    k1, k2, kc = p.k1 * q, p.k2 * q, p.kc * q
    m1, m2 = p.m1 / q, p.m2 / q          # 긴장 -> 유효질량 감소 -> 고음
    # 좌우 비대칭: 한쪽 성대의 강성만 스케일 (Steinecke-Herzel 의 파라미터 a)
    k1l, k2l = k1 * asym, k2 * asym

    x1 = torch.zeros((), device=device, dtype=dtype) + 0.01
    x2 = torch.zeros((), device=device, dtype=dtype)
    v1 = torch.zeros((), device=device, dtype=dtype)
    v2 = torch.zeros((), device=device, dtype=dtype)

    flow = torch.zeros(n_samples, device=device, dtype=dtype)
    tr1 = torch.zeros(n_samples, device=device, dtype=dtype)
    tr2 = torch.zeros(n_samples, device=device, dtype=dtype)

    c1 = 2.0 * p.r1 * (m1 * k1) ** 0.5
    c2 = 2.0 * p.r2 * (m2 * k2) ** 0.5
    two_l = 2.0 * p.length
    sqrt_2ps_rho = (2.0 * p.ps / p.rho) ** 0.5

    idx = 0
    for n in range(n_samples * oversample):
        a1 = p.a01 + two_l * x1
        a2 = p.a02 + two_l * x2
        amin = torch.minimum(a1, a2)
        open_ = (amin > 0).to(dtype)

        # 베르누이 압력 (Steinecke & Herzel 1995 식 (3))
        p1 = p.ps * (1.0 - open_ * (amin / a1.abs().clamp_min(1e-6)) ** 2) \
            * (a1 > 0).to(dtype)
        p2 = torch.zeros_like(p1)

        # 접촉(성문 폐쇄) 시 추가 강성
        col1 = torch.where(a1 < 0, p.collision * k1 * (a1 / two_l), torch.zeros_like(a1))
        col2 = torch.where(a2 < 0, p.collision * k2 * (a2 / two_l), torch.zeros_like(a2))

        f1 = (-c1 * v1 - 0.5 * (k1 + k1l) * x1 - kc * (x1 - x2) - col1
              + p.length * p.d1 * p1)
        f2 = (-c2 * v2 - 0.5 * (k2 + k2l) * x2 - kc * (x2 - x1) - col2
              + p.length * p.d2 * p2)

        v1 = v1 + dt * f1 / m1          # semi-implicit Euler (심플렉틱, 안정적)
        v2 = v2 + dt * f2 / m2
        x1 = x1 + dt * v1
        x2 = x2 + dt * v2

        if n % oversample == 0:
            u = torch.clamp(amin, min=0.0) * sqrt_2ps_rho
            flow[idx] = u
            tr1[idx] = x1
            tr2[idx] = x2
            idx += 1
    return flow, tr1, tr2


def cycle_rate(flow: torch.Tensor, sample_rate: int = 24000) -> float:
    """성문 폐쇄 주기의 발생률(Hz). 진동이 완전 주기적이 아닐 때도 안정적이다.

    (자가진동 모델은 서브하모닉 영역에 들어가면 자기상관/YIN 이 1/2, 1/3 배음을
    잡는다. 그건 버그가 아니라 분기 현상이며, 이 함수는 그와 무관하게 실제
    폐쇄 횟수를 센다.)
    """
    op = (flow > 0).to(torch.int8)
    onsets = int(((op[1:] - op[:-1]) == 1).sum())
    return onsets * sample_rate / max(len(flow), 1)


def flow_to_excitation(flow: torch.Tensor) -> torch.Tensor:
    """유량 U(t) -> 유량미분 dU/dt (입술 방사 효과 포함, LF 소스와 동일 규격)."""
    d = torch.zeros_like(flow)
    d[1:] = flow[1:] - flow[:-1]
    m = d.abs().max().clamp_min(1e-9)
    return d / m
