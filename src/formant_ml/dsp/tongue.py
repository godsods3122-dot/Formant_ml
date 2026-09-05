"""혀끝의 유동유발 진동 — 접근음·탄음·전동음을 한 방정식에서 낸다.

왜 이 구조인가
--------------
접근음 [ɹ], 탄음 [ɾ], 전동음 [r] 을 세 개의 코드 경로로 나누면 안 된다.
셋은 **같은 물리계의 서로 다른 동작점**이다. 혀끝은 구개 아래에 매달린
질량-스프링이고, 그 사이 틈으로 기류가 지난다. 틈이 좁아지면 베르누이 저압이
혀끝을 구개 쪽으로 빨아당기고, 탄성이 되민다. 채널 속 공기의 **관성(inertance)**
이 압력을 변위보다 늦게 따라오게 만들어서, 조건이 맞으면 이 되먹임이 극한주기
(limit cycle)가 된다. 성대의 자가진동과 같은 종류의 유동유발 진동이고
(inward-striking reed), 그래서 `vocalfold.py` 와 형제 모듈이다.

    목표 간극 h0(t) 하나가 조음 방식을 정한다 (방식은 파라미터가 아니라 결과다)

      h0 크게 유지            -> 접촉 없음, 정상 유동      -> 접근음 [ɹ]
      h0 작게 유지 + 낮은 강성 -> 극한주기, 접촉이 여러 번  -> 전동음 [r]
      h0 를 짧게 0 아래로     -> 접촉 한 번                -> 탄음 [ɾ]

전동음의 접촉 횟수를 우리가 세지 않는다. 방정식이 센다. 실제 전동음이 압력이
모자라면 탄음으로 무너지는 것(Solé 2002)도 같은 이유로 자동으로 나온다.

**현재 상태 (2026-09-05 측정): 접근음·탄음 분기는 검증되었고 전동음 분기는 아직
아니다.** 탄음은 접촉 1 회 · 폐쇄 37.7 ms 로 실측대(20~50 ms, Cathcart 2012 의
"약 1/24 초")에 들어온다. 접근음은 접촉 0 회. 그러나 유지 자세에서 나오는
자가진동은 25~35 Hz 가 아니라 느린 가지(8~16 Hz)와 빠른 가지(64~80 Hz)로 갈리고
그 사이가 비어 있다 — 목표 대역이 정확히 그 틈에 있다. 접촉 감쇠를 올리면 진동이
30 Hz 로 모이는 게 아니라 아예 죽는다(측정: cd=80 에서 접촉 1 회).

원인은 파라미터가 아니라 **구조**다. 혀끝을 질량 하나로 두었기 때문이다.
`vocalfold.py` 의 서두에 이미 적혀 있는 것과 같은 이유다 — 흡입이 닫는
inward-striking 계는 준정상 베르누이 + 단일 질량으로는 순 에너지를 못 받아
자가진동하지 않는다. 성대가 수직 위상차(mucosal wave) 때문에 2 질량이 필요하듯,
혀끝도 **앞뒤 위상차**가 필요하다. 근거도 있다: Cathcart(2012)는 flap 이
back-to-front 로 움직인다고 보고한다. 다음 작업은 혀끝을 후단-전단 2 질량으로
바꾸는 것이고, 그러면 `simulate()` 와 같은 구조가 된다.

측면 통로(설측음)는 여기서 다루지 않는다. 그건 혀끝 운동이 아니라 **혀 옆의
개구**이고, 음향적으로는 극-영점으로 나타나므로 `filters.antiresonator_response`
쪽 일이다. 이 모듈은 정중선(mid-sagittal) 협착만 담당한다.

단위는 `vocalfold.py` 와 같은 CGS (g, cm, s, dyn).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TipParams:
    """혀끝 유효 질량-스프링. 기본값은 성인 전동음이 25~35 Hz 로 떨도록 잡았다.

    f_n = sqrt(k/m)/(2*pi) 가 전동음 실측 대역(스페인어 /r/ 약 26~32 Hz)에
    오도록 m, k 를 정한다. 이건 곡선 맞추기가 아니라 진동수를 물리 상수로
    환산한 것이다 — 질량과 강성 중 하나를 바꾸면 진동수가 따라 움직인다.
    """
    mass: float = 0.9            # 혀끝 유효질량 [g]
    stiffness: float = 3.55e4    # 탄성 [dyn/cm]  -> f_n ~= 30 Hz
    damping_ratio: float = 0.08  # 조직 감쇠
    width: float = 1.2           # 채널 폭(좌우) [cm]
    depth: float = 0.5           # 협착 길이(앞뒤) [cm]
    po: float = 8000.0           # 협착 뒤 구강내압 [dyn/cm^2] (~8 cmH2O)
    rho: float = 1.14e-3         # 공기밀도 [g/cm^3]
    collision: float = 3.0       # 구개 접촉 시 강성 배수
    collision_damp: float = 15.0 # 접촉 감쇠 배수. 조직은 비탄성이라 튀지 않는다
    mu: float = 1.86e-4          # 공기 점성계수 [g/(cm*s)]
    gap_floor: float = 2e-4      # 수치 안정용 최소 간극 [cm]


def simulate_tip(
    h0: torch.Tensor,
    params: TipParams = TipParams(),
    sample_rate: int = 24000,
    oversample: int = 8,
    device=None,
    dtype=torch.float64,
):
    """목표 간극 궤적 h0(t) [cm] 를 받아 실제 간극과 유량을 적분한다.

    h0: (n_samples,) 샘플률 목표 간극. 양수 = 구개에서 떨어짐, 음수 = 눌러붙임.
    반환 dict:
      gap      (n,) 실제 간극 [cm] (음수면 접촉)
      area     (n,) 협착 단면적 [cm^2]
      flow     (n,) 체적유량 [cm^3/s]
      contact  (n,) 접촉 여부 0/1

    적분은 semi-implicit Euler (심플렉틱). 채널 유량 u 를 별도 상태로 두어
    관성 지연을 주는 것이 자가진동의 핵심이다 — 준정상(quasi-steady) 베르누이만
    쓰면 위상 지연이 없어 이 계는 절대 떨지 않는다.
    """
    p = params
    n = int(h0.shape[-1])
    dt = 1.0 / (sample_rate * oversample)
    h0 = h0.to(device=device, dtype=dtype)

    a_wall = p.width * p.depth                 # 혀끝 윗면 면적 [cm^2]
    c = 2.0 * p.damping_ratio * (p.mass * p.stiffness) ** 0.5
    visc_c = 12.0 * p.mu * p.depth / p.width   # Poiseuille 계수(슬릿)

    h = h0[0].clone()
    v = torch.zeros((), device=device, dtype=dtype)
    u = torch.zeros((), device=device, dtype=dtype)

    gap = torch.zeros(n, device=device, dtype=dtype)
    area = torch.zeros(n, device=device, dtype=dtype)
    flow = torch.zeros(n, device=device, dtype=dtype)

    idx = 0
    for k in range(n * oversample):
        tgt = h0[min(idx, n - 1)]
        hc = h.clamp_min(p.gap_floor)
        a = p.width * hc
        open_ = (h > 0).to(dtype)

        # 채널 압력강하 = 관성 + 점성 + 동압.
        #   관성(inertance)  : 압력을 변위보다 늦게 만든다 = 자가진동의 위상 지연
        #   점성(Poiseuille) : 간극^-3 으로 발산해 닫힐 때 유량을 물리적으로 끈다.
        #                      이게 없으면 h->0 에서 u/a 가 폭주해 계가 발산한다.
        dyn = 0.5 * p.rho * (u / a) ** 2
        r_visc = visc_c / hc ** 3
        du = (a / (p.rho * p.depth)) * (p.po - dyn - r_visc * u)
        u = ((u + dt * du) * open_).clamp_min(0.0)

        # 협착부 압력(베르누이 저압). 진공 아래로는 못 간다 — 물리적 하한을 둔다.
        p_gap = (p.po - dyn).clamp_min(-p.po) * open_ + p.po * (1.0 - open_)
        # 접촉: 강성뿐 아니라 **감쇠**도 준다. 감쇠가 없으면 혀끝이 구개에서
        # 튀어(chatter) 탄음 한 번이 여러 번으로 갈라진다. 실제 조직은 비탄성이다.
        touching = (h < 0).to(dtype)
        col = touching * (-p.collision * p.stiffness * h
                          - p.collision_damp * c * v)

        # 정지 기준을 Po 로 잡아, 유량이 없으면 알짜 공기력이 0 이 되게 한다.
        acc = (-c * v - p.stiffness * (h - tgt)
               + a_wall * (p_gap - p.po) + col) / p.mass
        v = v + dt * acc
        h = h + dt * v

        if k % oversample == 0:
            gap[idx] = h
            area[idx] = p.width * h.clamp_min(0.0)
            flow[idx] = u
            idx += 1
            if idx >= n:
                break

    return {"gap": gap, "area": area, "flow": flow,
            "contact": (gap <= 0).to(dtype)}


def contact_events(contact: torch.Tensor) -> int:
    """접촉 횟수. 탄음 1, 전동음 여러 번, 접근음 0 — 방식의 판별식이다."""
    c = contact.to(torch.int8)
    if len(c) < 2:
        return 0
    return int(((c[1:] - c[:-1]) == 1).sum()) + int(c[0] == 1)


def contact_rate(contact: torch.Tensor, sample_rate: int = 24000) -> float:
    """접촉 발생률 [Hz]. 전동음이면 25~35 Hz 가 나와야 한다."""
    n = contact_events(contact)
    return n * sample_rate / max(len(contact), 1)


def gesture(n_samples: int, points, sample_rate: int = 24000,
            device=None, dtype=torch.float64) -> torch.Tensor:
    """[(시각[s], 목표간극[cm]), ...] -> 샘플률 h0(t). 구간 선형보간."""
    t = torch.arange(n_samples, device=device, dtype=dtype) / sample_rate
    ts = torch.tensor([p for p, _ in points], device=device, dtype=dtype)
    hs = torch.tensor([h for _, h in points], device=device, dtype=dtype)
    i = torch.searchsorted(ts, t.clamp(ts[0], ts[-1])).clamp(1, len(ts) - 1)
    t0, t1 = ts[i - 1], ts[i]
    h0, h1 = hs[i - 1], hs[i]
    w = ((t - t0) / (t1 - t0).clamp_min(1e-9)).clamp(0, 1)
    return h0 + (h1 - h0) * w
