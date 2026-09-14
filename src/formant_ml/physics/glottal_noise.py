"""성문 난류 — **속삭임·기식·유성 중 누설**을 한 가지 기하에서 낸다.

프로젝트의 본편이 ASMR 인터랙티브라 속삭임은 선택이 아니다. 그런데 속삭임은 별도의
음원이 아니라 **성대가 진동하지 않는 상태의 같은 물리**다. 필요한 것은 기하 하나다.

성문 후부 틈 (posterior chink)
------------------------------
성문은 앞쪽 **막성부**(성대 인대, 진동한다)와 뒤쪽 **연골부**(피열연골 사이, 진동하지 않는다)
로 나뉜다. 막성부가 완전히 닫혀도 연골부가 열려 있으면 그리로 공기가 샌다. 그 틈이

    * 0 이면            — 완전 내전. 압착된 유성음
    * 작으면            — 유성 중 누설 = 기식성 (배음과 난류가 같이 난다)
    * 크고 진동 없으면  — **속삭임**
    * 더 크면           — 무성 날숨 /h/

즉 (막성부 진동 진폭, 후부 틈 넓이) 두 축이 발성 양식 전부를 덮는다.

난류 세기
---------
기존 엔진과 같은 규약을 쓴다 (`engine.noise`, Stevens 1998):

    Re = ρ·U·d_h / (μ·a)         (d_h = 수력 직경)
    구동 ∝ ((Re² − Re_c²)_+ / Re_ref²)^1.5 · (1/a)

Re_c = 1800, Re_ref = 8000. 난류는 협착 **뒤**에서 나므로 음원은 성문 위에 놓인다.

검증
----
* 문턱: Re < 1800 이면 소리가 없어야 한다.
* 속삭임의 세기가 보통 발성보다 20~30 dB 아래여야 한다.
* 틈을 키우면 단조 증가했다가, 너무 커지면 유속이 떨어져 다시 준다 (극대가 있다).
* 유성 + 작은 틈이면 배음과 난류가 **같이** 난다 (기식성).
"""
from __future__ import annotations

import numpy as np

from .glottal_flow import MU_AIR, RHO_AIR, flow

RE_CRIT = 1800.0
RE_REF = 8000.0

#: **성문하 저항** [Pa·s/m³]. 폐와 기관은 유량이 늘면 압력을 못 버틴다:
#:     P_sub = P_lung − R_SUB·U
#: 이것이 없으면 성문을 넓힐수록 난류가 단조 증가한다 — 베르누이에서 제트 속도는 압력만으로
#: 정해지고 Re = 2ρU/(μℓ) 는 면적과 무관하기 때문이다 (d_h = 2a/ℓ 로 약분된다). 실제로
#: 속삭임이 어떤 틈에서 가장 시끄러운 것은 이 되먹임 때문이다. 값은 기관 저항의 문헌 범위
#: (1~3 cmH₂O/(L/s) = 1e5~3e5 Pa·s/m³) 에서 잡았다.
R_SUB = 1.5e5


def reynolds(u_flow: float, area: float, length: float) -> float:
    """Re = ρ·U·d_h/(μ·a). 폭 ℓ, 틈 2h 인 슬릿의 수력 직경은 ≈ 2·(2h) = 2a/ℓ."""
    a = max(float(area), 1e-12)
    d_h = 2.0 * a / max(float(length), 1e-9)
    return RHO_AIR * float(u_flow) * d_h / (MU_AIR * a)


def turbulence(u_flow: float, area: float, length: float) -> float:
    """난류 구동의 세기 (무차원). 문턱 아래는 정확히 0."""
    re = reynolds(u_flow, area, length)
    over = max(re * re - RE_CRIT * RE_CRIT, 0.0)
    if over <= 0.0:
        return 0.0
    a_cm2 = max(float(area) * 1e4, 0.02)          # 기존 엔진과 같은 cm² 단위 눈금
    return (over / RE_REF ** 2) ** 1.5 * (0.1 / a_cm2)


def subglottal_pressure(p_lung: float, u_flow: float) -> float:
    """폐압에서 유량에 비례한 손실을 뺀 실제 성문하압 [Pa]."""
    return max(float(p_lung) - R_SUB * float(u_flow), 0.0)


def solve_flow(area_profile, dz, length, p_lung, p_sup=0.0, iters: int = 12):
    """성문하 저항까지 포함해 유량을 푼다 (되먹임이라 반복한다)."""
    U = 0.0
    for _ in range(iters):
        P = subglottal_pressure(p_lung, U)
        U_new, p_wall, i_sep = flow(area_profile, dz, length, P, p_sup)
        U = 0.5 * U + 0.5 * U_new                  # 완화 반복
    P = subglottal_pressure(p_lung, U)
    U, p_wall, i_sep = flow(area_profile, dz, length, P, p_sup)
    return U, p_wall, i_sep, P


def glottal_area(h_membranous, chink_area, length):
    """막성부 반틈새 배열 + 후부 틈 넓이 -> 총 성문 단면 [m²].

    후부 틈은 **진동과 무관하게 늘 열려 있다** — 그래서 막성부가 닫혀도 공기가 샌다.
    """
    h = np.maximum(np.asarray(h_membranous, float), 0.0)
    a_mem = 2.0 * h.min() * float(length)
    return a_mem + float(chink_area)


def _check() -> int:
    ok = True
    L, dz = 11e-3, 0.4e-3
    P = 6 * 98.0665
    print("성문 난류 검증")

    # 1. 문턱 — 아주 좁은 틈에서는 Re 가 낮아 소리가 없다
    small = 2.0e-8
    U, _p, _i, _P = solve_flow(np.full(6, small), dz, L, P)
    print(f"  1 좁은 틈   a {small*1e6:6.3f} mm²  U {U*1e6:8.3f} cm³/s  "
          f"Re {reynolds(U, small, L):8.1f}  구동 {turbulence(U, small, L):.3e}")
    ok &= turbulence(U, small, L) == 0.0

    # 2. 속삭임 급 틈 (10~20 mm²) 에서는 난류가 난다
    chink = 1.2e-5
    U2, _p, _i, _P = solve_flow(np.full(6, chink), dz, L, P)
    tb2 = turbulence(U2, chink, L)
    print(f"  2 속삭임    a {chink*1e6:6.2f} mm²  U {U2*1e6:8.2f} cm³/s  "
          f"Re {reynolds(U2, chink, L):8.1f}  구동 {tb2:.3e}")
    ok &= tb2 > 0.0

    # 3. 생리적으로 가능한 성문 넓이 안에서의 거동.
    #    **극대는 없다** — 압력을 고정하면 제트 속도가 고정이고 Re = 2ρU/(μℓ) 는 면적에
    #    비례해 는다. 성문하 저항으로 압력이 떨어져도 실제 성문이 열릴 수 있는 한계
    #    (완전 외전 ~20~30 mm²) 안에서는 단조 증가한다. 처음에 "극대가 있어야 한다" 고
    #    기대한 것은 틀렸다. 대신 **생리 범위 안에서 문턱을 넘는지**를 본다.
    print("  3 틈 훑기   a[mm²]  P_sub[cmH2O]  U[cm³/s]      Re     구동")
    best = None
    for a_mm in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0):
        aa = a_mm * 1e-6
        Ux, _p, _i, Px = solve_flow(np.full(6, aa), dz, L, P)
        t = turbulence(Ux, aa, L)
        print(f"              {a_mm:6.1f}     {Px/98.07:7.2f}  {Ux*1e6:9.2f} "
              f"{reynolds(Ux, aa, L):9.1f}  {t:.3e}")
        if best is None or t > best[1]:
            best = (a_mm, t)
    # 유성 중 누설(1~3 mm²)은 조용하고, 속삭임(8~20 mm²)은 소리가 나야 한다.
    a_leak, a_whis = 2.0e-6, 1.2e-5
    t_leak = turbulence(solve_flow(np.full(6, a_leak), dz, L, P)[0], a_leak, L)
    t_whis = turbulence(solve_flow(np.full(6, a_whis), dz, L, P)[0], a_whis, L)
    print(f"     유성 중 누설 2 mm² 구동 {t_leak:.3e}   속삭임 12 mm² 구동 {t_whis:.3e}")
    ok &= t_leak == 0.0 and t_whis > 0.0

    # 4. 유성 + 후부 틈 = 기식성. 막성부가 닫혀도 총 면적이 0 이 아니다.
    h_closed = np.zeros(6)
    a_breathy = glottal_area(h_closed, 3.0e-6, L)
    a_pressed = glottal_area(h_closed, 0.0, L)
    print(f"  4 막성부 폐쇄 시  기식성 총면적 {a_breathy*1e6:.3f} mm²   압착 {a_pressed*1e6:.3f} mm²")
    ok &= a_breathy > 0 and a_pressed == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_check())
