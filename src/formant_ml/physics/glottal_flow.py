"""성문 유동 — 준정상 베르누이 + **유출 분리** + 점성.

성문은 단면이 아래에서 위로 변하는 짧은 관이다. 압력이 유량을 만들고, 유량이 만든 압력
분포가 다시 성대를 민다. 그 되먹임이 자려 진동이므로 **압력 분포**가 유량만큼 중요하다.

세 가지를 같이 본다.

1. **베르누이** — 성문하에서 최소 단면까지는 손실 없이 가속한다.
   ΔP = (ρ/2)·U²·(1/a_min² − 1/a_sub²)

2. **유출 분리** — 발산부에서 제트가 벽에서 떨어진다. 떨어진 뒤로는 압력이 회복되지 않고
   성문상 압력 그대로다. 분리 자리는 단면이 최소의 `SEP_RATIO` 배가 되는 곳이다
   (Pelorson 1994; Story & Titze 1995 가 쓴 1.1~1.2). **이 비대칭이 없으면 자려 진동이
   일어나지 않는다** — 수렴형과 발산형에서 같은 압력을 받으면 한 주기의 알짜 일이 0 이다.

3. **점성** — 틈이 좁아지면 푸아죄유 항이 지배한다. 닫히는 순간 유량이 발산하지 않게 한다.
   평행판(폭 ℓ, 틈 2h, 길이 d) 의 유량은 Q = ℓ(2h)³ΔP/(12μd) 다. a = 2hℓ 로 쓰면

       ΔP_visc = 12·μ·d·ℓ² / a³ · U

   처음에 ℓ 을 한 번만 곱해서 ℓ 배 작게 잡았고, 검증도 **같은 식을 같은 식으로** 비교해서
   1.000 이 나왔다 — 순환논법이었다. 아래 `_check` 는 h 로 쓴 평행판 공식을 따로 세워 잰다.

둘을 합치면 U 에 대한 이차식이고, 양의 근을 쓴다.
"""
from __future__ import annotations

import numpy as np

#: 날숨 공기 (37 °C, 포화습도, CO₂ 5 %).
RHO_AIR = 1.12          # kg/m³
MU_AIR = 1.90e-5        # Pa·s

#: 분리 문턱 — 단면이 최소의 이 배가 되는 곳에서 제트가 벽을 떠난다.
SEP_RATIO = 1.2

#: 성문하 단면 [m²]. 성문보다 훨씬 넓어 가속 항에서 사실상 무시된다.
A_SUB = 2.0e-4


def flow(area, dz, length, p_sub, p_sup=0.0):
    """단면 분포 `area`(아래→위, [m²]) 에서 유량과 벽 압력.

    반환 `(U, p_wall, i_sep)` — 유량 [m³/s], 각 단면에서의 압력 [Pa], 분리 지점 색인.
    """
    a = np.asarray(area, float)
    a = np.maximum(a, 1e-12)
    i_min = int(np.argmin(a))
    # 분리: 최소 단면 **뒤쪽**에서 처음으로 SEP_RATIO 배를 넘는 곳
    i_sep = len(a) - 1
    for i in range(i_min, len(a)):
        if a[i] > SEP_RATIO * a[i_min]:
            i_sep = i
            break
    a_sep = a[i_sep]
    dp = float(p_sub - p_sup)
    if dp <= 0.0:
        return 0.0, np.full_like(a, float(p_sub)), i_sep
    # 이차식  A·U² + B·U − dp = 0
    A = 0.5 * RHO_AIR * (1.0 / a_sep ** 2 - 1.0 / A_SUB ** 2)
    A = max(A, 1e-12)
    d = dz * (i_sep + 1)                       # 분리까지의 길이
    B = 12.0 * MU_AIR * d * length ** 2 / max(a[:i_sep + 1].min() ** 3, 1e-30)
    U = (-B + np.sqrt(B * B + 4.0 * A * dp)) / (2.0 * A)
    # 벽 압력: 분리 전은 베르누이, 분리 후는 성문상 압력 그대로
    p = np.empty_like(a)
    for i in range(len(a)):
        if i <= i_sep:
            p[i] = p_sub - 0.5 * RHO_AIR * U * U * (1.0 / a[i] ** 2 - 1.0 / A_SUB ** 2)
        else:
            p[i] = p_sup
    return float(U), p, i_sep


def duct_area(h, length):
    """반틈새 `h`(아래→위, [m], 한쪽 성대의 변위 기준) -> 단면 [m²]. 음수는 접촉."""
    return 2.0 * np.maximum(np.asarray(h, float), 0.0) * float(length)


def _check() -> int:
    """검증. **해석해는 이 파일의 식이 아니라 독립적으로 세운 것이어야 한다.**"""
    global MU_AIR
    ok = True
    L, dz = 11e-3, 0.5e-3

    # 1. 무점성 극한 — 점성을 꺼서 순수 베르누이와 비교한다.
    mu0 = MU_AIR
    MU_AIR = 0.0
    a = np.full(6, 1.0e-5)
    U, p, i = flow(a, dz, L, 800.0)
    want = a[0] * np.sqrt(2 * 800.0 / (RHO_AIR * (1 - (a[0] / A_SUB) ** 2)))
    MU_AIR = mu0
    print(f"  1 무점성 베르누이  U {U*1e6:8.2f} cm³/s   해석 {want*1e6:8.2f}   "
          f"비 {U/want:5.3f}")
    ok &= abs(U / want - 1.0) < 0.02

    # 2. 점성 극한 — **h 로 쓴 평행판 공식**을 따로 세워 비교한다 (순환 아님).
    h = 2.0e-6                                   # 반틈새 2 µm
    a2 = duct_area(np.full(6, h), L)
    U2, _, _ = flow(a2, dz, L, 100.0)
    d = dz * 6
    want2 = L * (2 * h) ** 3 * 100.0 / (12 * MU_AIR * d)
    print(f"  2 좁은 틈 푸아죄유 U {U2*1e12:8.3f} µm³/s 해석 {want2*1e12:8.3f}   "
          f"비 {U2/want2:5.3f}")
    ok &= abs(U2 / want2 - 1.0) < 0.05

    # 3. 분리: 발산 관에서 분리 뒤 압력이 회복되지 않는다
    a3 = np.array([2.0, 1.0, 1.5, 2.0, 3.0, 4.0]) * 1e-5
    U3, p3, i3 = flow(a3, dz, L, 800.0)
    print(f"  3 발산 관         최소 색인 1, 분리 색인 {i3}   분리 뒤 압력 "
          f"{np.round(p3[i3+1:], 3)}")
    ok &= i3 == 2 and np.allclose(p3[i3 + 1:], 0.0)

    # 4. 단조성
    us = [flow(np.full(6, 1e-5), dz, L, p)[0] for p in (200, 400, 800, 1600)]
    print(f"  4 압력 단조       U {[round(u*1e6,1) for u in us]}")
    ok &= all(us[i] < us[i + 1] for i in range(len(us) - 1))

    # 5. 수렴형과 발산형의 벽 압력 차 — 자려 진동의 원천
    conv = np.array([3.0, 2.2, 1.6, 1.2, 1.0, 1.0]) * 1e-5      # 수렴 (아래가 넓다)
    dive = conv[::-1].copy()                                     # 발산
    _, pc, _ = flow(conv, dz, L, 800.0)
    _, pd, _ = flow(dive, dz, L, 800.0)
    print(f"  5 수렴 평균 벽압 {pc.mean():7.1f} Pa   발산 {pd.mean():7.1f} Pa   "
          f"차 {pc.mean()-pd.mean():+7.1f}")
    ok &= pc.mean() > pd.mean()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_check())
