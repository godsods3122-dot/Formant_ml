"""자려 진동 — 모드 + 유동의 되먹임을 시간적분한다.

`fold_modes` 가 준 고유모드를 일반화 좌표로 삼고, `glottal_flow` 가 준 벽 압력을 그 모드에
투영해 구동력으로 쓴다. 모드마다

    q̈_n + 2ζ_n ω_n q̇_n + ω_n² q_n = f_n(t) / m_n ,
    f_n = ∮ p(y,t) · φ_n(y) dy · ℓ

접촉은 **모드 집합을 갈아 끼우는 것이 아니라** 틈이 0 이하로 내려갈 때 강한 복원력을 주는
벌칙 강성으로 넣는다 (시간적분에서 모드를 갈아 끼우면 에너지가 튄다). 접촉 몫이 크면
그 벌칙이 걸리는 높이가 넓어져 `fold_modes` 의 접촉 경계와 같은 효과가 난다.

검증 (문헌값)
-------------
* **발성 문턱 압력** 3~5 cmH₂O (성인, 보통 음역). 그 아래에서는 진동이 죽어야 한다.
* **개방률** 0.4~0.7.
* **F0 의 압력 의존** 2~5 Hz/cmH₂O.
* 유량 파형이 **비대칭** — 닫히는 쪽이 가파르다 (그것이 음원의 고역을 만든다).
"""
from __future__ import annotations

import numpy as np

from .fold_modes import RHO, bands, geometry
from .glottal_flow import A_SUB, MU_AIR, RHO_AIR, duct_area, flow
from .glottal_noise import RE_CRIT


def _drop(a, i_sep, U, dz, ell):
    """주어진 U 에서의 정상 압력 손실 (베르누이 + 점성)."""
    a_sep = a[i_sep]
    conv = 0.5 * RHO_AIR * U * U * (1.0 / a_sep ** 2 - 1.0 / A_SUB ** 2)
    d = dz * (i_sep + 1)
    visc = 12.0 * MU_AIR * d * ell ** 2 / max(a[:i_sep + 1].min() ** 3, 1e-30) * U
    return conv + visc


def _wall_pressure(a, i_sep, U, p_sub, p_sup=0.0):
    """지금의 U 에서 벽에 걸리는 압력. 분리 뒤는 성문상 압력."""
    p = np.empty_like(a)
    for i in range(len(a)):
        if i <= i_sep:
            p[i] = p_sub - 0.5 * RHO_AIR * U * U * (1.0 / a[i] ** 2 - 1.0 / A_SUB ** 2)
        else:
            p[i] = p_sup
    return p

CM_H2O = 98.0665        # Pa

#: 모드 감쇠비. 성대 조직은 물렁해서 Q 가 낮다 (문헌 ζ ≈ 0.1~0.2).
ZETA = 0.15
#: 접촉 벌칙 강성 [Pa/m] — 틈이 음수로 내려가면 이만큼 되민다. 압력 단위로 쓴다.
K_CONTACT = 2.0e7
#: 접촉 감쇠 [Pa·s/m] — 벌칙만 주면 튄다. 조직은 물렁하므로 닿을 때 에너지를 잃는다.
C_CONTACT = 3.0e3
#: 성문 두께 방향 격자 수.
NZ = 8

#: **점막액의 점착과 압착막** — 수분 상태가 만드는 두 가지 힘 (§52.6).
#:
#: 사용자: *"성대의 수분에 따른 두 판막 점착에 대해서는 고려해봤어? 수분이 점막을 더 붙잡는다면 음성에
#: 더 스무딩 같은 거나, 아니면 분산 등을 만들어낼 수도 있잖아."*
#:
#: 점막 표면에는 점액막(두께 `FILM_H`)이 있다. 두 면의 틈이 막 두께보다 좁아지면
#:
#: 1. **모세관 점착** — 액체 다리의 흡입압 p_cap ≈ −2γ/h 가 두 면을 붙잡는다. γ 0.04 N/m, h 20 µm 면
#:    약 4000 Pa(40 cmH₂O) 로 성문하압보다 크다. 다리는 틈이 `BRIDGE_RUPTURE × FILM_H` 를 넘으면 끊기고,
#:    그 문턱은 주기마다 흔들린다(`BRIDGE_JITTER`) — 열림 시각의 주기 간 변동이 된다.
#: 2. **압착막 감쇠** — 틈이 좁아지며 막이 밀려날 때의 점성 저항 p_sq ≈ −μ_m·(W²/h³)·ḣ (W = 막의 유효 폭).
#:    닫힘을 쿠션처럼 누그러뜨려 음원의 고역을 줄인다.
#:
#: 탈수는 점액의 점도 `MUCUS_MU` 를 올리고 막을 얇고 끈적하게 만든다 — 문헌에서 탈수가 발성 문턱을 올리는 것과
#: 같은 방향이다. 기본은 **꺼짐**(γ = μ = 0) 이며, 꺼지면 예전 결과와 정확히 같다.
FILM_H = 2.0e-5          # 점액막 두께 [m]
MUCUS_GAMMA = 0.0        # 표면장력 [N/m]  (켤 때 0.03~0.05)
MUCUS_MU = 0.0           # 점도 [Pa·s]     (수화 0.01, 탈수 0.1~1)
FILM_W = 1.0e-3          # 압착막의 유효 폭 [m]
BRIDGE_RUPTURE = 2.0     # 틈이 막 두께의 이 배를 넘으면 다리가 끊긴다
BRIDGE_JITTER = 0.0      # 끊김 문턱의 주기 간 흔들림 (비율)
#: 닫힘 판정 틈 [m]. 이보다 좁으면 **유량을 0 으로 두고** 압력을 성문하압으로 채운다.
#: 1/a² 이 폭발하는 것을 수치로 막는 것이 아니라, 닫히면 흐르지 않는다는 물리를 쓴다.
H_CLOSED = 1.0e-6


def _modal_setup(strain, contact, vib_depth, n_modes, nx, ny, full=False, ta=0.0):
    """모드 진동수와 **내측면 형상** φ_n(y) 를 뽑는다.

    `full=True` 면 전체 변위장 모드 (3n, 모드) 와 절점 질량 ρ·dx·dy 도 준다 — 늘어남 교체 때 상태를 잇는 데 쓴다.
    """
    L = geometry(strain, ta)[0]
    f, vec, meta = bands([np.pi / L], strain, nx=nx, ny=ny, n_modes=n_modes,
                         shapes=True, contact=contact, vib_depth=vib_depth, ta=ta)
    f, vec, meta = f[0], vec[0], meta[0]
    n = nx * ny
    u = vec[:n]                                   # (n, n_modes)
    med = np.array([0 * ny + j for j in range(ny)])   # x = 0 (내측면) 절점
    phi = u[med]                                  # (ny, n_modes)
    # **eigsh 는 이미 M-정규화된 벡터를 준다** (vᵀMv = 1, 확인함). 여기에 ‖v‖ 로 또
    # 나누면 φ 가 1/(ρ·cell) 의 제곱근 배(여기서는 8385 배) 만큼 작아져 구동력이 사라진다.
    # 그래서 어떤 압력에서도 진동이 안 일어났다. 정규화는 건드리지 않는다.
    keep = np.isfinite(f) & (f > 1.0)
    if full:
        _L, T, D = geometry(strain, ta)
        m_cell = RHO * (D / (nx - 1)) * (T / (ny - 1))
        return f[keep], phi[:, keep], meta[keep], vec[:, keep], m_cell
    return f[keep], phi[:, keep], meta[keep]


#: **성문 난류의 압력 요동** (성문하압 대비 비율). 열린 구간의 제트는 Re > 1800 이라 난류이고, 그 압력
#: 요동이 성대를 주기마다 조금씩 다르게 민다 — 한계 순환이 완벽히 주기적이면 고역 배음이 목표의 10 배 넘게
#: 배음적이었다 (§52.6). 1 차 저역통과한 가우시안 (시상수 `P_NOISE_TAU`). 기본 0 이면 예전과 정확히 같다.
P_NOISE = 0.0
P_NOISE_TAU = 2.0e-3
#: **성대 장력(강성)의 미세 요동** — 고유진동수를 ω·(1 + F_NOISE·n) 로 흔든다. 압력 요동은 주로 **진폭**을
#: 흔들어(P_NOISE 0.20 에서 시머 15.9 %) 한계 순환의 닫힘 위상이 거의 안 흩어졌고, 그래서 지터가 목표(2.5 %)에
#: 맞아도 고역 배음성이 0.44~0.54 로 남았다 (§52.9). 고역을 흩는 것은 **위상 확산**이다 — 장력 요동은 진동수
#: 자체를 바꾸므로 주기를 흔든다. 근긴장·혈류 박동이 생리적 원천이다. 기본 0 이면 예전과 정확히 같다.
F_NOISE = 0.0
F_NOISE_TAU = 3.0e-3
#: **성문 제트 난류의 압력 요동** — 물리 원천으로 짠 판 (§52.28). 분리 뒤(제트 쪽) 압력에 p' = C·g(Re)·½ρv²·n(t) 를 더한다.
#: v 는 최소 단면의 제트 속도, n 은 차단 주파수 St·v/d_h (d_h = 2A/ℓ, 슬릿의 수력 직경) 로 저역통과한 단위 분산 잡음,
#: g = max(0, 1 − (Re_c/Re)²) 는 레이놀즈 문턱 아래에서 0 이다. C 는 난류 압력 요동의 rms 를 동압으로 나눈 값이라
#: 물리적으로 1 보다 한참 작아야 한다 (0.1~0.3 급). 기본 0 이면 난수를 쓰지 않아 예전과 비트 단위로 같다.
JET_P_NOISE = 0.0
JET_STROUHAL = 0.2
#: 요동을 만드는 소용돌이가 자라는 **제트 발달 거리** [m] (§52.44). 0 이면 성문 출구의 수력 직경 척도(차단 약 10 kHz, 거의 백색) 를 쓴다.
#: 성대 두께만큼 지나며 자란 제트는 폭 b = d_h + JET_SPREAD·x, 중심 속도 v·√(d_h/b) (평면 제트의 운동량 보존) 이고 차단은 St·v_c/b 다.
JET_X = 0.0
JET_SPREAD = 0.2
#: 요동 스펙트럼의 저역통과 **차수** (§52.45). 1 은 차단 위 −6 dB/oct 라 성도의 미분을 거치면 고역이 평평하게 남는다. 관성 영역의 난류
#: 압력 스펙트럼은 대략 f^(−7/3) 로 떨어지므로 2 가 물리에 가깝다. 같은 차단의 1 차를 직렬로 쌓고, 단위 분산이 되게 되돌린다.
JET_ORDER = 1
#: **성문 위 공간의 순응도** [m³/Pa] (§52.49). 0 이면 성문 위 압력을 준정상 두 오리피스로 푼다 (예전과 같다). 양수면 압력이 **상태**가 된다:
#: C·dp/dt = U − A_out·√(2p/ρ). 폐쇄음의 폐쇄 구간에서 압력이 수십 ms 에 걸쳐 쌓여 성문 사이 압력차를 줄이고, 그동안 발성이 조금 이어진다
#: (유성 폐쇄음의 voice bar). 공기 부피 약 70 cm³ 의 압축성은 5e-10 이고, 벽이 늘어나는 몫까지 합치면 1e-9~1e-8 급이다.
TRACT_COMPLIANCE = 0.0


class Glottis:
    """자려 진동하는 성문 — 모드 설정, 한 걸음 적분, 늘어남이 바뀔 때의 **연속 교체**.

    `simulate` 와 렌더러가 **같은 적분 코드**를 쓰게 하려고 만들었다. 예전에는 루프가 두 벌로 복제돼
    렌더러에 점막액 항이 빠지는 식으로 이미 어긋나기 시작했다.
    """

    def __init__(self, strain=0.15, h0=0.2e-3, contact=0.0, vib_depth=1.0, n_modes=4,
                 nx=14, ny=NZ, seed=0, l_supra=0.17, a_supra=3.0e-4, ta=0.0):
        self.ny, self.nx, self.n_modes = ny, nx, n_modes
        self.ta = float(ta)
        self.contact, self.vib_depth = contact, vib_depth
        self.h0 = float(h0)
        # 성도 공기 기둥의 관성 (ρℓ/A) — 발성 문턱을 낮추고 펄스를 닫힘 쪽으로 기울인다 (§52.5).
        self.L_tract = RHO_AIR * float(l_supra) / float(a_supra)
        self.rng = np.random.default_rng(seed)
        self.U = 0.0                               # 유량은 **상태**다 (공기 기둥의 관성)
        self.bridge = np.zeros(ny, dtype=bool)
        self.bridge_prev = False
        self.rupture_h = BRIDGE_RUPTURE * FILM_H
        self.p_noise = 0.0
        self.f_noise = 0.0
        self.jet_noise = 0.0
        self.jet_stage = [0.0, 0.0, 0.0]
        self.p_jet = 0.0
        # **성문 위 출구 면적** [m²] — 구강 협착(과 비강 포트) 의 유효 단면. None 이면 성문 위 압력 0 (예전과 같다).
        # 두 오리피스 직렬의 준정상 베르누이: p_sup = ½ρU²/A_out². 무성 자음에서 발성이 멎는 데 필요한 물리다 (§52.30).
        self.a_out = None
        self.p_sup = 0.0
        self.q = self.qd = self.phi = self.vec = None
        self.ok = self._set_modes(strain, first=True)

    def _set_modes(self, strain, first=False):
        f_n, phi, _meta, vec, m_cell = _modal_setup(strain, self.contact, self.vib_depth, self.n_modes,
                                                    self.nx, self.ny, full=True, ta=self.ta)
        if not len(f_n):
            return False
        nq = len(f_n)
        if first or self.phi is None:
            q = np.zeros(nq); qd = np.zeros(nq)
            q += self.rng.normal(0.0, 1e-9, nq)    # 미세한 씨앗
        else:
            # 늘어남이 바뀌면 모드 집합이 달라진다. **변위장 전체를 질량 가중으로** 새 모드에 사영한다:
            # q_new = Φ_newᵀ M (Φ_old q_old). 모드가 M-정규이므로 에너지를 늘리지 않는다.
            # 예전에는 내측면 절점의 변위만 pinv 로 이었다. 모드 6 개의 내측면 형상은 rank 5 라 (내측면을 움직이지
            # 않는 모드가 있다) pinv 가 내부 변형을 버리고 |q| 를 0.3~1.0 배로 바꿨고, 늘어남을 ±0.005 만
            # 오가도 닫힘 시각 지터가 0.02 → 2.44 % 로 뛰었다 (§52.18).
            q = vec.T @ (m_cell * (self.vec @ self.q))
            qd = vec.T @ (m_cell * (self.vec @ self.qd))
        self.f_n, self.phi, self.nq = f_n, phi, nq
        self.vec, self.m_cell = vec, m_cell
        self.w = 2 * np.pi * f_n
        L, T, _D = geometry(strain, self.ta)
        self.ell, self.dz = L, T / (self.ny - 1)
        yy = np.linspace(0.0, 1.0, self.ny)
        self.h_rest = self.h0 * (1.0 + 0.6 * (1.0 - yy))   # 아래가 조금 넓은 수렴형
        self.q, self.qd = q, qd
        return True

    def _tract_pressure_step(self, U, dt):
        """C·(p_new − p_old)/dt = U − k·√p_new,  k = A_out·√(2/ρ) — √p 에 대한 이차식을 풀어 암시적으로 옮긴다."""
        a_c = TRACT_COMPLIANCE / dt
        k = float(self.a_out) * np.sqrt(2.0 / RHO_AIR)
        b = a_c * max(self.p_sup, 0.0) + U
        if b <= 0.0:
            self.p_sup = 0.0
            return
        q = (-k + np.sqrt(k * k + 4.0 * a_c * b)) / (2.0 * a_c)
        self.p_sup = q * q

    def reshape(self, strain, ta=None, vib_depth=None):
        """늘어남(= F0)·TA 수축·진동 깊이(성구) 를 바꾼다. 상태는 변위장 전체를 질량 가중으로 사영해 이어받는다 (§52.18).

        진동 깊이가 줄면 깊은 쪽 절점이 고정되는데, 고정 자유도는 모드 벡터에서 0 이므로 같은 사영이 그대로 쓰인다 (§52.32).
        """
        if ta is not None:
            self.ta = float(ta)
        if vib_depth is not None:
            self.vib_depth = float(vib_depth)
        return self._set_modes(strain)

    def set_h0(self, h0):
        """내전(휴지 반틈새) 을 바꾼다. 모드는 그대로 두고 휴지 형상만 옮긴다 — 무성 자음의 외전·유성 시작의 내전 (§52.25)."""
        self.h0 = float(h0)
        yy = np.linspace(0.0, 1.0, self.ny)
        self.h_rest = self.h0 * (1.0 + 0.6 * (1.0 - yy))

    def step(self, P, dt):
        """한 걸음. 반환 (U [m³/s], 최소 반틈새 [m])."""
        ny, phi, w, dz, ell = self.ny, self.phi, self.w, self.dz, self.ell
        q, qd, U = self.q, self.qd, self.U
        if P_NOISE > 0.0:
            a_n = dt / (P_NOISE_TAU + dt)
            self.p_noise += a_n * (self.rng.normal() - self.p_noise)
            # 1 차 저역통과는 분산을 줄이므로 표준편차가 1 이 되도록 되돌린다.
            P = P * (1.0 + P_NOISE * self.p_noise * np.sqrt((2.0 - a_n) / a_n))
        if F_NOISE > 0.0:
            a_f = dt / (F_NOISE_TAU + dt)
            self.f_noise += a_f * (self.rng.normal() - self.f_noise)
            w = w * (1.0 + F_NOISE * self.f_noise * np.sqrt((2.0 - a_f) / a_f))
        h = self.h_rest + phi @ q
        a = duct_area(h, ell)
        # **비정상 베르누이** (공기 기둥의 관성). 준정상으로 풀면 한 주기 알짜 일이 0 이다.
        if h.min() <= 0.0:
            # 닫혔다 — 면적이 없으면 흐름이 없다. 성문 아래 전체가 성문하압을 받는다.
            U = 0.0
            i_sep = int(np.argmin(h))
            if TRACT_COMPLIANCE > 0.0 and self.a_out is not None:
                self._tract_pressure_step(0.0, dt)
                p = np.where(np.arange(ny) <= i_sep, P, self.p_sup).astype(float)
            else:
                self.p_sup = 0.0
                p = np.where(np.arange(ny) <= i_sep, P, 0.0).astype(float)
        else:
            _U_qs, _p0, i_sep = flow(a, dz, ell, P, 0.0)
            a_eff = max(a[:i_sep + 1].min(), 1e-12)
            a_sep = a[i_sep]
            d_len = dz * (i_sep + 1)
            # 후진 오일러 + 성도 관성 (§52.1 버그 9). a → 0 이면 R_v → ∞ 라 U → 0 으로 저절로 닫힌다.
            Li = RHO_AIR * d_len / a_eff + self.L_tract
            R_k = 0.5 * RHO_AIR * (1.0 / a_sep ** 2 - 1.0 / A_SUB ** 2)
            R_v = 12.0 * MU_AIR * d_len * ell ** 2 / a_eff ** 3
            bq = R_v + Li / dt
            p_j = self.p_jet if JET_P_NOISE > 0.0 else 0.0   # 직전 걸음의 제트 압력 요동 (분리 뒤에 걸린다)
            cq = max(P - p_j + Li * U / dt, 0.0) if JET_P_NOISE > 0.0 else P + Li * U / dt
            # 출구 협착의 운동 에너지 손실 ½ρ/A_out² 는 성문의 R_k 와 **직렬로 더해진다** (같은 U 가 두 오리피스를 지난다).
            if TRACT_COMPLIANCE > 0.0 and self.a_out is not None:
                # 성문 위 압력은 상태 — 지난 걸음의 압력으로 유량을 풀고, 그 유량으로 압력을 암시적으로 옮긴다.
                cq = max(cq - self.p_sup, 0.0)
                U = (2.0 * cq) / (bq + np.sqrt(bq * bq + 4.0 * max(R_k, 0.0) * cq))
                self._tract_pressure_step(U, dt)
                R_c = 1.0                                    # 아래에서 벽 압력에 성문 위 압력을 싣게 한다
            else:
                R_c = 0.0 if self.a_out is None else 0.5 * RHO_AIR / self.a_out ** 2
                U = (2.0 * cq) / (bq + np.sqrt(bq * bq + 4.0 * (max(R_k, 0.0) + R_c) * cq))
                self.p_sup = R_c * U * U
            if JET_P_NOISE > 0.0 or R_c > 0.0:
                p = _wall_pressure(a, i_sep, U, P, p_j + self.p_sup)
            else:
                p = _wall_pressure(a, i_sep, U, P)
            if JET_P_NOISE > 0.0:
                v = U / a_eff
                d_h = 2.0 * a_eff / ell
                re = RHO_AIR * v * d_h / MU_AIR
                gate = max(0.0, 1.0 - (RE_CRIT / re) ** 2) if re > 0.0 else 0.0
                if JET_X > 0.0:
                    b_j = d_h + JET_SPREAD * JET_X
                    fc = min(JET_STROUHAL * v * np.sqrt(d_h / b_j) / b_j, 0.4 / dt)
                else:
                    fc = min(JET_STROUHAL * v / max(d_h, 1e-9), 0.4 / dt)
                a_j = 1.0 - np.exp(-2.0 * np.pi * max(fc, 1.0) * dt)
                if JET_ORDER <= 1:
                    self.jet_noise += a_j * (self.rng.normal() * np.sqrt((2.0 - a_j) / a_j) - self.jet_noise)
                else:
                    # 같은 계수의 1 차 k 개 직렬. 분산 되돌림은 직렬 이득의 제곱합(이산 1 차의 거듭제곱 합) 으로 근사한다.
                    b_j = 1.0 - a_j
                    k_ord = int(min(JET_ORDER, 3))
                    if k_ord == 2:
                        g2 = a_j ** 4 * (1.0 + b_j * b_j) / (1.0 - b_j * b_j) ** 3
                    else:
                        g2 = a_j ** 6 * (1.0 + 4.0 * b_j * b_j + b_j ** 4) / (1.0 - b_j * b_j) ** 5
                    x_in = self.rng.normal() / np.sqrt(g2)
                    for kk in range(k_ord):
                        self.jet_stage[kk] += a_j * (x_in - self.jet_stage[kk])
                        x_in = self.jet_stage[kk]
                    self.jet_noise = x_in
                self.p_jet = JET_P_NOISE * gate * 0.5 * RHO_AIR * v * v * self.jet_noise
        hd = phi @ qd
        pen = np.where(h < 0.0, -K_CONTACT * h - C_CONTACT * np.minimum(hd, 0.0), 0.0)
        c_sq = None
        if MUCUS_GAMMA > 0.0 or MUCUS_MU > 0.0:
            # 점막액 (§52.7). 액체 틈 = max(h, 0) + 막 두께. 압착막은 아래에서 암시적으로.
            h_eff = np.maximum(h, 0.0) + FILM_H
            wet = np.maximum(h, 0.0) < FILM_H
            self.bridge = np.where(wet, True, self.bridge & (h < self.rupture_h))
            pen = pen + np.where(self.bridge, -2.0 * MUCUS_GAMMA / h_eff, 0.0)
            c_sq = np.where(wet, MUCUS_MU * (FILM_W ** 2) / h_eff ** 3, 0.0)
            if (not self.bridge.any()) and self.bridge_prev:
                self.rupture_h = BRIDGE_RUPTURE * FILM_H * max(
                    0.2, 1.0 + BRIDGE_JITTER * self.rng.normal())
            self.bridge_prev = bool(self.bridge.any())
        # 질량행렬이 단위 z 길이당이므로 힘도 단위 길이당이다 (ℓ 을 곱하지 않는다).
        fq = phi.T @ ((p + pen) * dz)
        if c_sq is None or not np.any(c_sq):
            qd = qd + (fq - 2 * ZETA * w * qd - (w ** 2) * q) * dt      # 반암시적 오일러
        else:
            C_sq = phi.T @ (phi * (c_sq * dz)[:, None])
            M_im = np.eye(self.nq) + dt * np.diag(2 * ZETA * w) + dt * C_sq
            qd = np.linalg.solve(M_im, qd + dt * (fq - (w ** 2) * q))
        q = q + qd * dt
        self.q, self.qd, self.U = q, qd, U
        return U, float(h.min())


def simulate(p_sub_cm=6.0, strain=0.15, contact=0.0, vib_depth=1.0,
             h0=0.2e-3, seconds=0.08, fs=200000.0, n_modes=4,
             nx=14, ny=NZ, seed=0, l_supra=0.17, a_supra=3.0e-4, ta=0.0):
    """자려 진동을 돌린다. 반환 dict (t, U, area, f0, oq, modes).

    접촉 몫 기본값은 0 이다 (§52.4) — 닫힘은 경계가 아니라 주기마다의 접촉 벌칙이 담당한다.
    """
    g = Glottis(strain=strain, h0=h0, contact=contact, vib_depth=vib_depth, n_modes=n_modes,
                nx=nx, ny=ny, seed=seed, l_supra=l_supra, a_supra=a_supra, ta=ta)
    if not g.ok:
        return {"ok": False}
    dt = 1.0 / fs
    nt = int(seconds * fs)
    P = p_sub_cm * CM_H2O
    out_U = np.empty(nt); out_a = np.empty(nt)
    for it in range(nt):
        out_U[it], out_a[it] = g.step(P, dt)
        if not np.isfinite(g.q).all():
            return {"ok": False, "blown": it}
    t = np.arange(nt) / fs
    half = out_U[nt // 2:]
    x = half - half.mean()
    if x.std() < 1e-12:
        return {"ok": True, "t": t, "U": out_U, "area": out_a, "f0": 0.0, "oq": 0.0,
                "modes": g.f_n}
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo = int(fs / 800); hi = int(fs / 80)
    lag = lo + int(np.argmax(ac[lo:hi])) if hi > lo else 0
    f0 = fs / lag if lag else 0.0
    oq = float(np.mean(out_a[nt // 2:] > 0.0))
    return {"ok": True, "t": t, "U": out_U, "area": out_a, "f0": f0, "oq": oq,
            "modes": g.f_n}


def _check() -> int:
    print("자려 진동 검증")
    print("  P[cmH2O]   진동  F0[Hz]  개방률   유량 정점[cm³/s]")
    res = []
    for pc in (1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0):
        r = simulate(p_sub_cm=pc, seconds=0.06)
        if not r.get("ok"):
            print(f"  {pc:6.1f}     발산"); continue
        amp = np.ptp(r["U"][len(r["U"]) // 2:])
        osc = amp > 1e-7
        res.append((pc, r["f0"] if osc else 0.0, r["oq"], amp))
        print(f"  {pc:6.1f}   {'O' if osc else 'X':^5}  {r['f0'] if osc else 0:6.1f}  "
              f"{r['oq']:6.2f}   {amp*1e6:10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_check())
