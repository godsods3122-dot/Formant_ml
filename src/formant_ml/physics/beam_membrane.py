"""**보–막 성대** (Serry, Zañartu & Peterson 2026, arXiv 2606.13480) — 모드(밴드) 기저로 (§52.504).

사용자: *"성대 근육 규칙 … 후속 연구 등등으로부터 얻어낼 순 없어?"* 와 *"성대 밴드구조, 위상차 응답, 임피던스, 리액턴스 … 절대 갖다버리면 안
되는 부분"*. Alzamendi 2021 (삼각 몸체–덮개 + 근육 자세) 의 후속인 이 모형은 경험 규칙(Titze & Story 2002) 없이 **조직 응력–변형에서 성대
역학을 물리로** 세운다. 원본 Matlab 코드는 `third_party/Membrane-beam-VF-model/` (로컬).

구조 (한쪽 성대, 좌우 대칭)
-------------------------------
* **점막 = 막** w_m(x, y): 세로 장력 T̃ = λ^{3/2}·σ̄_muc(ε)·λ·d_muc (x 방향), 상하(y) 방향은 전단 G̃(∂y w − φ) 와 회전 φ 의 관성 J̃ —
  두께 방향의 전단 변형이 **상하로 퍼지는 점막파**(아래·위 날의 위상차)를 만든다. 점성 전단 η̃.
* **인대 + 갑상피열근 = 복합 보** w_b(x): 축력 Ñ_b = λ^{3/2}·Σλ A_i σ̄_i, 굽힘 강성 μ̃ = λ^{3/2}·λ²·Σ E_i(I_i + (r_c − r_i)A_i α_i),
  층마다 응력이 달라 생기는 **굽힘 모멘트** M̄ = Σ(r_c − r_i)·λ A_i σ̄_i — 갑상피열근 활성이 성대를 안쪽으로 불룩하게 한다 (끝 조건으로 들어간다).
* 막–보 결합 K_c(y)(w_m − w_b) + C_c(ẇ_m − ẇ_b) — 아래(입구)·위의 결합 강성이 10 배 다르다 (`COUPLING_STIFF_BELOW`).
* 보–연골 기초 K_f, C_f. 끝: 막·보 변위 0, 보 끝의 회전 용수철 K_r,a·K_r,p (앞쪽은 쉼 각 θ_0 로 되돌린다).

이산화 (원본과 다른 점)
-----------------------
원본은 명시적 중앙차분(1.79 µs)이다. 여기서는 같은 식을 **에너지 형식**으로 이산화해 대칭 (K, M, C) 를 만들고 고유모드로 푼다 — 이것이 사용자의
**막 밴드 구조**다 (x 방향 파수 × 상하 모양의 가지). φ 는 y 칸 가운데(엇갈림 격자)에 둔다. 관 풀이기(96 kHz)에서는 굳은 모드(보 굽힘의 높은
파수)가 명시적 적분을 깨므로, 모드 좌표에서 각 모드를 정확히 적분하고 비선형 힘(성문 압력·접촉)만 매 표본 투영한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

from . import fold_rules as FR

# ---- Serry et al. 2026 표 1·2·4 (남성) ---------------------------------------------------------------------------------
SERRY_MALE = dict(L0=1.5e-2, b0=5.0e-3, A_muc=5.0e-6, A_lig=6.1e-6, A_ta=40.9e-6,
                  rho_muc=1000.0, rho_lig=1030.0, rho_ta=1050.0)
#: 접촉 감쇠 `C_col` (§52.524): Serry 값 1e2 N·s/m³ 는 줄의 접촉 강성·질량에 대해 감쇠비 0.075 라 닫힐 때 줄이 튕기며 울려 음원 4–8 kHz 에 봉우리를 냈다
#: (점막판 m3 유량 미분 4–8 kHz 가 2–4 kHz 보다 높았다). 무른 조직끼리 닿으면 에너지가 흩어진다 — Ishizaka & Flanagan (1972) 은 접촉 중 감쇠비 0.1 → 0.6,
#: VTL 은 닿은 몫만큼 임계 감쇠를 더한다. ε 0.2 의 줄 질량·접촉 강성에서 감쇠비 0.6 이 되는 8e2 로 (음원 4–8 kHz −8 dB, 점성 ×4 는 효과 없음).
SERRY_STRUCT = dict(K_c=5.0e4, C_c=10.0, K_f=1.0e2, C_f=1.0, K_col=1.0e6, C_col=8.0e2, G_m=1.0e2, eta_m=1.0,
                    K_ra=2.0e-2, K_rp=1.0e-2, theta_conv=0.02, sep_ratio=1.3)
#: 여성 척도 (Titze 1989): 길이 ÷1.6, 두께·깊이 ÷1.2 (단면적 ÷1.44).
FEMALE_LEN = 1.0 / 1.6
FEMALE_THICK = 1.0 / 1.2
#: 원본 코드의 결합 강성 분포 — 입구(아래) 쪽이 위쪽의 10 배. 논문 본문(식 25 아래)은 반대로 "위가 굳다" 고 쓴다. 방향이 아래·위 날의
#: 위상차(아래 날이 먼저 닫히는 것)를 정하므로 둘 다 재서 고른다 (§52.504).
COUPLING_STIFF_BELOW = True
COUPLING_RATIO = 10.0
#: 상하 전단의 꼴. "timoshenko" 는 논문 식 25–26 그대로 (독립 회전 φ, 관성 J̃ = ρd³/12). J̃ 가 아주 작아 φ 가 기울기를 곧바로 따라가므로
#: 전단력이 거의 사라지고 점막 줄들이 따로 논다 — 아래 날이 ±3 mm 흔들렸다 (원본 코드는 가장자리 줄을 이웃에 운동학적으로 묶어 이것을 가렸다).
#: "direct" 는 φ 없이 G̃(∂y w)² — 덮개 전단이 아래·위 줄을 곧바로 잇는다 (Titze–Story 덮개 결합 kc 와 같은 물리).
SHEAR = "timoshenko"


@dataclass
class Geometry:
    """늘어난 상태의 치수와 계수 (SI, 한쪽 성대)."""
    eps: float
    a_ta: float
    lam: float
    L: float
    b: float
    d_muc: float
    rho_s: float            # 막 면밀도 ρ_muc·d_muc [kg/m²]
    J_s: float              # 막 회전 관성 (면적당) ρ d³/12 [kg]
    T_m: float              # 막 장력 [N/m]
    G_s: float              # 막 상하 전단 (면적당 계수) d·G/λ [N/m]
    eta_s: float            # 막 상하 점성 d·η/λ [N·s/m]
    rho_l: float            # 보 선밀도 [kg/m]
    N_b: float              # 보 축력 [N]
    mu_b: float             # 보 굽힘 강성 [N·m²]
    M0: float               # 층 응력 차의 굽힘 모멘트 [N·m]
    sig: dict = field(default_factory=dict)


def geometry(eps: float, a_ta: float, sex: str = "female", tissue: str = "serry2026", scale_len: float = 1.0,
             scale_thick: float = 1.0) -> Geometry:
    """근육 상태 (늘어남 ε, 갑상피열근 활성 a_TA) → 늘어난 치수·계수. `Membrane_Beam_Parameters_Final.m` 을 옮긴 것."""
    p = dict(SERRY_MALE)
    ls = (FEMALE_LEN if sex == "female" else 1.0) * scale_len
    ts = (FEMALE_THICK if sex == "female" else 1.0) * scale_thick
    L0, b0 = p["L0"] * ls, p["b0"] * ts
    A0 = {k: p[f"A_{k}"] * ts * ts for k in ("muc", "lig", "ta")}
    lam = 1.0 + float(eps)
    l2 = 1.0 / math.sqrt(lam)
    d = {k: A0[k] / b0 * l2 for k in A0}
    A = {k: A0[k] / lam for k in A0}
    b = b0 * l2
    ts_ = FR.TISSUE_SETS[tissue]
    s_muc = float(FR.passive_stress(eps, ts_["muc"]))
    s_lig = float(FR.passive_stress(eps, ts_["lig"]))
    s_ta = float(FR.passive_stress(eps, ts_["ta"]) + FR.active_stress(eps, a_ta, ts_["ta"]))
    E_lig = max(float(FR.passive_modulus(eps, ts_["lig"])), 1.0)
    E_ta = max(float(FR.passive_modulus(eps, ts_["ta"]) + FR.active_modulus(eps, a_ta, ts_["ta"])), 1.0)
    rho_s = p["rho_muc"] * d["muc"]
    J_m = lam * p["rho_muc"] * d["muc"] ** 2 / 12.0
    rho_l = p["rho_lig"] * A["lig"] + p["rho_ta"] * A["ta"]
    I_lig = b * d["lig"] ** 3 / 12.0
    I_ta = b * d["ta"] ** 3 / 12.0
    rc = d["ta"] + 0.5 * d["lig"]
    r_lig, r_ta = rc, 0.5 * d["ta"]
    a_lig = -(d["ta"] + d["lig"]) / (2.0 * (E_lig * A["lig"]) / (E_ta * A["ta"]) + 1.0)
    a_ta_ = (d["ta"] + d["lig"]) / (2.0 * (E_ta * A["ta"]) / (E_lig * A["lig"]) + 1.0)
    F_lig, F_ta = lam * A["lig"] * s_lig, lam * A["ta"] * s_ta
    T_m = lam ** 1.5 * (s_muc * lam) * d["muc"]
    M0 = (rc - r_lig) * F_lig + (rc - r_ta) * F_ta
    th = E_lig * I_lig + E_ta * I_ta + (rc - r_lig) * A["lig"] * E_lig * a_lig + (rc - r_ta) * A["ta"] * E_ta * a_ta_
    mu = lam ** 1.5 * lam ** 2 * th
    N_b = lam ** 1.5 * (F_lig + F_ta)
    st = SERRY_STRUCT
    return Geometry(eps, a_ta, lam, L0 * lam, b, d["muc"], rho_s, J_m * d["muc"] / lam * lam / lam,
                    T_m, d["muc"] * st["G_m"] / lam, d["muc"] * st["eta_m"] / lam, rho_l, N_b, mu, M0,
                    dict(muc=s_muc, lig=s_lig, ta=s_ta, E_lig=E_lig, E_ta=E_ta))


@dataclass
class System:
    """이산 선형계 M ẍ + C ẋ + K x = f_s + B·p (자유도: 막 w (내부 x × Ny), 막 φ (내부 x × (Ny−1)), 보 w_b (내부 x))."""
    g: Geometry
    nx: int
    ny: int
    K: sp.csr_matrix
    M: np.ndarray           # 대각 질량 (벡터)
    C: sp.csr_matrix
    f_s: np.ndarray         # 정적 하중 (굽힘 모멘트·쉼 각)
    x: np.ndarray           # 막 절점 x (내부)
    y: np.ndarray           # 막 절점 y (0 = 아래·입구, b = 위)
    wy: np.ndarray          # 막 절점의 y 칸 폭 (양 끝 반 칸)
    dx: float
    n_w: int
    n_p: int
    n_b: int

    def w_index(self, i: int, j: int) -> int:
        return i * self.ny + j


def assemble(g: Geometry, nx: int = 12, ny: int = 8, theta_g: float = 0.0, theta_0: float = 0.0,
             coupling_below: bool | None = None, shear: str | None = None) -> System:
    """에너지 형식의 이산화 — 대칭 K, 대각 M, 대칭 C, 정적 하중.

    막 절점 i = 1..nx−2 (x = 0, L 은 고정), j = 0..ny−1 (y = j·dy, 0 이 아래 = 입구). φ 는 y 칸 가운데 (i, j+½).
    V = ½Σ T̃ (∂x w)² dA + ½Σ G̃ (∂y w − φ)² dA + ½Σ K_c(y)(w − w_b)² dA + ½Σ Ñ_b (w_b')² dx + ½Σ μ̃ (w_b'')² dx + ½Σ K_f w_b² dx
        + [끝] −(M̄ + K_ra(θ_g − θ_0))·w_b'(0) + ½K_ra λ^{3/2} w_b'(0)² + M̄·w_b'(L) + ½K_rp λ^{3/2} w_b'(L)²  (원본의 끝 모멘트 식과 같은 자연 조건)
    T = ½Σ ρ_s ẇ² dA + ½Σ J̃ φ̇² dA + ½Σ ρ_l ẇ_b² dx.
    """
    st = SERRY_STRUCT
    cb = COUPLING_STIFF_BELOW if coupling_below is None else coupling_below
    direct = (SHEAR if shear is None else shear) == "direct"
    ni = nx - 2
    dx = g.L / (nx - 1)
    dy = g.b / (ny - 1)
    xs = (np.arange(1, nx - 1)) * dx
    ys = np.arange(ny) * dy
    wy = np.full(ny, dy)
    wy[0] = wy[-1] = 0.5 * dy
    n_w, n_p, n_b = ni * ny, (0 if direct else ni * (ny - 1)), ni
    N = n_w + n_p + n_b
    W = lambda i, j: i * ny + j                       # noqa: E731  (i: 내부 번호 0..ni−1)
    P = lambda i, j: n_w + i * (ny - 1) + j           # noqa: E731
    B = lambda i: n_w + n_p + i                       # noqa: E731
    rows, cols, vals = [], [], []
    crow, ccol, cval = [], [], []

    def add(r, c, v, R=rows, Cc=cols, V=vals):
        R.append(r); Cc.append(c); V.append(v)

    def add_quad(idx, coef, w, target="K"):
        """½·w·(Σ coef_k x_idx_k)² 의 헤세 (K 또는 C 에)."""
        R, Cc, V = (rows, cols, vals) if target == "K" else (crow, ccol, cval)
        for a, ca in zip(idx, coef):
            for b_, cb_ in zip(idx, coef):
                R.append(a); Cc.append(b_); V.append(w * ca * cb_)

    # 막 x 장력: 이웃 (i, i+1), 끝은 고정 0
    for j in range(ny):
        for i in range(-1, ni):
            idx, coef = [], []
            if i >= 0:
                idx.append(W(i, j)); coef.append(-1.0 / dx)
            if i + 1 < ni:
                idx.append(W(i + 1, j)); coef.append(1.0 / dx)
            add_quad(idx, coef, g.T_m * dx * wy[j])
    # 막 y 전단 (∂y w − φ) 와 점성
    for i in range(ni):
        for j in range(ny - 1):
            idx = [W(i, j), W(i, j + 1)] + ([] if direct else [P(i, j)])
            coef = [-1.0 / dy, 1.0 / dy] + ([] if direct else [-1.0])
            add_quad(idx, coef, g.G_s * dx * dy)
            add_quad(idx, coef, g.eta_s * dx * dy, "C")
    # 막–보 결합 (y 에 따라 선형으로 COUPLING_RATIO 배)
    r = COUPLING_RATIO
    for j in range(ny):
        f = (r + (1.0 - r) * j / (ny - 1)) if cb else (1.0 + (r - 1.0) * j / (ny - 1))
        kc = st["K_c"] * f
        for i in range(ni):
            idx, coef = [W(i, j), B(i)], [1.0, -1.0]
            add_quad(idx, coef, kc * dx * wy[j])
            add_quad(idx, coef, st["C_c"] * dx * wy[j], "C")
    # 보: 축력·굽힘·기초
    for i in range(-1, ni):
        idx, coef = [], []
        if i >= 0:
            idx.append(B(i)); coef.append(-1.0 / dx)
        if i + 1 < ni:
            idx.append(B(i + 1)); coef.append(1.0 / dx)
        add_quad(idx, coef, g.N_b * dx)
    for i in range(ni):
        idx, coef = [B(i)], [-2.0 / dx ** 2]
        if i - 1 >= 0:
            idx.append(B(i - 1)); coef.append(1.0 / dx ** 2)
        if i + 1 < ni:
            idx.append(B(i + 1)); coef.append(1.0 / dx ** 2)
        add_quad(idx, coef, g.mu_b * dx)
        add_quad([B(i)], [1.0], st["K_f"] * dx)
        add_quad([B(i)], [1.0], st["C_f"] * dx, "C")
    # 끝 회전 용수철 (w_b'(0) ≈ w_b[0]/dx, w_b'(L) ≈ −w_b[ni−1]/dx)
    c15 = g.lam ** 1.5
    add_quad([B(0)], [1.0 / dx], st["K_ra"] * c15)
    add_quad([B(ni - 1)], [-1.0 / dx], st["K_rp"] * c15)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    C = sp.coo_matrix((cval, (crow, ccol)), shape=(N, N)).tocsr()
    f_s = np.zeros(N)
    # 일 = (M̄ + K_ra(θ_g − θ_0))·w_b'(0) − M̄·w_b'(L)
    f_s[B(0)] += (g.M0 + st["K_ra"] * (theta_g - theta_0)) / dx
    f_s[B(ni - 1)] += g.M0 / dx
    Md = np.zeros(N)
    for i in range(ni):
        for j in range(ny):
            Md[W(i, j)] = g.rho_s * dx * wy[j]
        for j in range(ny - 1 if not direct else 0):
            Md[P(i, j)] = g.J_s * dx * dy
        Md[B(i)] = g.rho_l * dx
    return System(g, nx, ny, K, Md, C, f_s, xs, ys, wy, dx, n_w, n_p, n_b)


@dataclass
class Modes:
    """모드 기저 — 막 내측면 변위의 모양 Φ_w (막 절점, 모드), 고유진동수, 모드 감쇠비, 정적 처짐 (막 절점)."""
    sys: System
    f: np.ndarray
    zeta: np.ndarray
    phi_w: np.ndarray       # (n_w, n_modes) — M-정규 모드의 막 w 성분
    phi_b: np.ndarray       # (n_b, n_modes)
    w_static: np.ndarray    # (n_w,) — 정적 하중(굽힘 모멘트)의 처짐
    wb_static: np.ndarray


def modes(sys: System, n_modes: int = 24, f_max: float | None = None) -> Modes:
    """일반화 고유 문제 K φ = ω² M φ (조밀 — 자유도 수백). 모드 감쇠는 φᵀCφ / 2ω."""
    Kd = sys.K.toarray()
    Md = sys.M
    s = 1.0 / np.sqrt(Md)
    A = (Kd * s[:, None]) * s[None, :]
    w2, V = sla.eigh(A)
    V = V * s[:, None]                                  # M-정규
    om = np.sqrt(np.maximum(w2, 0.0))
    f = om / (2 * np.pi)
    k = n_modes if f_max is None else int(min(n_modes, np.searchsorted(f, f_max)))
    V, om, f = V[:, :k], om[:k], f[:k]
    Cm = V.T @ (sys.C @ V)
    zeta = np.diag(Cm) / (2.0 * np.maximum(om, 1e-9))
    xs = np.linalg.solve(Kd, sys.f_s)
    return Modes(sys, f, zeta, V[:sys.n_w], V[sys.n_w + sys.n_p:], xs[:sys.n_w], xs[sys.n_w + sys.n_p:])


# ---- 단독 모의 (검증용): 준정상 흐름 (Serry 식 40–42) + 접촉, 성문 위 압력 0 ----------------------------------------------
from numba import njit  # noqa: E402

RHO_AIR = 1.14
MU_AIR = 1.8e-5


@njit(cache=True)
def _aero(w, G, nx_int, ny, dx, dy, Ps, sr, rho, mu, p_out, dmin_out):
    """막 변위 w (nx_int·ny) 와 중앙면 거리 G → 압력 p (nx_int·ny) [Pa], 유량 [m³/s], 최소 면적 [m²]. 단면(x 칸)마다 준정상.
    입구는 j = 0 (아래). 틈 d = 2·max(G − w, 0). 분리는 입구에서부터 틈이 sr·d_min 을 처음 지나는 곳 (원본과 같은 탐색)."""
    Q = 0.0
    Amin = 0.0
    for i in range(nx_int):
        dmin = 1e30
        jmin = 0
        dmax = 0.0
        for j in range(ny):
            d = 2.0 * max(G[i * ny + j] - w[i * ny + j], 0.0)
            if d < dmin:
                dmin = d
                jmin = j
            if d > dmax:
                dmax = d
        dmin_out[i] = dmin
        Amin += dmin * dx
        for j in range(ny):
            p_out[i * ny + j] = 0.0
        if dmin <= 0.0:
            # 닫힘: 입구 쪽(닫힌 자리까지)은 폐압
            for j in range(jmin + 1):
                p_out[i * ny + j] = Ps
            continue
        dsep = min(sr * dmin, dmax)
        jsep = ny - 1
        for j in range(1, ny):
            s1 = 2.0 * max(G[i * ny + j] - w[i * ny + j], 0.0) - dsep
            s0 = 2.0 * max(G[i * ny + j - 1] - w[i * ny + j - 1], 0.0) - dsep
            if s1 * s0 <= 0.0:
                jsep = j - 1
                break
        # 점성 누적 ∫ 12μ/(d³ dx) dy
        vis = 0.0
        vc = np.zeros(ny)
        for j in range(ny):
            d = 2.0 * max(G[i * ny + j] - w[i * ny + j], 0.0)
            if d > 0.0:
                vis += 12.0 * mu / (dx * d * d * d) * dy
            vc[j] = vis
        Aq = 0.5 * rho / (dsep * dsep * dx * dx)
        Bq = vc[jsep]
        dq = (-Bq + math.sqrt(Bq * Bq + 4.0 * Aq * Ps)) / (2.0 * Aq)
        Q += dq
        for j in range(jsep + 1):
            d = 2.0 * max(G[i * ny + j] - w[i * ny + j], 0.0)
            if d > 0.0:
                v = dq / (d * dx)
                p_out[i * ny + j] = Ps - 0.5 * rho * v * v - dq * vc[j]
            else:
                p_out[i * ny + j] = Ps
    return Q, Amin


@njit(cache=True)
def _run_modal(om, ze, PW, ws, G, area_w, Ps, nt, dt, nx_int, ny, dx, dy, sr, Kcol, Ccol, q0):
    """모드 적분 (각 모드 Newmark 평균 가속 — 무조건 안정), 비선형 힘은 앞 표본 상태로. 반환 (유량, 면적, 아래·위 날 평균 변위)."""
    nm = om.shape[0]
    nw = PW.shape[0]
    q = q0.copy()
    v = np.zeros(nm)
    a = np.zeros(nm)
    w = np.zeros(nw)
    wd = np.zeros(nw)
    p = np.zeros(nw)
    dmin = np.zeros(nx_int)
    outQ = np.zeros(nt)
    outA = np.zeros(nt)
    outE = np.zeros((nt, 2))
    for n in range(nt):
        for k in range(nw):
            s = ws[k]
            sd = 0.0
            for m in range(nm):
                s += PW[k, m] * q[m]
                sd += PW[k, m] * v[m]
            w[k] = s
            wd[k] = sd
        Q, Am = _aero(w, G, nx_int, ny, dx, dy, Ps, sr, RHO_AIR, MU_AIR, p, dmin)
        outQ[n] = Q
        outA[n] = Am
        lo = 0.0
        hi = 0.0
        for i in range(nx_int):
            lo += w[i * ny]
            hi += w[i * ny + ny - 1]
        outE[n, 0] = lo / nx_int
        outE[n, 1] = hi / nx_int
        # 힘: −(p + p_col)·면적 (w 가 안쪽 +)
        for m in range(nm):
            f = 0.0
            for k in range(nw):
                pc = 0.0
                if w[k] > G[k]:
                    pc = Kcol * (w[k] - G[k]) + Ccol * wd[k]
                f -= PW[k, m] * (p[k] + pc) * area_w[k]
            den = 1.0 + ze[m] * om[m] * dt + 0.25 * om[m] * om[m] * dt * dt
            an = (f - 2.0 * ze[m] * om[m] * (v[m] + 0.5 * dt * a[m]) - om[m] * om[m] * (q[m] + dt * v[m] + 0.25 * dt * dt * a[m])) / den
            q[m] = q[m] + dt * v[m] + 0.25 * dt * dt * (a[m] + an)
            v[m] = v[m] + 0.5 * dt * (a[m] + an)
            a[m] = an
    return outQ, outA, outE


#: 모드 3 계수 열의 차례 (`tube_td` VR 열 3–15).
NS_COEF_NAMES = ("cT", "cG", "cK", "cB", "ms", "mb", "fB", "Lc", "bT", "bet", "cC", "ceta", "cF")
#: 세로(장력 방향) 조직 손실 — 감쇠비 ζ. 원본 모형은 막의 장력 모드에 층간 결합 감쇠(C_c)밖에 없어 감쇠비가 약 0.004 였고, 진폭·유량이
#: 생리값의 수 배였다 (원본 논문의 남성 정상 발성도 평균 유량 1.3 L/s, 음압 136 dB). 연조직의 손실 계수는 주파수에 거의 무관하므로
#: (β 고정의 Kelvin–Voigt 는 ζ = βω/2 로 고음에서 커져 ε 0.45 이상의 떨림을 죽였다) 틀마다 β = 2ζ/ω_T (ω_T = √(cT/ms), 줄의 장력 진동수)
#: 로 둔다 — VTL·Story–Titze 의 상수 감쇠비와 같은 뜻. 적합이 화자 상수로 고친다.
TISSUE_ZETA = 0.1
#: 점막 상하 전단의 점성 배율. Serry 표 4 의 η = 1 Pa·s 는 저주파 측정값이다 — 점막의 동적 점성은 주파수와 함께 크게 줄어 (Chan & Titze)
#: 발성 주파수에서는 0.01–0.05 Pa·s 급이고, 1 Pa·s 그대로면 줄 사이 흔들 모드의 감쇠비가 약 0.9 로 점막파를 죽였다 (§52.504).
ETA_SCALE = 0.05
#: 갑상피열근 굽힘 모멘트의 정적 하중을 동역학에 넣을지. 우리 적합의 내전 제어(뒤끝 쉼 변위)는 **정적 평형을 기준으로 한** 쉼 틈이라
#: 정적 불룩함(여성 ε 0.2·a_TA 0.3 에서 약 3–4 mm)이 그 안에 이미 들어 있다 — 기본은 끈다.
STATIC_BULGE = False
#: 줄 사이 상하 결합의 전단 — Titze–Story 덮개(점막 + 인대 절반, `fold_rules`)의 깊이와 덮개 전단 계수 μ_c 500 Pa (Zañartu 코드) 로.
#: Serry 의 G_m 100 Pa × 점막 깊이만 쓰면 결합이 장력의 1/30 이라, 흐름이 아래 줄을 밀어도 위 줄이 따라 열리지 않아 고음(ε ≥ 0.35) 발성 문턱이
#: 폐압 12–18 cmH2O 로 올랐다 (실측 3–4 cmH2O, Titze 1992). 덮개 깊이·μ_c 로는 약 8 배 — 폐압 7 에서 ε 0.35 → 364–386 Hz · 0.45 → 500–555 Hz.
SHEAR_COVER = True


def cover_depth(eps, sex: str = "female", scale_thick=1.0):
    """덮개 깊이 [m] — 점막 + 인대의 절반 (Titze–Story 덮개), 늘어나면 1/√λ."""
    ts = (FEMALE_THICK if sex == "female" else 1.0) * scale_thick
    b0 = SERRY_MALE["b0"] * ts
    d0 = (SERRY_MALE["A_muc"] + 0.5 * SERRY_MALE["A_lig"]) * ts * ts / b0
    return d0 / ((1.0 + eps) ** 0.5)


def ns_coefs(eps: float, a_ta: float, nstrip: int = 6, sex: str = "female", tissue: str = "serry2026",
             scale_len: float = 1.0, scale_thick: float = 1.0, k_scale: float = 1.0, zeta: float | None = None,
             eta_scale: float | None = None, shear_scale: float = 1.0) -> np.ndarray:
    """근육 상태 → 모드 3 계수 13 개 (CGS: dyn, cm, g, s). 보–막을 sin(πx/L) 에 투영하고 상하를 N 줄(칸 중심)로 나눈 것.

    줄 k 의 장력 cT/N = T̃·Δy·(π/L)²·L/2 (cT = T̃ π² b / 2L), 줄 사이 전단 cG = G̃·(L/2)/Δy, 층간 결합 cK·prof_k/N (cK = K_c·b·L/2, 위쪽 기준),
    몸체 cB = [Ñ_b(π/L)² + μ̃(π/L)⁴ + K_f]·L/2 + (K_ra + K_rp)·λ^{3/2}·(π/L)², 질량 ms = ρ_s b L/2 (덮개 전체) · mb = ρ_l L/2,
    정적 하중 fB = −2π M̄ / L (바깥이 + — 갑상피열근 모멘트가 몸체를 안쪽으로), 감쇠 cC = C_c b L/2 · ceta = η̃ (L/2)/Δy · cF = C_f L/2.
    `k_scale` 은 층간 결합 강성 배율 (화자 상수)."""
    g = geometry(eps, a_ta, sex=sex, tissue=tissue, scale_len=scale_len, scale_thick=scale_thick)
    st = SERRY_STRUCT
    L = g.L * 100.0                 # cm
    b = g.b * 100.0
    dy = b / nstrip
    T_m = g.T_m * 1e3               # N/m → dyn/cm
    G_s = g.G_s * 1e3
    if SHEAR_COVER:
        dc = cover_depth(eps, sex=sex, scale_thick=scale_thick)
        G_s = dc * FR.MU_COVER / (1.0 + eps) * 1e3 * shear_scale
    eta_s = g.eta_s * 1e3           # N·s/m → dyn·s/cm
    K_c = st["K_c"] * 0.1 * k_scale  # N/m³ → dyn/cm³
    K_f = st["K_f"] * 10.0          # 보 기초 [N/m²] → dyn/cm² (1 N/m² = 10 dyn/cm²)
    C_c = st["C_c"] * 0.1           # N·s/m³ → dyn·s/cm³
    C_f = st["C_f"] * 10.0          # N·s/m² → dyn·s/cm²
    rho_s = g.rho_s * 0.1           # kg/m² → g/cm²
    rho_l = g.rho_l * 10.0          # kg/m → g/cm
    N_b = g.N_b * 1e5               # N → dyn
    mu = g.mu_b * 1e5 * 1e4         # N·m² → dyn·cm²
    M0 = g.M0 * 1e7                 # N·m → dyn·cm
    Kr = (st["K_ra"] + st["K_rp"]) * 1e7   # N·m → dyn·cm
    k = math.pi / L
    cT = T_m * math.pi ** 2 * b / (2.0 * L)
    cG = G_s * (L / 2.0) / dy
    cK = K_c * b * L / 2.0
    cB = (N_b * k * k + mu * k ** 4 + K_f) * L / 2.0 + Kr * g.lam ** 1.5 * k * k
    ms = rho_s * b * L / 2.0
    mb = rho_l * L / 2.0
    fB = -2.0 * math.pi * M0 / L if STATIC_BULGE else 0.0
    bet = 2.0 * (TISSUE_ZETA if zeta is None else zeta) / math.sqrt(cT / ms)
    cC = C_c * b * L / 2.0
    ceta = eta_s * (ETA_SCALE if eta_scale is None else eta_scale) * (L / 2.0) / dy
    cF = C_f * L / 2.0
    return np.array([cT, cG, cK, cB, ms, mb, fB, L, b, bet, cC, ceta, cF])


#: 모드 3 성대 상태의 잘린 역전파 시정수 [ms] (`tube_td` VS[17], 0 이면 정확한 수반) — 적합 기울기가 떨림 개시 시각의 긴 민감도에 휩쓸리지 않게.
ADJ_TAU_MS = 0.0


#: **닫힘 이음 폭** [cm] (§52.530, `copyfit --vf-closure-edge μm`) — 반틈새가 이 폭 안에서 열림 → 닫힘으로 옮겨 간다 (`tube_td._ns_geom`). 예전 고정값
#: 50 μm (점액막 · 무른 상피) 에서는 m13 의 성대 길이 16 점이 거의 한순간에 닫혀 (뒤끝 쉼 틈 ≈ 0) 음원 유량 미분이 0.3–1 kHz 대비 4–8 · 8–12 · 12–16 kHz
#: −25 · −35 · −43 dB — 여성 모달 LF 음원 (닫힘 되돌림 Ta 0.2–0.6 ms) 의 −36…−42 · −46…−52 · −52…−58 보다 11–17 dB 밝았고, 그 배음이 고역을 채워 가로줄
#: 0.19 · 0.40 · 0.42 (원본 0.06 · 0.09 · 0.02). 200 μm 면 −35 · −50 · −59 (Ta ≈ 0.4 ms 와 같다), 줄 0.24 · 0.13 · 0.12. 이 폭은 한 길이 모드 투영이 담지 못하는
#: 닫힘의 퍼짐 — 점막 표면의 불균일 · 점액 다리 · 앞뒤 위상차(지퍼 닫힘) — 의 유효 크기다. 길이 방향 모드를 풀면 그쪽으로 옮긴다.
CLOSURE_EDGE_CM = 5.0e-3
#: **길이 방향 모드 수** (§52.531, 1 또는 2). 막을 sin(πs) 하나로만 투영하면 성대 길이 전체가 같은 위상으로 닫혀 (뒤끝 쉼 틈 ≈ 0 에서 한순간) 닫힘이 날카로웠다 —
#: 그 날카로움을 `CLOSURE_EDGE_CM` 이 유효값으로 메운다. 2 면 sin(2πs) 가지(막 밴드의 k = 2π/L)를 더한다: 장력 4 배, 같은 전단 · 질량, 몸체와 직교 (층간 결합은 바닥으로),
#: 접촉이 두 모드를 잇는다. 길이 방향 압력은 균일하니 둘째 모드는 삼각 쉼 틈과 접촉의 앞뒤 비대칭으로만 들뜬다 — 앞뒤 위상차(지퍼 닫힘). 수반은 아직 첫 모드만.
LONG_MODES = 1
#: **좌우 성대 따로** (§52.533, `copyfit --vf-lr`). 사용자: *"y축의 위상차가 잘 맞는지도 봐야 … 위상차가 발생하면 이중 음성이 나온다네?"*. 한쪽을 거울로 쓰면
#: 좌우 위상차가 구조상 0 이다 — 정상 습관 발성의 54 % 가 좌우 위상차 ≤ 6 % (PMC7587608), 긴장 비대칭이 크면 (Steinecke & Herzel 1995, 긴장비 ≲ 0.7) 두 성대가
#: 1:1 로 잠기지 못해 반배음 · 이중 음성이 난다. 켜면 상태가 [왼 덮개 · 몸체 · 오른 덮개 · 몸체], 긴장 Q(1 ± `LR_DQ`) · 질량 (1 ± `LR_DM`), 성문 기하는
#: 평균 반틈새, 접촉압은 두 성대에 같게 (`tube_td._ns_mats_lr`). 대칭 (0 · 0) 이면 단일 성대와 같은 운동.
LR_FOLDS = False
LR_DQ = 0.0
LR_DM = 0.0


def vf_static_ns(k_col: float | None = None, ratio: float | None = None, c_col: float | None = None) -> np.ndarray:
    """모드 3 의 정적 변수 19 개 — 앞 14 개는 `tube_td.VF_STATIC` 자리 (입구·출구 길이 [12]·[13] 만 쓴다), [14] 접촉 강성 [dyn/cm³],
    [15] 층간 결합의 입구/위 비, [16] 접촉 감쇠 [dyn·s/cm³], [17] 수반 망각 시정수 [ms], [18] 닫힘 이음 폭 [cm] (`CLOSURE_EDGE_CM`)."""
    from ..engine import tube_td as td
    st = SERRY_STRUCT
    s = np.zeros(23)
    s[:14] = td.VF_STATIC
    s[14] = st["K_col"] * 0.1 if k_col is None else k_col
    s[15] = (COUPLING_RATIO if COUPLING_STIFF_BELOW else 1.0 / COUPLING_RATIO) if ratio is None else ratio
    s[16] = st["C_col"] * 0.1 if c_col is None else c_col
    s[17] = ADJ_TAU_MS
    s[18] = CLOSURE_EDGE_CM
    s[19] = float(LONG_MODES)
    s[20] = 1.0 if LR_FOLDS else 0.0           # 좌우 성대 따로 (§52.533)
    s[21] = LR_DQ
    s[22] = LR_DM
    return s


def medial_gap(sys: System, theta_g: float, r_lower: float = 0.0, r_upper: float = 0.0, theta_conv: float | None = None):
    """중앙면까지의 거리 G (막 절점) [m] — 원본: x·tan θ_g + (아래일수록) (b − y)·tan θ_conv. 뒤끝 쉼 변위 r (x = L 에서의 추가 틈)을
    x/L 로 더한다 (VTL 삼각 성문의 뒤쪽 쉼 변위와 같은 뜻 — 적합에서 내전이 정한다)."""
    tc = SERRY_STRUCT["theta_conv"] if theta_conv is None else theta_conv
    g = sys.g
    X = np.repeat(sys.x, sys.ny)
    Y = np.tile(sys.y, len(sys.x))
    r = r_lower + (r_upper - r_lower) * Y / g.b
    return X * math.tan(theta_g) + (g.b - Y) * math.tan(tc) + r * X / g.L


def simulate(md: Modes, Ps: float = 1000.0, secs: float = 0.3, fs: float = 96000.0, theta_g: float = 0.0,
             r_lower: float = 0.0, r_upper: float = 0.0, seed: int = 0):
    """모드 성대 단독 모의 (성도 없이 성문 위 0 Pa). 반환 dict (Q, A, 날, f0, OQ, 위상차 …)."""
    sys = md.sys
    G = medial_gap(sys, theta_g, r_lower, r_upper)
    area_w = sys.dx * np.tile(sys.wy, len(sys.x))
    nt = int(secs * fs)
    q0 = np.random.default_rng(seed).standard_normal(len(md.f)) * 1e-7
    Q, A, E = _run_modal(2 * np.pi * md.f, md.zeta, np.ascontiguousarray(md.phi_w), md.w_static, G, area_w, Ps, nt, 1.0 / fs,
                         len(sys.x), sys.ny, sys.dx, sys.y[1], SERRY_STRUCT["sep_ratio"], SERRY_STRUCT["K_col"],
                         SERRY_STRUCT["C_col"], q0)
    return _measures(Q, A, E, fs)


def _measures(Q, A, E, fs):
    seg = slice(len(Q) // 2, len(Q))
    u = Q[seg] - Q[seg].mean()
    out = dict(Q=Q, A=A, E=E, ok=bool(np.isfinite(Q).all()))
    if not out["ok"] or u.std() < 1e-12:
        out.update(f0=0.0, oq=1.0, phase=0.0, amp=0.0)
        return out
    ac = np.correlate(u, u, "full")[len(u) - 1:]
    lo, hi = int(fs / 1200), int(fs / 60)
    lag = lo + int(np.argmax(ac[lo:hi]))
    x1, x2 = E[seg, 0] - E[seg, 0].mean(), E[seg, 1] - E[seg, 1].mean()
    cc = np.correlate(x2, x1, "full")
    mid = len(x1) - 1
    h = lag // 2
    dl = int(np.argmax(cc[mid - h:mid + h + 1])) - h
    out.update(f0=fs / lag, oq=float(np.mean(A[seg] > 1e-9)), phase=360.0 * dl / lag, amp=float(np.ptp(E[seg, 0])),
               qpk=float(Q[seg].max()), mfdr=float(-np.diff(Q[seg]).min() * fs))
    return out


@njit(cache=True)
def _run_full(Kp, Ki, Kx, Cp, Ci, Cx, Minv, xs0, fs_static, n_w, G, area_w, Ps, nt, dt, nx_int, ny, dx, dy, sr, Kcol, Ccol,
              x0, dec):
    """전체 상태 명시적 중앙차분 (원본과 같은 방식, 기준판). K·C 는 CSR. `dec` 표본마다 기록."""
    N = Minv.shape[0]
    xp = x0.copy()
    xc = x0.copy()
    xn = np.zeros(N)
    p = np.zeros(n_w)
    dmin = np.zeros(nx_int)
    no = nt // dec
    outQ = np.zeros(no)
    outA = np.zeros(no)
    outE = np.zeros((no, 2))
    w = np.zeros(n_w)
    for n in range(nt):
        for k in range(n_w):
            w[k] = xc[k] + xs0[k]
        Q, Am = _aero(w, G, nx_int, ny, dx, dy, Ps, sr, RHO_AIR, MU_AIR, p, dmin)
        for r in range(N):
            kx = 0.0
            for q in range(Kp[r], Kp[r + 1]):
                kx += Kx[q] * xc[Ki[q]]
            cv = 0.0
            for q in range(Cp[r], Cp[r + 1]):
                cv += Cx[q] * (xc[Ci[q]] - xp[Ci[q]]) / dt
            f = -kx - cv
            if r < n_w:
                pc = 0.0
                if w[r] > G[r]:
                    pc = Kcol * (w[r] - G[r]) + Ccol * (xc[r] - xp[r]) / dt
                f -= (p[r] + pc) * area_w[r]
            xn[r] = 2.0 * xc[r] - xp[r] + dt * dt * Minv[r] * f
        for r in range(N):
            xp[r] = xc[r]
            xc[r] = xn[r]
        if n % dec == 0 and n // dec < no:
            m = n // dec
            outQ[m] = Q
            outA[m] = Am
            lo = 0.0
            hi = 0.0
            for i in range(nx_int):
                lo += w[i * ny]
                hi += w[i * ny + ny - 1]
            outE[m, 0] = lo / nx_int
            outE[m, 1] = hi / nx_int
    return outQ, outA, outE


def simulate_full(sys: System, Ps: float = 1000.0, secs: float = 0.3, dt: float = 1e-6, theta_g: float = 0.0,
                  r_lower: float = 0.0, r_upper: float = 0.0, fs_out: float = 96000.0, seed: int = 0):
    """전체 상태 기준 모의 (작은 간격). 정적 처짐(굽힘 모멘트)에서 출발한다."""
    G = medial_gap(sys, theta_g, r_lower, r_upper)
    area_w = sys.dx * np.tile(sys.wy, len(sys.x))
    xs = np.linalg.solve(sys.K.toarray(), sys.f_s)
    K, C = sys.K.tocsr(), sys.C.tocsr()
    N = K.shape[0]
    x0 = np.random.default_rng(seed).standard_normal(N) * 1e-9
    x0[sys.n_w:] = 0.0
    dec = max(1, int(round(1.0 / (fs_out * dt))))
    nt = int(secs / dt)
    Q, A, E = _run_full(K.indptr, K.indices, K.data, C.indptr, C.indices, C.data, 1.0 / sys.M, xs[:sys.n_w], sys.f_s,
                        sys.n_w, G, area_w, Ps, nt, dt, len(sys.x), sys.ny, sys.dx, sys.y[1], SERRY_STRUCT["sep_ratio"],
                        SERRY_STRUCT["K_col"], SERRY_STRUCT["C_col"], x0, dec)
    return _measures(Q, A, E, 1.0 / (dt * dec))


def _stress_t(e, p, a=None):
    """조직 응력 (torch) — `fold_rules.passive_stress` (+ 능동) 와 같은 식. e·a 는 텐서."""
    import torch
    s0, s2, e1, e2, B = p[:5]
    x = torch.clamp(e - e2, min=0.0)
    sig = -(s0 / e1) * (e - e1) + s2 * (torch.exp(B * x) - 1.0 - B * x)
    if a is not None and p[5] > 0.0:
        sig = sig + a * p[5] * torch.clamp(1.0 - p[7] * (e - p[6]) ** 2, min=0.0)
    return sig


def ns_coefs_torch(eps, a_ta, nstrip: int = 6, sex: str = "female", tissue: str = "serry2026", scale_len=1.0, scale_thick=1.0,
                   k_scale=1.0, zeta=None, eta_scale=None, shear_scale=1.0):
    """`ns_coefs` 의 torch 판 — ε·a_TA (틀 텐서 (B, F)) 와 화자 배율(0 차원 텐서)에 대해 미분 가능. 반환 (B, F, 13).

    보의 굽힘 강성·축력은 몸체 강성 cB 에만 들어가고(몸체는 거의 서 있다), 정적 하중은 `STATIC_BULGE` 가 꺼져 있으면 0."""
    import torch
    p = SERRY_MALE
    st = SERRY_STRUCT
    ts_ = FR.TISSUE_SETS[tissue]
    ls = (FEMALE_LEN if sex == "female" else 1.0) * scale_len
    tsc = (FEMALE_THICK if sex == "female" else 1.0) * scale_thick
    L0, b0 = p["L0"] * ls, p["b0"] * tsc
    A0 = {k: p[f"A_{k}"] * tsc * tsc for k in ("muc", "lig", "ta")}
    lam = 1.0 + eps
    l2 = torch.rsqrt(lam)
    d = {k: A0[k] / b0 * l2 for k in A0}
    A = {k: A0[k] / lam for k in A0}
    b = b0 * l2
    s_muc = _stress_t(eps, ts_["muc"])
    s_lig = _stress_t(eps, ts_["lig"])
    s_ta = _stress_t(eps, ts_["ta"], a_ta)
    h = 1e-4
    E_lig = torch.clamp((_stress_t(eps + h, ts_["lig"]) - _stress_t(eps - h, ts_["lig"])) / (2 * h), min=1.0)
    E_ta = torch.clamp((_stress_t(eps + h, ts_["ta"], a_ta) - _stress_t(eps - h, ts_["ta"], a_ta)) / (2 * h), min=1.0)
    rho_s = p["rho_muc"] * d["muc"]
    rho_l = p["rho_lig"] * A["lig"] + p["rho_ta"] * A["ta"]
    I_lig = b * d["lig"] ** 3 / 12.0
    I_ta = b * d["ta"] ** 3 / 12.0
    rc = d["ta"] + 0.5 * d["lig"]
    r_ta = 0.5 * d["ta"]
    a_lig = -(d["ta"] + d["lig"]) / (2.0 * (E_lig * A["lig"]) / (E_ta * A["ta"]) + 1.0)
    a_ta_ = (d["ta"] + d["lig"]) / (2.0 * (E_ta * A["ta"]) / (E_lig * A["lig"]) + 1.0)
    F_lig, F_ta = lam * A["lig"] * s_lig, lam * A["ta"] * s_ta
    T_m = lam ** 1.5 * (s_muc * lam) * d["muc"]
    M0 = (rc - r_ta) * F_ta
    th = E_lig * I_lig + E_ta * I_ta + (rc - rc) * A["lig"] * E_lig * a_lig + (rc - r_ta) * A["ta"] * E_ta * a_ta_
    mu = lam ** 1.5 * lam ** 2 * th
    N_b = lam ** 1.5 * (F_lig + F_ta)
    G_s = d["muc"] * st["G_m"] / lam
    if SHEAR_COVER:
        G_s = (d["muc"] + 0.5 * d["lig"]) * FR.MU_COVER / lam * shear_scale
    eta_s = d["muc"] * st["eta_m"] / lam
    # CGS
    L = L0 * lam * 100.0
    bc = b * 100.0
    dy = bc / nstrip
    k = math.pi / L
    cT = T_m * 1e3 * math.pi ** 2 * bc / (2.0 * L)
    cG = G_s * 1e3 * (L / 2.0) / dy
    cK = st["K_c"] * 0.1 * k_scale * bc * L / 2.0
    Kr = (st["K_ra"] + st["K_rp"]) * 1e7
    cB = (N_b * 1e5 * k * k + mu * 1e9 * k ** 4 + st["K_f"] * 10.0) * L / 2.0 + Kr * lam ** 1.5 * k * k
    ms = rho_s * 0.1 * bc * L / 2.0
    mb = rho_l * 10.0 * L / 2.0
    fB = (-2.0 * math.pi * M0 * 1e7 / L) if STATIC_BULGE else torch.zeros_like(L)
    z = TISSUE_ZETA if zeta is None else zeta
    bet = 2.0 * z / torch.sqrt(cT / ms)
    cC = st["C_c"] * 0.1 * bc * L / 2.0
    ceta = eta_s * 1e3 * (ETA_SCALE if eta_scale is None else eta_scale) * (L / 2.0) / dy
    cF = st["C_f"] * 10.0 * L / 2.0
    return torch.stack([cT, cG, cK, cB, ms, mb, fB, L, bc, bet, cC, ceta, cF], -1)
