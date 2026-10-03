"""성대 막의 **고유모드 밴드 구조** — 집중 질량 모형을 대신한다.

사용자: *"그냥 실제로 두께가 있는 막을 물리로 만들어서 시뮬레이션을 돌리고, 막이 늘어나고,
두께가 줄어들 때의 진동 모드를 죄다 기록해서 남기라는 의미야. phonon mode 조사하듯이."*

무엇을 푸는가
-------------
성대를 **두께가 있는 탄성 연속체**로 본다. 단면은 (x: 내외측 깊이 D, y: 상하 두께 T) 이고,
앞뒤(z, 성대 길이 L) 방향은 균일하다고 보아 **푸리에 분해**한다:

    u(x, y, z, t) = û(x, y) · exp(i k z) · exp(i ω t)

k 를 촘촘히 훑으면 각 k 마다 고유진동수 가지가 나온다 — 그것이 ω_n(k) **밴드 구조**다.
포논 밴드와 같은 구조이고, 사용자가 말한 "DFT 로 k-sampling 을 촘촘하게" 가 이것이다.
경계가 고정된 막의 정재파는 k = nπ/L 에 해당하므로, 밴드에서 그 k 를 읽으면 실제 모드가 된다.

조직 모형
---------
성대는 **횡등방성**이다 (섬유가 앞뒤 z 축). 독립 상수를 다섯으로 둔다:

    E_z   z 방향 영률          — 늘어남에 따라 지수적으로 는다 (아래)
    E_t   x–y 면내 영률
    G_z   z 를 포함한 전단      (cover 1~10 kPa, body 10~40 kPa 급)
    nu_t  면내 푸아송비
    nu_zt z–t 푸아송비

늘어남 ε 의 효과 둘:
  (1) **강성** — 성대 조직의 응력-변형은 지수형이다. E_z(ε) = E_z0 · (1 + A·(exp(B·ε) − 1)).
  (2) **기하** — 늘어나면 얇아진다. 부피가 보존되면 T·D·(1+ε) = T0·D0 이므로
      등방 수축 가정에서 T = T0/√(1+ε), D = D0/√(1+ε).

이산화
------
단면을 N_x × N_y 격자로 나누고 변위 세 성분 (u, v, w) 를 절점에 둔다. z 의존이 exp(ikz) 이므로
∂/∂z → i k 다. w → i·w̃ 로 치환하면 모든 변형률이 실수가 되어 **실대칭** 일반화 고유값 문제가 된다:

    K(k, ε) · û = ω² · M · û

경계: 외측(x = D, 갑상연골·근육 쪽)은 고정, 나머지 면은 자유다.

검증
----
이 파일을 직접 실행하면 ε 에 따른 최저 가지의 진동수를 찍는다. 여성 화자의 생리 범위
(이완 시 180~250 Hz, 30~40 % 늘어남에서 400~600 Hz) 를 재현해야 물성이 맞게 잡힌 것이다.
그 표를 통과하지 못하면 **상수를 고쳐야지 결과를 쓰면 안 된다.**
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#: 조직 밀도 [kg/m³]. 성대는 대부분 물이다.
RHO = 1040.0

#: 이완 상태의 물성 [Pa]. cover 와 body 의 중간값을 하나로 본다 (한 층 모형).
E_Z0 = 20.0e3          # z(앞뒤) 영률
E_T = 4.0e3            # 면내 영률 — 섬유와 직각이라 훨씬 무르다
G_Z = 12.0e3           # z 를 포함한 전단
NU_T = 0.45            # 면내 푸아송비. 0.4999 로 두면 유한차분이 체적 잠김에 걸린다
NU_ZT = 0.45

#: 지수형 응력-변형 (성대 조직). E_z(ε) = E_z0·(1 + A(exp(Bε) − 1))
STRAIN_A = 1.2
STRAIN_B = 6.0
#: 이완 상태의 세로 응력 [Pa] — 내전만 해도 성대에는 얼마간 장력이 걸려 있다.
#: **이 화자(yang_female)에 맞춘 값이다.** 기준점 둘로 역추정했다 —
#: `wavs/` 20 개의 파일별 중앙 F0 하위 5 % = 231 Hz(흉성 이완)와 프로파일의 f0_hi = 580 Hz
#: (가성). (L0, SIGMA0) 격자 20 점을 훑어 로그 오차 합이 최소인 것을 골랐다:
#: L0 11.0 mm, SIGMA0 3.0 kPa -> 흉성 237 / 가성 563 Hz (오차 2.6 % / 2.9 %).
SIGMA0 = 3.0e3

#: 이완 상태의 치수 [m]. 성인 여성의 막성부.
L0 = 11.0e-3           # 앞뒤 길이 — 이 화자에 맞춘 값 (위 SIGMA0 주석 참조)
T0 = 3.0e-3            # 상하 두께
D0 = 4.0e-3            # 내외측 깊이

#: **갑상피열근(TA) 수축 축** ta ∈ [0, 1]. TA 가 수축하면 성대가 두꺼워지고(내측면 세로 두께) 진동에 참여하는 깊이가 는다.
#: 두께 증가 상한 +20 % 는 MRI 기반 유한요소 연구의 값(TA 활성에 따라 −16~+20 %). 깊이 계수는 **이 화자에 맞춘 값**이다 —
#: 늘어남만으로는 F0 하한이 약 222 Hz 라 목표 1~5 % 분위(186~199 Hz) 에 못 닿았다. ta 1 에서 173 Hz 까지 내려간다 (§52.22).
#: 짧아짐(TA 의 길이 효과) 은 늘어남 ε 자체에 들어 있다. 한 층 모형이라 체부 능동 응력은 넣지 않는다.
TA_THICK = 0.2
TA_DEPTH = 0.75


def geometry(strain: float, ta: float = 0.0) -> tuple[float, float, float]:
    """늘어남 ε, TA 수축 ta -> (L, T, D). 부피 보존 + 등방 수축에 TA 의 두께·깊이 증가를 곱한다."""
    s = 1.0 + float(strain)
    a = float(ta)
    return L0 * s, T0 / np.sqrt(s) * (1.0 + TA_THICK * a), D0 / np.sqrt(s) * (1.0 + TA_DEPTH * a)


def moduli(strain: float) -> tuple[float, float, float]:
    """늘어남 ε -> (E_z, E_t, G_z). z 만 지수적으로 굳는다."""
    f = 1.0 + STRAIN_A * (np.exp(STRAIN_B * float(strain)) - 1.0)
    return E_Z0 * f, E_T, G_Z * (1.0 + 0.5 * (f - 1.0))


def sigma_z(strain: float) -> float:
    """세로 **응력** [Pa]. 지수형 응력-변형을 적분한 것 — ε=0 에서 기울기가 E_z0 이 되게 잡는다.

    **이것이 F0 를 지배한다.** 현의 진동수는 √(σ/ρ)/2L 이지 √(E/ρ)/2L 이 아니다. 늘어남이
    응력을 지수적으로 올리고, 그것이 가로 변위에 대한 기하 강성 σ·k² 로 들어간다. 이 항이
    빠지면 최저 가지가 이완 93 Hz 로 나와 여성 화자의 생리 범위(180~250)를 못 맞춘다.
    """
    e = float(strain)
    return SIGMA0 + (E_Z0 * STRAIN_A / STRAIN_B) * (np.exp(STRAIN_B * e) - 1.0)


def _stiffness_matrix(E_z: float, E_t: float, G_z: float) -> np.ndarray:
    """횡등방성 강성 C (6x6, Voigt). 면내 등방, 섬유축 z."""
    nt, nzt = NU_T, NU_ZT
    # 면내 등방 + z 축 방향만 다른 표준형을 컴플라이언스로 쌓고 뒤집는다.
    S = np.zeros((6, 6))
    S[0, 0] = S[1, 1] = 1.0 / E_t
    S[2, 2] = 1.0 / E_z
    S[0, 1] = S[1, 0] = -nt / E_t
    S[0, 2] = S[2, 0] = S[1, 2] = S[2, 1] = -nzt / E_z
    G_t = E_t / (2.0 * (1.0 + nt))
    S[3, 3] = 1.0 / G_t          # γ_xy
    S[4, 4] = S[5, 5] = 1.0 / G_z  # γ_xz, γ_yz
    return np.linalg.inv(S)


def bands(k_list, strain: float, nx: int = 16, ny: int = 12,
          n_modes: int = 8, shapes: bool = False, contact: float = 0.0,
          vib_depth: float = 1.0, ta: float = 0.0):
    """각 k 에서 가장 낮은 `n_modes` 개 고유진동수 [Hz]. 반환 (len(k_list), n_modes).

    `shapes=True` 면 `(f, vec, meta)` 를 준다. `vec` 는 (len(k), 3n, n_modes) 이고
    `meta` 는 모드마다 (가로 에너지 몫, 내측면 면적 참여도) 다.

    **모드 분류가 필요하다.** 장력은 가로 변위 (u, v) 에만 걸리므로 σ 를 키우면 최저 가지가
    세로(w) 모드로 갈아탄다 — "최저 = 가로" 라고 가정하면 검증이 틀린 것을 보고도 모른다.
    소리를 내는 것은 성문 틈의 면적, 즉 **내측면의 u** 이므로 그 참여도도 같이 낸다.
    """
    L, T, D = geometry(strain, ta)
    E_z, E_t, G_z = moduli(strain)
    C = _stiffness_matrix(E_z, E_t, G_z)
    dx, dy = D / (nx - 1), T / (ny - 1)
    n = nx * ny
    # **자유도 배열은 블록 순서다**: [u(0..n-1), v(0..n-1), w̃(0..n-1)].
    # `sp.bmat`/`sp.block_diag` 로 쌓는 연산자가 이 순서를 전제한다. 절점 묶음(u,v,w 교대)
    # 으로 색인하면 경계조건이 엉뚱한 자유도를 고정해서 조용히 틀린 답이 나온다 —
    # 실제로 그렇게 짰다가 현(string) 극한 검증에서 138.7 Hz 대신 3.7 Hz 가 나왔다.
    idx = lambda i, j, c: c * n + (i * ny + j)          # noqa: E731

    # 1 차 중앙차분 연산자 (자유 경계는 한쪽 차분).
    def dmat(axis):
        rows, cols, vals = [], [], []
        for i in range(nx):
            for j in range(ny):
                p = i * ny + j
                if axis == 0:
                    a = max(i - 1, 0); b = min(i + 1, nx - 1); h = (b - a) * dx
                    rows += [p, p]; cols += [a * ny + j, b * ny + j]
                else:
                    a = max(j - 1, 0); b = min(j + 1, ny - 1); h = (b - a) * dy
                    rows += [p, p]; cols += [i * ny + a, i * ny + b]
                vals += [-1.0 / h, 1.0 / h]
        return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    Dx, Dy = dmat(0), dmat(1)
    I = sp.identity(n, format="csr")
    Z = sp.csr_matrix((n, n))
    out, vs, ms = [], [], []
    for k in np.asarray(k_list, float):
        # 변형률 = B · [u, v, w̃].  w = i·w̃ 치환으로 전부 실수가 된다.
        B = sp.bmat([
            [Dx,      Z,      Z],            # e_xx
            [Z,       Dy,     Z],            # e_yy
            [Z,       Z,      -k * I],       # e_zz = ∂w/∂z -> -k w̃
            [Dy,      Dx,     Z],            # g_xy
            [k * I,   Z,      Dx],           # g_xz
            [Z,       k * I,  Dy],           # g_yz
        ], format="csr")
        Cb = sp.block_diag([C[a, a] * I for a in range(6)], format="csr").tolil()
        # 대각 밖의 결합(C12, C13 …) 을 채운다.
        for a in range(3):
            for b in range(3):
                if a != b:
                    Cb[a * n:(a + 1) * n, b * n:(b + 1) * n] = C[a, b] * I
        Cb = Cb.tocsr()
        w_cell = dx * dy
        K = (B.T @ Cb @ B) * w_cell
        # **기하 강성** — 세로 응력 σ 아래의 가로 변위는 ½∫σ(∂u/∂z)² 만큼 더 든다.
        # ∂/∂z -> ik 이므로 σ·k²·(u² + v²). 현의 장력이 F0 를 만드는 것이 이 항이다.
        sg = sigma_z(strain) * (k ** 2) * w_cell
        K = K + sp.block_diag([sg * I, sg * I, Z], format="csr")
        M = sp.identity(3 * n, format="csr") * (RHO * w_cell)
        # 외측 벽(x = D) 고정. `vib_depth` 가 1 보다 작으면 **깊은 쪽을 더 묶는다** —
        # 가성에서 성대체(body)가 굳어 점막(cover)만 진동하는 것이 이것이다. 진동하는
        # 깊이가 얕아지면 같은 장력에서도 진동수가 크게 오른다. 흉성은 1.0 이다.
        nfix = max(1, int(round((1.0 - float(np.clip(vib_depth, 0.05, 1.0))) * nx)) + 1)
        fixed = [idx(i, j, c) for i in range(nx - nfix, nx)
                 for j in range(ny) for c in range(3)]
        # **닫힌 구간: 두 성대가 만나 하나의 판막이 된다.** 사용자: *"성대가 만나면서 하나의
        # 큰 판막이 될 건데, 그 판막 자체의 진동(스피커처럼)도 고려해야 하고"*. 맞닿은
        # 내측면(x = 0)은 상대 성대에 막혀 **법선 변위가 0** 이다 (좌우 대칭 모드).
        # 접선(v, w)은 미끄러지므로 자유다.
        #
        # `contact` 는 **맞닿는 높이의 몫** (0~1) 이다. 이진값이 아니어야 하는 이유가
        # 가성이다 — 사용자: *"성대가 서로 닿지 않는 가성도 만들어야 해. 이 경우는 판막이
        # 완전히 하나로 합쳐지지가 않으니"*. 흉성은 두께 전체가 닿으므로 1 에 가깝고,
        # 가성은 닿지 않거나(0) 상연만 스치므로 0~0.3 이다. 아래(하연)부터 닿는다.
        nc = int(round(float(np.clip(contact, 0.0, 1.0)) * ny))
        if nc > 0:
            fixed += [idx(0, j, 0) for j in range(nc)]
        keep = np.setdiff1d(np.arange(3 * n), fixed)
        Kr, Mr = K[keep][:, keep], M[keep][:, keep]
        # **시작 벡터를 고정한다.** ARPACK 은 기본으로 무작위 시작 벡터를 써서 같은 호출도 고유벡터가 1e-13 쯤 달라졌고,
        # 자려 진동이 그 차이를 키워 같은 인자의 `simulate` 두 번이 0.02 s 만에 U 최대의 0.7 % 만큼 갈라졌다 (§52.23).
        # 상수 벡터는 대칭 때문에 어떤 모드와 직교할 수 있으므로 고정 씨앗의 가우시안을 쓴다.
        v0 = np.random.default_rng(12345).standard_normal(Kr.shape[0])
        vals, vecs = spla.eigsh(Kr.tocsc(), k=n_modes, M=Mr.tocsc(), sigma=0.0,
                                which="LM", v0=v0)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        f = np.sqrt(np.maximum(vals, 0.0)) / (2.0 * np.pi)
        out.append(f)
        if shapes:
            full = np.zeros((3 * n, n_modes))
            full[keep] = vecs
            u, v, w = full[:n], full[n:2 * n], full[2 * n:]
            e_tr = (u ** 2 + v ** 2).sum(0) / np.maximum((full ** 2).sum(0), 1e-30)
            med = np.array([i * ny + j for j in range(ny) for i in (0,)])  # x = 0 면
            area = np.abs(u[med].sum(0)) * dy / np.maximum(
                np.sqrt((full ** 2).sum(0)), 1e-30)
            vs.append(full); ms.append(np.stack([e_tr, area], 1))
    if shapes:
        return np.array(out), np.array(vs), np.array(ms)
    return np.array(out)


def phonation_mode(strain: float, contact: float = 0.0, vib_depth: float = 1.0,
                   nx: int = 16, ny: int = 12, n_modes: int = 8,
                   frac: float = 0.15) -> tuple[float, float]:
    """발성이 타는 모드 — **성문 면적을 움직일 수 있는 모드 중 가장 낮은 것**.

    참여도가 최대인 모드를 고르면 안 된다. 자려 진동은 에너지를 가장 싸게 받는 가지에
    걸리고, 그것은 면적을 바꿀 수 있는 가지 중 최저다. `frac` 은 "면적을 실제로 움직인다"
    고 볼 참여도의 문턱 (그 k 에서의 최대 참여도 대비 몫).

    반환 (그 모드의 진동수 [Hz], 참여도).
    """
    L = geometry(strain)[0]
    f, _v, m = bands([np.pi / L], strain, nx=nx, ny=ny, n_modes=n_modes,
                     shapes=True, contact=contact, vib_depth=vib_depth)
    f, m = f[0], m[0]
    a = m[:, 1]
    thr = frac * a.max() if a.max() > 0 else np.inf
    ok = np.flatnonzero(a >= thr)
    if not len(ok):
        return float("nan"), 0.0
    i = int(ok[np.argmin(f[ok])])
    return float(f[i]), float(a[i])


# ---------------------------------------------------------------------------------------------------------------------
# **층 구조 막** (§52.504) — 점막(덮개)·인대·갑상피열근(몸체) 세 층, 층마다 조직 응력–변형과 능동 응력 (`fold_rules`).
#
# 위 한 층 모형은 이 화자의 F0 범위에 맞춘 상수 하나(σ0 3 kPa, 전단 12 kPa)로 모든 층을 대신했다. 여기서는 층마다 후속 연구의 조직
# 법칙을 쓴다: 세로 응력 λ·σ̄_i(ε) (Serry et al. 2026 식 11 — 늘어난 단면 기준의 유효 응력) 가 가로 변위의 기하 강성 σ·k² 가 되고, 접선
# 영률 λ²·dσ̄_i/dε 가 세로 강성이다. 단면(내외측 깊이 × 상하 두께)의 전단은 층마다 μ (덮개 500 · 몸체 1000 Pa, Zañartu 코드). 늘어나면
# 비압축 조건으로 두께·깊이가 1/√λ (Serry 식 4). 바깥 벽(갑상연골)은 고정, 나머지는 자유.
#
# 그리고 **Titze–Story 의 경험 규칙이 하던 일을 모드에서 뽑는다** (`layered_lumped`): 마디점(아래·위 덮개의 경계)은 흔들 모드의 내측면
# 변위가 부호를 바꾸는 자리, 세 블록(아래 덮개·위 덮개·몸체)의 질량·용수철은 가장 낮은 세 모드를 그 블록 좌표로 정확히 축약한 계를
# 몸체–덮개 위상(질량 대각, 용수철 넷)에 최소제곱으로 맞춘 것이다.
# ---------------------------------------------------------------------------------------------------------------------


def _dmats(nx: int, ny: int, dx: float, dy: float):
    """1 차 중앙차분 (자유 경계는 한쪽 차분) — `bands` 와 같은 연산자."""
    n = nx * ny
    out = []
    for axis in (0, 1):
        rows, cols, vals = [], [], []
        for i in range(nx):
            for j in range(ny):
                p = i * ny + j
                if axis == 0:
                    a = max(i - 1, 0); b = min(i + 1, nx - 1); h = (b - a) * dx
                    rows += [p, p]; cols += [a * ny + j, b * ny + j]
                else:
                    a = max(j - 1, 0); b = min(j + 1, ny - 1); h = (b - a) * dy
                    rows += [p, p]; cols += [i * ny + a, i * ny + b]
                vals += [-1.0 / h, 1.0 / h]
        out.append(sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr())
    return out


def layered_geometry(eps: float, sex: str = "female", scale: float = 1.0):
    """늘어남 ε → (L, T, 층 깊이 {muc, lig, ta}) [m]. 비압축: 두께·깊이 1/√λ."""
    from .fold_rules import DIMS
    d = DIMS[sex]
    lam = 1.0 + float(eps)
    s = scale / np.sqrt(lam)
    return (d["L0"] * scale * lam, d["T0"] * s,
            {"muc": d["Dmuc"] * s, "lig": d["Dlig"] * s, "ta": d["Dmus"] * s})


def _layered_system(eps: float, a_ta: float, k: float, tissue: str, sex: str, nx: int, ny: int,
                    mu_c: float | None, mu_b: float | None, scale: float):
    """k 하나에서의 (K, M) — 바깥 벽 고정 자유도를 뺀 것 — 와 정보."""
    from . import fold_rules as FR
    mu_c = FR.MU_COVER if mu_c is None else mu_c
    mu_b = FR.MU_BODY if mu_b is None else mu_b
    L, T, dep = layered_geometry(eps, sex, scale)
    lam = 1.0 + float(eps)
    D = dep["muc"] + dep["lig"] + dep["ta"]
    dx, dy = D / (nx - 1), T / (ny - 1)
    n = nx * ny
    xs = np.repeat(np.arange(nx) * dx, ny)
    ys = np.tile(np.arange(ny) * dy, nx)
    lay = np.where(xs < dep["muc"] - 1e-12, 0, np.where(xs < dep["muc"] + dep["lig"] - 1e-12, 1, 2))
    st = FR.layer_stress(eps, a_ta, tissue)
    em = FR.layer_modulus(eps, a_ta, tissue)
    sig_l = np.array([float(st["muc"]), float(st["lig"]), float(st["ta"])]) * lam
    ez_l = np.maximum(np.array([float(em["muc"]), float(em["lig"]), float(em["ta"])]) * lam ** 2, 50.0)
    mu_l = np.array([mu_c, 0.5 * (mu_c + mu_b), mu_b])
    Cl = [_stiffness_matrix(ez_l[q], 2.0 * mu_l[q] * (1.0 + NU_T), mu_l[q]) for q in range(3)]
    Cn = np.stack([Cl[q] for q in lay])                    # (n, 6, 6)
    sig = sig_l[lay]
    Dx, Dy = _dmats(nx, ny, dx, dy)
    I = sp.identity(n, format="csr")
    Z = sp.csr_matrix((n, n))
    w_cell = dx * dy
    M = sp.identity(3 * n, format="csr") * (FR.RHO * w_cell)
    fixed = [c * n + ((nx - 1) * ny + j) for j in range(ny) for c in range(3)]    # 바깥 벽(갑상연골) 고정
    keep = np.setdiff1d(np.arange(3 * n), fixed)
    Cb = sp.bmat([[sp.diags(Cn[:, a, b]) for b in range(6)] for a in range(6)], format="csr")
    B = sp.bmat([[Dx, Z, Z], [Z, Dy, Z], [Z, Z, -k * I], [Dy, Dx, Z], [k * I, Z, Dx], [Z, k * I, Dy]], format="csr")
    K = (B.T @ Cb @ B) * w_cell
    sg = sp.diags(sig * k * k * w_cell)
    K = K + sp.block_diag([sg, sg, Z], format="csr")
    info = dict(L=L, T=T, D=D, dep=dep, xs=xs, ys=ys, lay=lay, nx=nx, ny=ny, n=n, keep=keep, dy=dy, dx=dx,
                m_node=FR.RHO * w_cell, sig=sig_l, ez=ez_l)
    return K[keep][:, keep].tocsc(), M[keep][:, keep].tocsc(), info


def layered_bands(k_list, eps: float, a_ta: float = 0.0, tissue: str = "ts2002", sex: str = "female",
                  nx: int = 25, ny: int = 11, n_modes: int = 12, mu_c: float | None = None, mu_b: float | None = None,
                  scale: float = 1.0):
    """층 구조 막의 밴드 ω_n(k). 반환 (f (len(k), n_modes) [Hz], vec (len(k), 3n, n_modes) M-정규, info dict)."""
    out, vs, info = [], [], None
    for k in np.asarray(k_list, float):
        Kr, Mr, info = _layered_system(eps, a_ta, k, tissue, sex, nx, ny, mu_c, mu_b, scale)
        v0 = np.random.default_rng(12345).standard_normal(Kr.shape[0])
        vals, vecs = spla.eigsh(Kr, k=n_modes, M=Mr, sigma=0.0, which="LM", v0=v0)
        o = np.argsort(vals)
        full = np.zeros((3 * info["n"], n_modes))
        full[info["keep"]] = vecs[:, o]
        out.append(np.sqrt(np.maximum(vals[o], 0.0)) / (2 * np.pi))
        vs.append(full)
    return np.array(out), np.array(vs), info


def layered_lumped(eps: float, a_ta: float = 0.0, tissue: str = "ts2002", sex: str = "female", nx: int = 25, ny: int = 11,
                   cover_lig: float = 0.5, iters: int = 6, verbose: bool = False, mu_c: float | None = None,
                   mu_b: float | None = None, scale: float = 1.0):
    """층 구조 막 → 몸체–덮개 집중 매개변수 (`fold_rules.Lumped`, 온 성대 기준 SI) + 진단.

    **하중 Ritz 축약** — 성문이 성대에 거는 것은 아래·위 내측면의 압력이다. 그 하중에 대한 연속체의 정적 응답 φ_l = K⁻¹f_l,
    φ_u = K⁻¹f_u 와 몸체의 관성 하중 응답 φ_b = K⁻¹M·1_몸체 를 기저로 삼아 (Wilson 의 하중 의존 Ritz 벡터) 3 자유도계
    Φᵀ K Φ, Φᵀ M Φ 를 만든다. 가장 낮은 모드로 고르면 무른 갑상피열근 속의 깊이 모드(ε 0.2 에서 279–399 Hz 에 12 개)가 먼저 걸려
    덮개를 놓쳤다. 블록 좌표는 **아래·위 내측면의 평균 가로 변위** (압력이 일하는 좌표 — 집중 모형의 질량 변위가 면적을 정하는 것과
    같다) 와 몸체 블록의 평균. 마디점 Zn 은 Ritz 계의 흔들 모드(아래·위 좌표의 부호가 반대인 모드)의 내측면 변위가 0 을 지나는
    자리 — 규칙 (1 + a_TA)T/3 대신 — 로, Zn 을 바꾸면 하중이 바뀌므로 몇 번 되풀이한다. 끝으로 몸체–덮개 위상(질량 대각, 용수철
    kl·ku·kc·kb) 을 Ritz 계의 세 고유진동수(log)와 블록 좌표 모드 모양에 최소제곱으로 맞춘다. 단위 z 길이당 값에 L 을 곱한다
    (Titze–Story 관례).
    """
    from scipy.linalg import eigh
    from scipy.optimize import least_squares
    from . import fold_rules as FR
    L = layered_geometry(eps, sex, scale)[0]
    Kr, Mr, info = _layered_system(eps, a_ta, np.pi / L, tissue, sex, nx, ny, mu_c, mu_b, scale)
    lu = spla.splu(Kr)
    n, keep, xs, ys, T, dep = info["n"], info["keep"], info["xs"], info["ys"], info["T"], info["dep"]
    Dc = dep["muc"] + cover_lig * dep["lig"]
    pos = np.full(3 * n, -1)
    pos[keep] = np.arange(len(keep))
    med = np.flatnonzero(xs < 1e-12)
    ym = ys[med]
    body = np.flatnonzero((xs >= Dc - 1e-12) & (xs < info["D"] - 1e-12))
    m_node = float(Mr.diagonal().mean())

    def red_vec(idx_u, w):
        f = np.zeros(Kr.shape[0])
        p = pos[idx_u]
        ok = p >= 0
        np.add.at(f, p[ok], np.asarray(w, float)[ok])
        return f

    zn, rock = T / 3.0, None
    for _ in range(iters):
        sl, su = ym < zn, ym >= zn
        f_l = red_vec(med[sl], np.where(ym[sl] <= 1e-12, 0.5, 1.0) * info["dy"])
        f_u = red_vec(med[su], np.where(ym[su] >= T - 1e-12, 0.5, 1.0) * info["dy"])
        f_b = red_vec(body, np.full(len(body), m_node))
        Phi = np.stack([lu.solve(f_l), lu.solve(f_u), lu.solve(f_b)], 1)
        Kp, Mp = Phi.T @ (Kr @ Phi), Phi.T @ (Mr @ Phi)
        w2, A = eigh(Kp, Mp)
        U = Phi @ A                                        # M-정규 Ritz 모드 (감소 자유도)
        Uf = np.zeros((3 * n, 3))
        Uf[keep] = U
        um = Uf[med]                                       # 내측면 u (ny, 3)
        rock = None
        for j in range(3):
            a_, b_ = um[sl, j].mean(), um[su, j].mean()
            if a_ * b_ < 0:
                rock = j
                break
        if rock is None:
            break
        s = np.sign(um[:, rock])
        jj = np.flatnonzero(s[:-1] * s[1:] < 0)
        if not len(jj):
            break
        j0 = jj[0]
        zn_new = ym[j0] + (ym[j0 + 1] - ym[j0]) * um[j0, rock] / (um[j0, rock] - um[j0 + 1, rock])
        zn_new = float(np.clip(zn_new, 0.15 * T, 0.85 * T))
        done = abs(zn_new - zn) < 0.01 * T
        zn = zn_new
        if done:
            break
    sl, su = ym < zn, ym >= zn
    W = np.zeros((3, 3 * n))
    W[0, med[sl]] = 1.0 / sl.sum()
    W[1, med[su]] = 1.0 / su.sum()
    W[2, body] = 1.0 / len(body)
    Psi = (W @ Uf) / np.sqrt(L)                          # 온 성대 기준 (질량·강성 ×L)
    Om = np.sqrt(np.maximum(w2, 0.0))
    Pi = np.linalg.inv(Psi)
    Mq = Pi.T @ Pi
    Kq = Pi.T @ np.diag(Om ** 2) @ Pi
    Tl, Tu = zn, T - zn
    x0 = np.log(np.concatenate([np.abs(np.diag(Mq)),
                                np.maximum(np.abs([Kq[0, 2], Kq[1, 2], Kq[0, 1], Kq[2].sum()]), 1e-4 * np.abs(Kq).max())]))

    def lump(x):
        e = np.exp(x)
        return FR.Lumped(eps, L, T, Tl, Tu, e[0], e[1], e[2], e[3], e[4], e[5], e[6])

    def res(x):
        Ml, Kl = lump(x).matrices()
        ww, Vl = eigh(Kl, Ml)
        r = [0.5 * np.log(np.maximum(ww, 1e-9) / np.maximum(Om ** 2, 1e-9))]
        for j in range(3):
            a = Vl[:, j] * np.sign(Vl[:, j] @ Psi[:, j] + 1e-30)
            r.append((a - Psi[:, j]) / np.linalg.norm(Psi[:, j]))
        return np.concatenate(r)

    sol = least_squares(res, x0, method="lm", max_nfev=6000)
    lp = lump(sol.x)
    diag = dict(zn=zn, T=T, f_ritz=Om / (2 * np.pi), Mq=Mq, Kq=Kq, Psi=Psi, fit_rms=float(np.sqrt(np.mean(sol.fun ** 2))),
                Dc=Dc, D=info["D"], sig=info["sig"], rock=rock)
    if verbose:
        print(f"  ε {eps:+.2f} a_TA {a_ta:.2f}: Zn/T {zn / T:.2f} · Ritz {np.round(diag['f_ritz'], 0)} Hz · 맞춤 rms {diag['fit_rms']:.3f}")
    return lp, diag


def main() -> int:
    print("성대 막 고유모드 — 늘어남에 따른 최저 가지 (검증표)")
    print("  ε      L[mm]  T[mm]  E_z[kPa]  σ[kPa]   k=π/L 에서의 최저 4 가지 [Hz]")
    for eps in (0.0, 0.1, 0.2, 0.3, 0.4):
        L, T, D = geometry(eps)
        E_z, _, _ = moduli(eps)
        f = bands([np.pi / L], eps, n_modes=4)[0]
        print(f"  {eps:4.2f}   {L*1e3:5.2f}  {T*1e3:5.2f}  {E_z/1e3:8.1f}  {sigma_z(eps)/1e3:7.1f}   "
              + "  ".join(f"{v:7.1f}" for v in f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
