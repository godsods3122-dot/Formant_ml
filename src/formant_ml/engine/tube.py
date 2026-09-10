"""면적 함수 → 공진 — 성도를 극 목록이 아니라 **관**으로 본다.

왜 이것이 필요한가
------------------
지금 성도는 "독립 공명기들의 종속연결" 이다. 포먼트 주파수와 대역폭이 서로 무관한
자유 파라미터이고, 극은 8 개에서 잘린 뒤 그 위를 보정으로 때운다. 그 구조가
만드는 결함을 오늘 다섯 개 쟀다 (MEASUREMENTS §34~§36):

* `(2K−1)·c/(4L)` 에 구멍 — 이 화자에서 8990 Hz, 실측 −9.1~−9.8 dB
* `bw4` 가 2526 Hz 까지 감 — 어떤 기하도 그런 극을 안 낸다
* 포먼트 사이 골이 목표보다 1.25~2.12 dB 얕음 (손실이 기하와 무관하니까)
* 극 밀도에 아무 제약이 없음 (`c/(2L)` 는 면적함수와 무관한 절대 제약인데도)
* 곁가지가 덧붙인 필터라 "꺼짐" 상태를 따로 만들어야 하고, 거기서 버그가 났다

면적 함수에서 극을 계산하면 **다섯 개가 전부 구조적으로 사라진다.**

**소스-필터 이론은 그대로다.** 필터는 여전히 전달함수이고, 극을 자유롭게 두는
대신 기하에서 계산할 뿐이다. 오히려 필터가 "임의의 극 목록" 이 아니라 실제
성도가 되므로 이론에 더 충실해진다.

물리
----
웹스터 혼 방정식 (1 차원 관, 단면적 A(x)):

    ∂/∂x ( A ∂p/∂x ) + (ω/c)² A p = 0

경계조건이 **절대적**이다 — 조음과 무관하게 성립한다:

* 성문 쪽 (x=0): 거의 닫힘 → 강체벽 → ∂p/∂x = 0   (압력 배, 유량 마디)
* 입술 쪽 (x=L): 열림 → p = 0                      (압력 마디)

이산화하면 대칭 삼중대각 일반화 고유값 문제가 되고, 고윳값이 (ω/c)² 이다.
균일관이면 해석해 `f_n = (2n−1)c/(4L)` 를 정확히 준다 — 그것이 이 모듈의 첫 시험이다.
"""
from __future__ import annotations

import math

import torch

C_SOUND = 35000.0          # cm/s


# 물리 상수 — CGS, 체온(37 °C)의 습한 공기.
RHO = 1.14e-3          # g/cm^3
MU = 1.86e-4           # dyne*s/cm^2   전단 점성
LAMBDA_TH = 2.30e3     # erg/(cm*s*K)  열전도도
CP = 1.00e7            # erg/(g*K)     정압 비열
GAMMA = 1.4
# 성도 벽 (Flanagan 1972, 단위 면적당)
WALL_M = 1.5           # g/cm^2
WALL_R = 1.6e3         # dyne*s/cm^3
WALL_K = 4.0e5         # dyne/cm^3


def _perimeter(area: torch.Tensor) -> torch.Tensor:
    """단면 둘레. 원형 근사 S = 2*sqrt(pi*A).

    실제 성도는 납작해서 같은 면적에 둘레가 더 길다 (손실이 더 크다). 그 차이는
    `wall_shape` 인자로 흡수한다 — 원형이면 1.0, 납작할수록 크다.
    """
    return 2.0 * torch.sqrt(math.pi * area.clamp_min(1e-4))


def webster_modes(area: torch.Tensor, length_cm: float, n_modes: int | None = None,
                  c: float = C_SOUND):
    """면적 함수 -> (공진 주파수 [Hz], 압력 모드형).

    이산화 (셀 중심 격자, 2 차 정확도)
    ----------------------------------
    셀 i 의 중심은 x_i = (i+1/2)h, h = L/N. 셀 i 에서

        (1/h^2)[A_{i+1/2}(p_{i+1}-p_i) - A_{i-1/2}(p_i-p_{i-1})] + (w/c)^2 A_i p_i = 0

    * 성문 (x=0): 강체벽 노이만 -> A_{-1/2} = 0. 셀 중심 격자에서 정확하다.
    * 입술 (x=L): p(L) = 0 -> 유령셀 p_N = -p_{N-1}. 격자점이 아니라 **면**에서
      0 이 되므로 2 차를 유지한다. (앞선 1 차 판은 마지막 미지수를 그냥 버려서
      실효 길이가 h/2 짧아졌고, N=64 에서 0.8 % 오차를 냈다.)

    반격자 면적은 이웃의 기하평균이다 — 조음은 로그 면적에서 선형이라 산술평균보다
    정확하다.
    """
    a = area.clamp_min(1e-4)
    n = a.shape[-1]
    if n < 3:
        raise ValueError("면적 함수는 3 점 이상이어야 한다")
    h = float(length_cm) / n
    inner = torch.sqrt(a[..., :-1] * a[..., 1:])          # 면 1..N-1
    diag = torch.zeros_like(a)
    diag[..., 0] = inner[..., 0]                          # 왼쪽 면은 A_{-1/2}=0
    if n > 2:
        diag[..., 1:-1] = inner[..., :-1] + inner[..., 1:]
    diag[..., -1] = inner[..., -1] + 2.0 * a[..., -1]     # 유령셀 -> 계수 2
    diag = diag / (h * h)
    off = -inner / (h * h)
    s = torch.rsqrt(a)
    d2 = diag * s * s
    o2 = off * s[..., :-1] * s[..., 1:]
    K = torch.diag_embed(d2) + torch.diag_embed(o2, offset=1) + torch.diag_embed(o2, offset=-1)
    lam, psi = torch.linalg.eigh(K)
    lam = lam.clamp_min(0.0)
    f = c * torch.sqrt(lam) / (2.0 * math.pi)
    p = psi * s.unsqueeze(-1)                             # p = M^{-1/2} psi, (..., N, N)
    if n_modes is not None:
        f, p = f[..., :n_modes], p[..., :n_modes]
    return f, p


def webster_poles(area: torch.Tensor, length_cm: float, n_modes: int | None = None,
                  c: float = C_SOUND) -> torch.Tensor:
    """면적 함수 -> 공진 주파수 [Hz]. (..., N) -> (..., n_modes)"""
    return webster_modes(area, length_cm, n_modes, c)[0]


def loss_bandwidths(area: torch.Tensor, length_cm: float, f: torch.Tensor,
                    p: torch.Tensor, c: float = C_SOUND, wall_shape: float = 1.0,
                    wall_scale: float = 1.0, rad_scale: float = 1.0) -> torch.Tensor:
    """모드별 대역폭 [Hz] — 손실을 **계산**한다. 자유 파라미터가 아니다.

    섭동론. 무손실 모드형 p(x) 와 그에 딸린 체적유량 u = -(A/(j w rho)) dp/dx 에
    손실 항을 얹으면

        B = P_loss / (2 pi W_tot),   W_tot = 2 * W_kin

    세 손실은 주파수 의존성이 서로 달라서, 셋이 합쳐진 B(f) 곡선의 **모양**이
    관측된 포먼트 대역폭을 재현하는지가 이 모델의 시험이다.

    1. 점성 경계층 — 직렬 저항 R_v = (S/A^2) sqrt(rho mu w / 2),  ~ sqrt(f)
    2. 열전도 경계층 — 병렬 컨덕턴스 G_t = (S/(rho c^2)) (gamma-1) sqrt(lambda w/(2 rho cp)), ~ sqrt(f)
    3. 벽 진동 — 병렬 어드미턴스 S/z_w, z_w = R + j(w m - k/w). 저주파에서만 크다 (~1/f^2)
    4. 입술 복사 — R_rad = rho c (ka)^2/(2A),  ~ f^2. 입술에서만.
    """
    a = area.clamp_min(1e-4)
    n = a.shape[-1]
    h = float(length_cm) / n
    w = 2.0 * math.pi * f.clamp_min(1.0)                      # (..., M)
    S = _perimeter(a) * wall_shape                            # (..., N)
    # 모드형을 모드축이 앞에 오도록 (..., M, N) 로 옮긴다
    pm = p.transpose(-1, -2)
    # u ∝ -A dp/dx.  면에서의 차분 -> 면 위의 유량, 면 면적은 기하평균.
    inner = torch.sqrt(a[..., :-1] * a[..., 1:])
    du = (pm[..., 1:] - pm[..., :-1]) / h
    u_face = -inner.unsqueeze(-2) * du                        # (..., M, N-1)  (1/(j w rho) 는 아래서)
    # 입술 면: 유령셀 p_N = -p_{N-1}
    u_lip = -a[..., -1:].unsqueeze(-2) * (-2.0 * pm[..., -1:]) / h
    u_all = torch.cat([torch.zeros_like(u_face[..., :1]), u_face, u_lip], dim=-1)  # 면 0..N
    # 셀 중심 유량 (면 평균) — 저항 적분용
    u_cell = 0.5 * (u_all[..., :-1] + u_all[..., 1:])
    k1 = 1.0 / (w * RHO).unsqueeze(-1)                        # 1/(w rho),  (..., M, 1)
    U = u_cell * k1
    U_lip = u_all[..., -1:] * k1
    # --- 저장 에너지 --------------------------------------------------------
    w_kin = 0.25 * (RHO / a).unsqueeze(-2) * U * U
    W = 2.0 * w_kin.sum(-1) * h                               # (..., M)
    # --- 손실 ---------------------------------------------------------------
    ww = w.unsqueeze(-1)
    Sb, ab = S.unsqueeze(-2), a.unsqueeze(-2)
    R_v = (Sb / (ab * ab)) * torch.sqrt(RHO * MU * ww / 2.0)
    G_t = (Sb / (RHO * c * c)) * (GAMMA - 1.0) * torch.sqrt(LAMBDA_TH * ww / (2.0 * RHO * CP))
    zr = WALL_R
    zi = ww * WALL_M - WALL_K / ww
    G_w = wall_scale * Sb * zr / (zr * zr + zi * zi)
    a_lip = a[..., -1:].unsqueeze(-2)
    r_lip = torch.sqrt(a_lip / math.pi)
    ka = ww * r_lip / c
    # 배플 원형 피스톤의 복사 저항 R/(rho c/A) = 1 - 2 J1(2ka)/(2ka).
    # x^2/(8+x^2), x=2ka 는 그 유리 근사다 — 저주파에서 (ka)^2/2 로 맞고, ka>>1 에서
    # rho c/A 로 포화한다. (ka)^2/2 만 쓰면 포화가 없어 5~6 번 포먼트 대역폭이
    # 문헌의 1.4~1.9 배로 벌어진다. 미분 가능하다.
    ka2 = ka * ka
    R_rad = rad_scale * (RHO * c / a_lip) * ka2 / (2.0 + ka2)
    P = 0.5 * ((R_v * U * U).sum(-1) * h
               + ((G_t + G_w) * pm * pm).sum(-1) * h
               + (R_rad * U_lip * U_lip).sum(-1))
    return P / (2.0 * math.pi * W.clamp_min(1e-30))


def uniform_area(n: int, value: float = 3.0, **kw) -> torch.Tensor:
    """균일관 면적 함수 — 해석해와 대조할 때 쓴다."""
    return torch.full((n,), float(value), **kw)
