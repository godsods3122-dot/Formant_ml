"""조음 모형 — Maeda 선형 조음 모형(LAM)을 미분 가능하게 (MEASUREMENTS §52.485).

사용자: *"조음에 대한 물리 모델이 아예 없어? 나는 중규모 소프트웨어라는 생각으로 진행하고 있었는데, 그냥 지금 toy model하나 만들어놓고
장난치려고 하는 건 아니지?"*

예전 조음은 틀마다 자유로운 코사인 계수 10 개로 로그 면적 곡선을 맞추고 가우스 협착 하나를 얹은 **곡선 맞춤**이었다 (`tube.cos_area`).
조음기가 없어서 적합기는 불가능한 관 모양(ㅅ 에서 후두 칸을 0.07 cm² 로 조이기)과 주기별 쫓기로 오차를 메웠다.

여기서는 관 모양을 **조음기의 자리**에서 낸다 — Maeda (1990) 의 X선 영화 자료(화자 PB)에서 요인 분석으로 뽑은 일곱 변수:

    p0 턱          p1 혀 몸통 위치   p2 혀 몸통 모양   p3 혀끝
    p4 입술 높이   p5 입술 돌출      p6 후두 높이                (단위: 표준편차, 대개 ±3)

변수 → 요인 적재 → 반극좌표 격자 위의 혀·입술·후두 윤곽 (벽 윤곽은 고정) → 격자 칸마다 중시상 폭 w → 면적 A = 1.4·α·w^β
(α–β 변환, 격자마다 다른 계수) → 입술관은 타원 (높이 × 너비). 원래 구현(sensein/VocalTractModels `vtcalcs/src/lam_lib.c`,
Apache 2.0, `profiles/MAEDA_NOTICE.txt`)과 같은 식이고, 미분이 끊기는 곳만 바꿨다:

* 혀를 벽에서 막는 `min(혀, 벽)` → **벽을 넘지 않는** 부드러운 최소 `벽 − τ·softplus((벽 − 혀)/τ)` (닫힘 = 면적 → 0 이 된다).
* 입술 치수를 0 에서 막는 `max(·, 0)` → `τ·softplus(·/τ)`.
* 칸의 사각형 넓이는 두 삼각형의 Heron 공식 대신 신발끈 공식 (볼록 사각형에서 같고, 넓이 0 에서도 기울기가 유한하다).

원 화자는 성인 남성(중립 자세 성도 16.3 cm)이다. 화자 적응은 **화자 상수**로 한다 — 인두·구강 길이 배율과 단면 배율 (`SpeakerScale`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SPEC_PATH = Path(__file__).resolve().parents[3] / "profiles" / "maeda_pb1_spec.dat"
N_PAR = 7
PAR_NAMES = ("jaw", "tongue_pos", "tongue_shape", "tongue_tip", "lip_ht", "lip_pr", "larynx")
TAU_WALL = 0.02     # 벽 막기의 부드러움 [격자 단위 ≈ 0.01 cm]
TAU_LIP = 0.02      # 입술 치수 0 막기의 부드러움
INCI_LIP_CM = 0.8   # 윗니와 윗입술 사이 [cm] (원래 구현의 상수)
SIZE_CORRECTION = 1.10
AREA_BOOST = 1.4    # 원래 구현의 "40% ad hoc increase"


class _Reader:
    """C 의 fscanf/skiplines 흉내 — 원래 파서와 같은 순서로 읽는다."""

    def __init__(self, text: str):
        self.s, self.i = text, 0

    def skiplines(self, n: int) -> None:
        for _ in range(n):
            j = self.s.index("\n", self.i)
            self.i = j + 1

    def _ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def tok(self) -> str:
        self._ws()
        j = self.i
        while j < len(self.s) and not self.s[j].isspace():
            j += 1
        t, self.i = self.s[self.i:j], j
        self._ws()                               # 형식 끝의 "\n" 은 공백을 모두 먹는다
        return t

    def f(self) -> float:
        return float(self.tok())

    def d(self) -> int:
        return int(float(self.tok()))


@dataclass
class MaedaSpec:
    m1: int; m2: int; m3: int; dl: float; omega: float; theta: float; ix0: float; iy0: float
    TEKvt: float; TEKlip: float
    alph: np.ndarray; beta: np.ndarray
    u_lip: np.ndarray; s_lip: np.ndarray; A_lip: np.ndarray; inci_x: float; inci_y: float
    u_tng: np.ndarray; s_tng: np.ndarray; A_tng: np.ndarray; iniva_tng: int; lstva_tng: int
    u_lrx: np.ndarray; s_lrx: np.ndarray; A_lrx: np.ndarray
    u_wal: np.ndarray


def read_spec(path: Path = SPEC_PATH) -> MaedaSpec:
    """`read_model_spec` 을 그대로 옮긴 것 (줄·토큰 순서가 같다)."""
    r = _Reader(path.read_text(encoding="latin-1"))
    r.skiplines(9)
    m1, m2, m3 = r.d(), r.d(), r.d()
    dl, omega, theta = r.f(), r.f(), r.f()
    ix0, iy0 = r.d(), r.d()
    r.skiplines(1)
    TEKvt, TEKlip = r.f(), r.f()
    r.skiplines(1)
    M4 = m1 + m2 + m3
    alph, beta = np.zeros(M4), np.zeros(M4)
    for i in range(M4):
        r.d(); alph[i] = r.f(); beta[i] = r.f()

    def block(njaw_plus: int, with_scores: bool = True):
        r.skiplines(2)
        nv, jaw, ini, lst, nfs, nafs = (r.d() for _ in range(6))
        r.skiplines(nafs + 2)
        for _ in range(nv):
            r.tok()
        return nv, jaw, ini, lst, nfs, nafs

    # 입술
    nv, jaw, _, _, nfs, nafs = block(3)
    r.skiplines(1)
    for _ in range(3):
        r.tok()
    r.skiplines(2)
    u_lip = np.array([r.f() for _ in range(nv)])
    r.skiplines(1)
    s_lip = np.array([r.f() for _ in range(nv)])
    r.skiplines(1)
    inci_x, inci_y = r.f(), r.f()
    r.skiplines(3)
    A_lip = np.zeros((nv, 3))
    for i in range(nv):
        for j in range(3):
            A_lip[i, j] = r.f()
        r.skiplines(1)
    r.skiplines(1 + nv)
    # 혀
    nv, jaw, ini, lst, nfs, nafs = block(4)
    r.skiplines(1)
    for _ in range(4):
        r.tok()
    r.skiplines(2)
    u_tng = np.array([r.f() for _ in range(nv)])
    r.skiplines(1)
    s_tng = np.array([r.f() for _ in range(nv)])
    r.skiplines(1)
    A_tng = np.zeros((nv, 4))
    for i in range(nv):
        for j in range(4):
            A_tng[i, j] = r.f()
        r.skiplines(1)
    r.skiplines(1 + nv)
    iniva_tng, lstva_tng = ini - 1, lst - 1
    # 후두
    nv, jaw, _, _, nfs, nafs = block(2)
    r.skiplines(1)
    for _ in range(2):
        r.tok()
    r.skiplines(2)
    u_lrx = np.array([r.f() for _ in range(nv)])
    r.skiplines(1)
    s_lrx = np.array([r.f() for _ in range(nv)])
    r.skiplines(1)
    A_lrx = np.zeros((nv, 2))
    for i in range(nv):
        for j in range(2):
            A_lrx[i, j] = r.f()
        r.skiplines(1)
    r.skiplines(1 + nv)
    # 벽
    nv, jaw, ini_w, lst_w, nfs, nafs = block(0)
    r.skiplines(3)
    u_wal = np.array([r.f() for _ in range(nv)])
    return MaedaSpec(m1, m2, m3, dl, omega, theta, float(ix0), float(iy0), TEKvt, TEKlip, alph, beta,
                     u_lip, s_lip, A_lip, inci_x, inci_y, u_tng, s_tng, A_tng, iniva_tng, lstva_tng,
                     u_lrx, s_lrx, A_lrx, u_wal)


def _convert_and_grid(sp: MaedaSpec):
    """`convert_scale` + `semi_polar` — TEK 단위를 뷰포트 단위로, 격자 끝점과 방향 벡터."""
    vp_map = 10.0 / (200 / 10)                    # vp_width_cm / (DWIDTH/10) = 0.5 cm / 단위
    TEKvt = sp.TEKvt * vp_map
    TEKlip = TEKvt if sp.TEKlip == 0.0 else sp.TEKlip * vp_map
    inci_x = (sp.inci_x - sp.ix0) / TEKvt
    inci_y = (sp.inci_y - sp.iy0) / TEKvt
    u_lip, s_lip = sp.u_lip.copy(), sp.s_lip.copy()
    u_lip[:2] /= TEKvt; s_lip[:2] /= TEKvt
    u_lip[2:] /= TEKlip; s_lip[2:] /= TEKlip
    u_tng, s_tng = sp.u_tng / TEKvt, sp.s_tng / TEKvt
    u_lrx, s_lrx = sp.u_lrx / TEKvt, sp.s_lrx / TEKvt
    u_wal = sp.u_wal / TEKvt
    ix0 = iy0 = 0.6 * (200 / 10)
    # 반극좌표 (float 로 C 와 같게)
    r_vp, dl_vp = 5.0 / vp_map, sp.dl / vp_map
    pi = 3.14159265
    ome, the = pi * sp.omega / 180.0, pi * sp.theta / 180.0
    M4 = sp.m1 + sp.m2 + sp.m3
    igd, egd = np.zeros((M4, 2)), np.zeros((M4, 2))
    dx_i, dy_i = dl_vp * math.cos(ome - pi / 2.0), dl_vp * math.sin(ome - pi / 2.0)
    dx_e, dy_e = r_vp * math.cos(ome), r_vp * math.sin(ome)
    for i in range(sp.m1):
        igd[i] = (dx_i * (sp.m1 - (i + 1)) + ix0, dy_i * (sp.m1 - (i + 1)) + iy0)
        egd[i] = (dx_e + igd[i, 0], dy_e + igd[i, 1])
    gam = ome
    for i in range(sp.m1, sp.m1 + sp.m2):
        gam = the * (i + 1 - sp.m1) + ome
        igd[i] = (ix0, iy0)
        egd[i] = (r_vp * math.cos(gam) + ix0, r_vp * math.sin(gam) + iy0)
    dx_i, dy_i = dl_vp * math.cos(gam + pi / 2.0), dl_vp * math.sin(gam + pi / 2.0)
    dx_e, dy_e = r_vp * math.cos(gam), r_vp * math.sin(gam)
    for i in range(sp.m1 + sp.m2, M4):
        igd[i] = (dx_i * (i + 1 - sp.m1 - sp.m2) + ix0, dy_i * (i + 1 - sp.m1 - sp.m2) + iy0)
        egd[i] = (dx_e + igd[i, 0], dy_e + igd[i, 1])
    d = egd - igd
    vtos = d / np.linalg.norm(d, axis=1, keepdims=True)
    return dict(vp_map=vp_map, inci_x=inci_x, inci_y=inci_y, u_lip=u_lip, s_lip=s_lip, u_tng=u_tng, s_tng=s_tng,
                u_lrx=u_lrx, s_lrx=s_lrx, u_wal=u_wal, ix0=ix0, iy0=iy0, igd=igd, vtos=vtos,
                inci_lip_vp=INCI_LIP_CM / vp_map)


class MaedaLAM(torch.nn.Module):
    """조음 변수 (…, 7) → 원래 칸 (면적 (…, 29) [cm²], 칸 길이 (…, 29) [cm]). 식은 `lam` + `sagittal_to_area`."""

    def __init__(self, path: Path = SPEC_PATH):
        super().__init__()
        sp = read_spec(path)
        g = _convert_and_grid(sp)
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))
        self.register_buffer("A_tng", t(sp.A_tng)); self.register_buffer("s_tng", t(g["s_tng"]))
        self.register_buffer("u_tng", t(g["u_tng"]))
        self.register_buffer("A_lip", t(sp.A_lip)); self.register_buffer("s_lip", t(g["s_lip"]))
        self.register_buffer("u_lip", t(g["u_lip"]))
        self.register_buffer("A_lrx", t(sp.A_lrx)); self.register_buffer("s_lrx", t(g["s_lrx"]))
        self.register_buffer("u_lrx", t(g["u_lrx"]))
        self.register_buffer("u_wal", t(g["u_wal"]))
        ini, lst = sp.iniva_tng, sp.lstva_tng
        self.register_buffer("igd", t(g["igd"][ini:lst + 1])); self.register_buffer("vtos", t(g["vtos"][ini:lst + 1]))
        idx = np.arange(1, 28) + ini - 3                 # sagittal_to_area 의 j = i + iniva − 3 (i = 1..np−2)
        self.register_buffer("alph", t(sp.alph[idx])); self.register_buffer("beta", t(sp.beta[idx]))
        self.ix0, self.iy0 = float(g["ix0"]), float(g["iy0"])
        self.inci = (float(g["inci_x"]) + self.ix0, float(g["inci_y"]) + float(g["inci_lip_vp"]) + self.iy0)
        self.c = SIZE_CORRECTION * g["vp_map"]          # 뷰포트 단위 → cm
        self.n_ph = int(np.sum(idx < sp.m1))             # 인두(선형 격자) 칸 수 — 화자 배율이 쓴다

    def contours(self, p: torch.Tensor):
        """(…, 7) → 안쪽·바깥 윤곽 (…, 29, 2) [뷰포트 단위]."""
        p = p.to(self.A_tng.dtype)
        jaw = p[..., :1]
        v_tng = (torch.cat([jaw, p[..., 1:4]], -1) @ self.A_tng.T) * self.s_tng + self.u_tng
        v_lip = (torch.cat([jaw, p[..., 4:6]], -1) @ self.A_lip.T) * self.s_lip + self.u_lip
        v_lip = TAU_LIP * torch.nn.functional.softplus(v_lip / TAU_LIP)
        v_lrx = (torch.cat([jaw, p[..., 6:7]], -1) @ self.A_lrx.T) * self.s_lrx + self.u_lrx
        wal = self.u_wal
        v = wal - TAU_WALL * torch.nn.functional.softplus((wal - v_tng[..., 1:]) / TAU_WALL)
        inner = self.vtos * v.unsqueeze(-1) + self.igd                  # (…, 25, 2)
        outer = (self.vtos * wal.unsqueeze(-1) + self.igd).expand_as(inner)
        l_in = torch.stack([v_lrx[..., 1] + self.ix0, v_lrx[..., 2] + self.iy0], -1).unsqueeze(-2)
        l_out = torch.stack([v_lrx[..., 3] + self.ix0, v_lrx[..., 4] + self.iy0], -1).unsqueeze(-2)
        mid_in, mid_out = 0.5 * (l_in + inner[..., :1, :]), 0.5 * (l_out + outer[..., :1, :])
        ex, ey = self.inci
        e1 = torch.tensor([ex, ey], dtype=p.dtype, device=p.device).expand_as(l_in)
        i1 = torch.stack([e1[..., 0], e1[..., 1] - v_lip[..., 2:3]], -1)
        e2 = torch.stack([e1[..., 0] - v_lip[..., 1:2], e1[..., 1]], -1)
        i2 = torch.stack([e2[..., 0], i1[..., 1]], -1)
        ivt = torch.cat([l_in, mid_in, inner, i1, i2], -2)
        evt = torch.cat([l_out, mid_out, outer, e1, e2], -2)
        return ivt, evt, v_lip

    def forward(self, p: torch.Tensor):
        ivt, evt, v_lip = self.contours(p)
        a, b, cc, dd = ivt[..., :-2, :], ivt[..., 1:-1, :], evt[..., 1:-1, :], evt[..., :-2, :]   # 칸 i = 1..27 의 네 꼭짓점
        cross = lambda u, w: u[..., 0] * w[..., 1] - u[..., 1] * w[..., 0]
        quad = 0.5 * torch.abs(cross(a, b) + cross(b, cc) + cross(cc, dd) + cross(dd, a))        # 신발끈
        mv = (a + dd - b - cc)
        d = 0.5 * torch.sqrt((mv * mv).sum(-1) + 1e-12)
        w = self.c * quad / d
        A = AREA_BOOST * self.alph * w.clamp_min(1e-9) ** self.beta
        x = self.c * d
        lip_h, lip_w = 0.5 * v_lip[..., 2], 0.5 * v_lip[..., 3]
        A_lip = (math.pi * lip_h * lip_w * self.c * self.c).unsqueeze(-1)
        x_lip = (0.5 * v_lip[..., 1] * self.c).unsqueeze(-1)
        A = torch.cat([A, A_lip, A_lip], -1).clamp_min(1e-4)
        x = torch.cat([x, x_lip, x_lip], -1).clamp_min(0.01)
        return A, x


@dataclass
class SpeakerScale:
    """원 화자(PB, 남성) → 이 화자. 길이 배율은 인두·구강을 따로 (여성은 인두가 더 짧다, Fant 1966), 단면 배율 하나.
    기본값은 yang 코퍼스 20 파일의 유성 틀 F1–F2 (Praat) 를 조음 공간이 덮는 격자 탐색에서 (§52.485): 중앙 거리 ~1 %, 90 % 8.8 %,
    중립 자세 성도 ≈ 13 cm. F3 까지 넣으면 모형의 F3 가 이 화자보다 낮아 90 % 가 25 % 로 벌어진다 — 적합에서 지켜본다."""
    pharynx: float = 0.76
    oral: float = 0.80
    area: float = 0.80


def uniform_tube(A: torch.Tensor, x: torch.Tensor, n: int, n_ph: int | None = None,
                 scale=None) -> tuple[torch.Tensor, torch.Tensor]:
    """가변 길이 칸 → 균일 n 칸 (`appro_area_function` 의 부피 보존 평균). (면적 (…, n), 전체 길이 (…,)).

    scale: `SpeakerScale` 또는 (인두, 구강, 단면) 배율 — 0 차원 텐서면 기울기가 흐른다 (화자 상수로 적합).
    누적 부피 V(z) 를 칸 경계에서 선형 보간해 균일 칸의 평균 면적 = ΔV/Δz. 입술 끝 칸은 원래 입술 면적을 그대로 쓴다
    (원래 구현과 같다 — 평균하면 [u] 의 입술이 넓어져 F2 가 높아진다)."""
    if scale is not None:
        ph, orl, ar = (scale.pharynx, scale.oral, scale.area) if isinstance(scale, SpeakerScale) else scale
        x = torch.cat([x[..., :n_ph] * ph, x[..., n_ph:] * orl], -1)
        A = A * ar
    z = torch.cat([torch.zeros_like(x[..., :1]), torch.cumsum(x, -1)], -1)          # 칸 경계 (…, m+1)
    V = torch.cat([torch.zeros_like(x[..., :1]), torch.cumsum(A * x, -1)], -1)
    L = z[..., -1]
    q = L.unsqueeze(-1) * torch.linspace(0.0, 1.0, n + 1, dtype=A.dtype, device=A.device)  # 균일 경계
    j = torch.searchsorted(z.contiguous(), q.contiguous()).clamp(1, z.shape[-1] - 1)
    z0, z1 = torch.gather(z, -1, j - 1), torch.gather(z, -1, j)
    V0, V1 = torch.gather(V, -1, j - 1), torch.gather(V, -1, j)
    Vq = V0 + (V1 - V0) * (q - z0) / (z1 - z0).clamp_min(1e-9)
    Au = (Vq[..., 1:] - Vq[..., :-1]) / (L / n).unsqueeze(-1)
    Au = torch.cat([Au[..., :-1], A[..., -1:]], -1)
    return Au.clamp_min(1e-4), L


def codebook(n: int = 30000, scale=None, seed: int = 0, n_cells: int = 28, lo: float = -2.5, hi: float = 2.5):
    """조음 공간 코드북 — 무작위 자세 n 개의 (변수 (n, 7), F1–F3 (n, 3) [Hz], 최소 면적 (n,), 성도 길이 (n,)).

    포먼트는 웹스터 고유모드 (입술 압력 0, 무손실 — 역산의 순위용이라 끝 보정·손실은 넣지 않는다)."""
    from .tube import webster_modes
    rng = np.random.default_rng(seed)
    P = rng.uniform(lo, hi, (n, N_PAR))
    m = MaedaLAM()
    F, amin, Ls = [], [], []
    with torch.no_grad():
        for s in range(0, n, 2000):
            A, x = m(torch.as_tensor(P[s:s + 2000]))
            Au, L = uniform_tube(A, x, n_cells, m.n_ph, scale)
            f, _ = webster_modes(Au, 1.0, 3)
            F.append((f / L.unsqueeze(-1)).numpy()); amin.append(Au.min(-1).values.numpy()); Ls.append(L.numpy())
    return P, np.concatenate(F), np.concatenate(amin), np.concatenate(Ls)


def invert_formants(fobs: np.ndarray, weight: np.ndarray, scale=None, n_code: int = 30000, top: int = 40,
                    smooth: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """분석 포먼트 (T, 3) [Hz] → 조음 변수 (T, 7) — 코드북 후보 + 비터비 (조음기가 매끄럽게 움직이는 경로).

    틀 비용 = weight_t · Σ_k (log F_k^코드북 − log F_k^관측)² (관측이 없는 포먼트는 뺀다), 옮김 비용 = smooth · ‖Δp‖² / 틀.
    관측이 없거나 weight 0 인 틀(무성·폐쇄)은 옮김 비용만 — 이웃 자세를 잇는다. 모음 틀의 후보는 막힌 자세(최소 면적 < 0.15 cm²)를 뺀다.
    반환: (변수 (T, 7), 틀별 로그 포먼트 오차의 rms)."""
    P, F, amin, _ = codebook(n_code, scale, seed)
    ok_cb = amin > 0.15
    P, F = P[ok_cb], F[ok_cb]
    lF = np.log(np.maximum(F, 1.0))
    T = fobs.shape[0]
    lo = np.log(np.maximum(fobs, 1.0))
    has = (fobs > 50.0) & np.isfinite(fobs)
    cand = np.zeros((T, top), dtype=np.int64)
    cost = np.zeros((T, top))
    for t in range(T):
        if weight[t] <= 0 or not has[t].any():
            cand[t] = cand[t - 1] if t else np.arange(top)
            cost[t] = 0.0
            continue
        d = (((lF - lo[t]) ** 2) * has[t]).sum(1)
        k = np.argpartition(d, top)[:top]
        cand[t], cost[t] = k, weight[t] * d[k]
    # 비터비
    acc = cost[0].copy()
    back = np.zeros((T, top), dtype=np.int64)
    for t in range(1, T):
        dp = ((P[cand[t]][:, None, :] - P[cand[t - 1]][None, :, :]) ** 2).sum(-1)     # (top, top)
        tot = acc[None, :] + smooth * dp
        back[t] = tot.argmin(1)
        acc = tot.min(1) + cost[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(acc.argmin())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    idx = cand[np.arange(T), path]
    err = np.sqrt((((lF[idx] - lo) ** 2) * has).sum(1) / np.maximum(has.sum(1), 1))
    return P[idx], err


W02_PATH = Path(__file__).resolve().parents[3] / "profiles" / "w02_surrogate.pt"
W02_N_CELLS = 28
AREA_EPS = 0.01     # 대리 모형의 면적 표현 log(A + ε) [cm²] — 폐쇄 근처에서 log A 가 −9 로 떨어져 학습이 흔들렸다 (§52.486)
#: 대리 모형 아래쪽 끝의 치우침 보정 [cm²]. VTL 이 면적 0 으로 내는 폐쇄를 대리 모형은 중앙 0.0006, 90 % 0.009 cm² 로 내 파열음 폐쇄가 샜다.
#: 0.004 를 빼면 폐쇄 90 % 0.005, 좁은 협착(0.02–0.3 cm², /s/ 등)은 대리/원본 비 중앙 0.99 → 0.965 (학습에 안 쓴 3000 자세에서 VTL 과 대조).
AREA_BIAS = 0.004


class W02Surrogate(torch.nn.Module):
    """VocalTractLab 여성 화자 W02 의 기하 (해부학 변수 19 → 균일 28 칸 면적·성도 길이·연구개 열림) 를 흉내 내는 MLP (§52.486).

    VTL 은 C++ 라 기울기가 없다 — `scripts/build_w02_surrogate.py` 가 VTL 로 뽑은 자세로 학습하고, 떼어 둔 자세에서 원본과 잰다.
    입력은 변수 범위로 [−1, 1] 정규화, 출력은 (log(면적 + AREA_EPS) ×28, log 길이, log(연구개 열림 + 1e−4))."""

    def __init__(self, lo, hi, names, width: int = 768):
        super().__init__()
        self.names = list(names)
        self.register_buffer("lo", torch.as_tensor(np.asarray(lo), dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(np.asarray(hi), dtype=torch.float32))
        n = len(self.names)
        self.net = torch.nn.Sequential(torch.nn.Linear(n, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, W02_N_CELLS + 2))

    def raw(self, p: torch.Tensor) -> torch.Tensor:
        x = 2.0 * (p.float() - self.lo) / (self.hi - self.lo) - 1.0
        return self.net(x)

    def forward(self, p: torch.Tensor):
        """(…, 19) → (면적 (…, 28) [cm²], 성도 길이 (…,) [cm], 연구개 열림 (…,) [cm²])."""
        y = self.raw(p)
        A = torch.nn.functional.softplus(torch.exp(y[..., :W02_N_CELLS]) - AREA_EPS - AREA_BIAS, beta=2000.0) + 1e-4
        return A, torch.exp(y[..., W02_N_CELLS]), (torch.exp(y[..., W02_N_CELLS + 1]) - 1e-4).clamp_min(0.0)

    @classmethod
    def load(cls, path: Path = W02_PATH) -> "W02Surrogate":
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(d["lo"], d["hi"], d["names"])
        m.load_state_dict(d["state"])
        m.eval()
        return m


W02_ANAT_PATH = Path(__file__).resolve().parents[3] / "profiles" / "w02_anat_surrogate.pt"


class W02AnatSurrogate(W02Surrogate):
    """해부 적응까지 받는 W02 기하 대리 모형 (§52.501) — (조음 19 [W02 단위: 그 해부의 범위로 정규화한 위치], 해부 13 [cm·도]) →
    (면적 28, 성도 길이, 연구개 열림). 해부는 VTL `AnatomyParams` 차례 (`vtl_api.ANAT_NAMES`), 입력은 학습 범위로 [−1, 1] 정규화."""

    def __init__(self, lo, hi, names, alo, ahi, anames, ax0=None, width: int = 1024):
        super().__init__(lo, hi, names, width)
        self.anames = list(anames)
        self.register_buffer("alo", torch.as_tensor(np.asarray(alo), dtype=torch.float32))
        self.register_buffer("ahi", torch.as_tensor(np.asarray(ahi), dtype=torch.float32))
        self.register_buffer("ax0", torch.as_tensor(np.asarray(ax0 if ax0 is not None else 0.5 * (np.asarray(alo) + np.asarray(ahi))),
                                                    dtype=torch.float32))
        n, na = len(self.names), len(self.anames)
        self.net = torch.nn.Sequential(torch.nn.Linear(n + na, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, W02_N_CELLS + 2))
        self.width = width

    def raw(self, p: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        if a is None:
            a = self.ax0.expand(*p.shape[:-1], -1)
        x = torch.cat([2.0 * (p.float() - self.lo) / (self.hi - self.lo) - 1.0,
                       2.0 * (a.float() - self.alo) / (self.ahi - self.alo) - 1.0], -1)
        return self.net(x)

    def forward(self, p: torch.Tensor, a: torch.Tensor | None = None):
        y = self.raw(p, a)
        A = torch.nn.functional.softplus(torch.exp(y[..., :W02_N_CELLS]) - AREA_EPS - AREA_BIAS, beta=2000.0) + 1e-4
        return A, torch.exp(y[..., W02_N_CELLS]), (torch.exp(y[..., W02_N_CELLS + 1]) - 1e-4).clamp_min(0.0)

    def pack(self) -> dict:
        return {"state": self.state_dict(), "lo": self.lo.numpy(), "hi": self.hi.numpy(), "names": self.names,
                "alo": self.alo.numpy(), "ahi": self.ahi.numpy(), "anames": self.anames, "ax0": self.ax0.numpy(), "width": self.width}

    @classmethod
    def load(cls, path: Path = W02_ANAT_PATH) -> "W02AnatSurrogate":
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(d["lo"], d["hi"], d["names"], d["alo"], d["ahi"], d["anames"], d.get("ax0"), d.get("width", 1024))
        m.load_state_dict(d["state"])
        m.eval()
        return m


W02_TEETH_PATH = Path(__file__).resolve().parents[3] / "profiles" / "w02_teeth.pt"


class W02TeethNet(torch.nn.Module):
    """W02 (해부 적응) 의 **입술–앞니 거리** [cm] — (조음 19, 해부 13) → 성도 길이 − 앞니 자리 (VTL `vtlTractToTube` 의 incisorPos, §52.519).
    협착 난류의 음원 모양을 정하는 앞니 장애물의 자리 (`tube_td.NOISE_PLACE`). `scripts/build_w02_teeth.py` 가 학습한다."""

    def __init__(self, lo, hi, names, alo, ahi, width: int = 256):
        super().__init__()
        self.names = list(names)
        for k, v in (("lo", lo), ("hi", hi), ("alo", alo), ("ahi", ahi)):
            self.register_buffer(k, torch.as_tensor(np.asarray(v), dtype=torch.float32))
        n = len(self.names) + len(alo)
        self.net = torch.nn.Sequential(torch.nn.Linear(n, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 1))
        self.width = width

    def forward(self, p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([2.0 * (p.float() - self.lo) / (self.hi - self.lo) - 1.0,
                       2.0 * (a.float() - self.alo) / (self.ahi - self.alo) - 1.0], -1)
        return self.net(x)[..., 0]

    def pack(self) -> dict:
        return {"state": self.state_dict(), "lo": self.lo.numpy(), "hi": self.hi.numpy(), "names": self.names,
                "alo": self.alo.numpy(), "ahi": self.ahi.numpy(), "width": self.width}

    @classmethod
    def load(cls, path: Path = W02_TEETH_PATH) -> "W02TeethNet":
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(d["lo"], d["hi"], d["names"], d["alo"], d["ahi"], d.get("width", 256))
        m.load_state_dict(d["state"])
        m.eval()
        return m


W02_SHAPES = Path(__file__).resolve().parents[3] / "profiles" / "w02_shapes.npz"
W02_FIT = ("HX", "HY", "JX", "JA", "LP", "LD", "TCX", "TCY", "TTX", "TTY", "TBX", "TBY", "TRX", "TRY", "TS1", "TS2", "TS3")  # VS·VO 는 적합하지 않는다


def invert_formants_w02(fobs: np.ndarray, weight: np.ndarray, n_code: int = 40000, top: int = 40, smooth: float = 0.3,
                        seed: int = 0, len_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """분석 포먼트 (T, 3) → W02 변수 (T, 19) — W02 의 MRI 음소 자세를 둘씩 섞고 흔든 코드북 + 비터비 (`invert_formants` 와 같은 틀).
    포먼트는 대리 모형 면적의 웹스터 고유모드 (방사 끝 보정 0.85 r). 반환: (변수, 틀별 로그 오차 rms, 변수 이름)."""
    from .tube import webster_modes
    z = np.load(W02_SHAPES, allow_pickle=True)
    S, names, lo, hi = z["shapes"], list(z["param_names"]), z["lo"], z["hi"]
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(S), n_code), rng.integers(0, len(S), n_code)
    w = rng.uniform(0, 1, (n_code, 1))
    P = (1 - w) * S[i] + w * S[j] + rng.normal(0, 0.05, (n_code, len(names))) * (hi - lo)
    fixed = [names.index(n) for n in names if n not in W02_FIT]
    P[:, fixed] = np.asarray(z["neutral"])[fixed]
    P = np.clip(P, lo, hi)
    sur = W02Surrogate.load()
    with torch.no_grad():
        A, L, _ = sur(torch.as_tensor(P, dtype=torch.float32))
        A, L = A.double(), L.double() * len_scale
        f, _ = webster_modes(A, 1.0, 3)
        F = (f / (L + 0.85 * torch.sqrt(A[:, -1] / math.pi)).unsqueeze(-1)).numpy()
        ok = (A.min(-1).values > 0.15).numpy()
    P, F = P[ok], F[ok]
    lF = np.log(np.maximum(F, 1.0))
    T = fobs.shape[0]
    lo_ = np.log(np.maximum(fobs, 1.0))
    has = (fobs > 50.0) & np.isfinite(fobs)
    span = (hi - lo)
    Pn = P / span                                            # 옮김 비용은 범위로 정규화한 변수 공간에서
    cand = np.zeros((T, top), dtype=np.int64); cost = np.zeros((T, top))
    for t in range(T):
        if weight[t] <= 0 or not has[t].any():
            cand[t] = cand[t - 1] if t else np.arange(top); continue
        d = (((lF - lo_[t]) ** 2) * has[t]).sum(1)
        k = np.argpartition(d, top)[:top]
        cand[t], cost[t] = k, weight[t] * d[k]
    acc = cost[0].copy(); back = np.zeros((T, top), dtype=np.int64)
    for t in range(1, T):
        dp = ((Pn[cand[t]][:, None, :] - Pn[cand[t - 1]][None, :, :]) ** 2).sum(-1)
        tot = acc[None, :] + smooth * dp
        back[t] = tot.argmin(1); acc = tot.min(1) + cost[t]
    path = np.zeros(T, dtype=np.int64); path[-1] = int(acc.argmin())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    idx = cand[np.arange(T), path]
    err = np.sqrt((((lF[idx] - lo_) ** 2) * has).sum(1) / np.maximum(has.sum(1), 1))
    return P[idx], err, names
