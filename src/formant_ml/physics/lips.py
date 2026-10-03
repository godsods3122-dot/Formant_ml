"""무른 입술 — 근육으로 움직이는 두 몸 (윗입술 · 아랫입술), 접촉 · 침 점착 · 앞니 (MEASUREMENTS §52.527).

사용자 (2026-10-02): *"입술은 특히 고려해야 할 게 많을 거야. 입술의 움직임도 많고, 입술을 다물었을 때 걸리는 힘과 입술 변형, 입술이 닿는 면적까지 싹다 봐야
좋은 결과가 나올 거야. 파열음, 순음 관련한 퀄리티가 중요하니까. 입술도 마찬가지로 강체로 볼 게 아니고 soft해야 해."*

기하 (잠정 — 화자 계측으로 바꿀 것): 입 너비 W (입꼬리–입꼬리, 동아시아 여성 ~4.6 cm), 윗입술 높이 H_u (붙는 곳 → 붉은 입술 끝), 두께 T_u, 아랫입술
H_l · T_l. 몸마다 (u 옆, v 앞뒤, w 위아래) 구조 격자 → 4 면체. 치열궁을 따라 x = x_앞 − v − u²/(2R) 로 휘고, 입꼬리(|u| → W/2)로 두께 · 높이가 줄어든다.
좌표: VTL (x 앞, y 위, z 옆) [cm] — FEM 은 SI 로 돌리고 내보낼 때 cm.

근육 (조직 안 섬유: Hill 형 능동 응력, `softtissue`): 0 구륜근 (가로 u, 두 입술의 붉은 입술 쪽 절반 — 오므림 · 내밂), 1 상순거근 (윗입술 세로 w, 앞쪽 —
들어 올림), 2 하순하제근 (아랫입술 세로 — 내림). 3 이근은 입술 밖(턱끝)의 근육이다 — 아래턱 앞니 뿌리 아래(절치와)에서 일어나 턱끝 피부에 붙고,
줄면 턱끝 연조직을 올려 아랫입술을 위 · 앞으로 밀어 내민다 (예전에는 아랫입술 아래 ¼ 의 세로 섬유로 두어 입을 열었다 — §52.529 결함). 아랫입술 아래
앞쪽 마디를 위 · 앞 (0.5, 0.87) 으로 미는 선 작용자로. 입가 밖 근육은 **입꼬리 결절(modiolus)** 을 당기는 선 작용자: 4 소근 (옆), 5 대관골근 (옆 · 위),
6 협근 (뒤), 7 입꼬리내림근 (아래 · 옆), 8 입꼬리올림근 (위). 결절: 윗 · 아랫입술은 입꼬리에서 이어진 한 고리다 — 두 몸의 입꼬리 열(맞닿는 면의 바깥
끝 두 열)을 굳은 묶음 용수철로 이어 입 틈이 입꼬리에서 끝나게 하고, 입가 근육의 힘은 그 묶음(양쪽 입술의 입꼬리 3 mm)에 건다 (예전에는 두 몸이 따로
놀아 입가 근육 다섯의 결과가 같았다 — §52.529).
경계: 윗입술 맨 위 층은 위턱(고정), 아랫입술 맨 아래 층은 아래턱 (턱 회전을 따라 움직임). 입술 뒷면은 앞니 면(단단, 평면 근사)과 벌점 접촉.
접촉: 윗입술 아랫면 ↔ 아랫입술 윗면 (같은 (u, v) 짝), 침 점착 (`softtissue.FEM.add_contact`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import softtissue as ST

MUSCLES = ("OO", "LLS", "DLI", "MEN", "RIS", "ZYG", "BUC", "DAO", "LAO")
LINE = {"RIS": (0.0, 0.0, 1.0), "ZYG": (0.0, 0.6, 0.8), "BUC": (-1.0, 0.0, 0.3), "DAO": (0.0, -0.7, 0.7), "LAO": (0.0, 1.0, 0.2)}
#: 선 작용자 최대 힘의 배율 (선 작용자 기본 1 N 에 곱한다, §52.531). 대관골근은 최대 수의 수축 1.4–1.9 N (외부 하중 측정: 196 g ≈ 1.92 N, PubMed 19631428;
#: 웃음 0.93 ± 0.48 N, PMC13105802) → 1.9. 나머지는 측정이 없어 잠정 1 N.
LINE_FMAX_SCALE = {"ZYG": 1.9}


@dataclass
class LipGeom:
    W: float = 4.6            # 입 너비 [cm]
    H_u: float = 1.6          # 윗입술 (붙는 곳 → 끝) [cm]
    T_u: float = 1.1
    H_l: float = 1.5
    T_l: float = 1.2
    R: float = 3.5            # 치열궁 곡률 반지름 [cm]
    gap0: float = 0.05        # 쉼 자세 두 입술 사이 틈 [cm]
    x_front: float = 5.3      # 정중선 입술 앞 끝 x [cm] (VTL)
    y_sto: float = -0.9       # 입술 사이 높이 [cm] (VTL)
    x_inc: float = 4.9        # 위 앞니 입술 쪽 면 x [cm] (VTL)
    overjet: float = 0.25     # 아래 앞니가 위 앞니보다 뒤 [cm] (수평 덮임 2–3 mm)
    nu: int = 24
    nv: int = 5
    nw: int = 8


def _lip_body(G: LipGeom, upper: bool, mat: ST.Material):
    H, T = (G.H_u, G.T_u) if upper else (G.H_l, G.T_l)
    X, tets, idx = ST.box_tets(G.nu, G.nv, G.nw, (1.0, 1.0, 1.0))
    u = (X[:, 0] - 0.5) * G.W
    v = X[:, 1]
    w = X[:, 2]
    taper = np.sqrt(np.clip(1.0 - (np.abs(u) / (0.5 * G.W)) ** 4, 0.0, 1.0)) * 0.75 + 0.25
    x_inc = G.x_inc if upper else G.x_inc - G.overjet
    x_front = x_inc + T + 0.02                                  # 쉼 자세: 입술 뒷면이 앞니 면 0.2 mm 앞
    xx = x_front - v * T * taper - u ** 2 / (2 * G.R)
    sgn = 1.0 if upper else -1.0
    yy = G.y_sto + sgn * (0.5 * G.gap0 + w * H * taper)
    zz = u
    P = np.stack([xx, yy, zz], 1) * 1e-2                         # → m
    ex = np.array([0.0, 0.0, 1.0])
    # 섬유 · 근육
    cen = P[tets].mean(1) * 1e2
    uc = cen[:, 2]
    vc = (x_front - uc ** 2 / (2 * G.R) - cen[:, 0])              # 앞에서 뒤로 깊이 [cm]
    wc = np.abs(cen[:, 1] - G.y_sto)
    fiber = np.zeros((len(tets), 3))
    muscle = -np.ones(len(tets), int)
    Tloc = T * (np.sqrt(np.clip(1.0 - (np.abs(uc) / (0.5 * G.W)) ** 4, 0, 1)) * 0.75 + 0.25)
    Hloc = H * (np.sqrt(np.clip(1.0 - (np.abs(uc) / (0.5 * G.W)) ** 4, 0, 1)) * 0.75 + 0.25)
    oo = wc <= 0.5 * Hloc
    fiber[oo] = ex
    muscle[oo] = MUSCLES.index("OO")
    vert = np.array([0.0, 1.0, 0.0])
    if upper:
        m = (~oo) & (vc <= 0.6 * Tloc)
        fiber[m] = vert
        muscle[m] = MUSCLES.index("LLS")
    else:
        m = (~oo) & (vc <= 0.6 * Tloc)                           # 하순하제근: 아랫입술 앞쪽 세로 (턱 쪽까지)
        fiber[m] = vert
        muscle[m] = MUSCLES.index("DLI")
    # 뼈에 붙는 곳: 윗입술 맨 위 · 아랫입술 맨 아래 층 — 입꼬리 `MODIOLUS_FREE_CM` 안은 뼈가 아니라 볼(입꼬리 결절)에 매달린다 (아래 `cheek`)
    edge = np.abs(w - 1.0) < 1e-9
    near = np.abs(u) > 0.5 * G.W - MODIOLUS_FREE_CM
    fixed = edge & ~near
    cheek = np.flatnonzero(edge & near)
    # 표면: 맞닿는 면 (w = 0) 마디의 (i, j) 격자와 뒷면 (v = 1) 마디
    contact = np.array([idx(i, j, 0) for i in range(G.nu + 1) for j in range(G.nv + 1)])
    back = np.array([idx(i, G.nv, k) for i in range(G.nu + 1) for k in range(G.nw + 1)])
    nc = max(1, int(round(0.3 / (G.W / G.nu))))                     # 입꼬리 3 mm 영역 (선 작용자 힘을 나눠 건다)
    corners = np.array([idx(i, j, k) for i in list(range(0, nc + 1)) + list(range(G.nu - nc, G.nu + 1)) for j in range(G.nv + 1) for k in range(G.nw + 1)])
    # 입 안 압력이 미는 면: 뒷면 (v = 1) 세모
    tris = []
    for i in range(G.nu):
        for k in range(G.nw):
            a, b, c, d = idx(i, G.nv, k), idx(i + 1, G.nv, k), idx(i + 1, G.nv, k + 1), idx(i, G.nv, k + 1)
            tris += [[a, c, b], [a, d, c]] if upper else [[a, b, c], [a, c, d]]
    # 입꼬리 묶음: 맞닿는 면 (w = 0) 의 바깥 끝 두 열 (같은 (i, j) 를 위 · 아래 몸에서 짝지음)
    tie = np.array([idx(i, j, 0) for i in (0, 1, G.nu - 1, G.nu) for j in range(G.nv + 1)])
    tie_w = np.array([1.0 if i in (0, G.nu) else 0.5 for i in (0, 1, G.nu - 1, G.nu) for j in range(G.nv + 1)])
    # 이근이 미는 턱끝 쪽: 아랫입술 아래 절반 (w 0.45–0.9) 의 앞쪽 (v ≤ 0.4), 가운데 ⅔ 너비
    chin = np.array([idx(i, j, k) for i in range(G.nu + 1) for j in range(G.nv + 1) for k in range(G.nw + 1)
                     if (not upper) and 0.45 <= k / G.nw <= 0.9 and j / G.nv <= 0.4 and abs(i / G.nu - 0.5) <= 1.0 / 3.0], int)
    body = ST.Body(X0=P, tets=tets, mat=mat, fixed=fixed, fiber=fiber, muscle=muscle, surf_tris=np.array(tris))
    body.meta.update(contact=contact, back=back, corners=corners, idx=idx, upper=upper, tie=tie, tie_w=tie_w, chin=chin, cheek=cheek)
    return body


#: **입꼬리 결절은 볼에 매달린다** (§52.530). 예전에는 입꼬리 끝까지 붙는 곳이 뼈에 고정돼 입가 근육 다섯(최대 1 N)이 입꼬리를 평균 0.1–0.3 mm 밖에 못
#: 움직였다 (웃음의 입꼬리 옆 이동 ~8–10 mm). 입꼬리 `MODIOLUS_FREE_CM` 안의 붙는 마디는 뼈 대신 볼 조직의 용수철로 쉼 자리에 — 볼 전체 강성
#: `K_CHEEK_TOTAL` [N/m] 를 그 마디들에 나눠 (볼 조직 E 2.6–4.4 kPa 로 매우 무르다, Luboz et al. 2014). 강성은 소근 최대(선 작용자 1 N)에서 입꼬리가
#: ~7 mm 옆으로 가게 잡은 어림 (힘 · 강성 모두 불확실해 둘의 비 = 이동을 측정값에 맞춘다).
MODIOLUS_FREE_CM = 0.6
K_CHEEK_TOTAL = 140.0


#: 입꼬리 묶음 용수철 [N/m] — 이어진 조직처럼 (요소 강성 E·h ≈ 40 kPa × 2 mm = 80 N/m 의 50 배). 명시 적분 안정: 마디 질량 ~8e-6 kg 에서 ω ≈ 2e4 rad/s,
#: dt < 2/ω ≈ 9e-5 s ≫ FEM 의 dt_stable ~1.6e-5 s.
K_TIE = 4000.0
#: 이근 선 작용자의 방향 (x 앞, y 위) — 턱끝 연조직을 위 · 앞으로.
MEN_DIR = (0.5, 0.87, 0.0)


class Lips:
    """윗입술 + 아랫입술 FEM. 활성 (9,) [0, 1], 턱 회전 [rad] (아래턱 붙는 곳을 경첩 둘레로), 입 안 압력 [Pa] → 입술 모양 · 접촉."""

    def __init__(self, G: LipGeom | None = None, E_upper=45e3, E_lower=33.7e3, device="cuda", line_fmax=1.0):
        self.G = G or LipGeom()
        self.mu = _lip_body(self.G, True, ST.Material(E=E_upper))
        self.ml = _lip_body(self.G, False, ST.Material(E=E_lower))
        self.fem = ST.FEM([self.mu, self.ml], device=device)
        o = self.fem.offs
        cu = self.mu.meta["contact"] + o[0]
        cl = self.ml.meta["contact"] + o[1]
        du = (self.G.W / self.G.nu) * (self.G.T_u / self.G.nv) * 1e-4
        self.fem.add_contact(cu, cl, np.full(len(cu), du), (0.0, 1.0, 0.0), sigma_c=3e3, delta_c=3e-4, damp=50.0)
        self.back = np.concatenate([self.mu.meta["back"] + o[0], self.ml.meta["back"] + o[1]])
        self.corners = np.concatenate([self.mu.meta["corners"] + o[0], self.ml.meta["corners"] + o[1]])
        self.line_fmax = line_fmax                        # 선 작용자 최대 힘 [N] (잠정)
        X0 = self.fem.X0
        self.fixed_idx = torch.nonzero(self.fem.fixed).squeeze(1)
        self.fixed_X0 = X0[self.fixed_idx].clone()
        self.lower_fixed = self.fixed_idx >= int(o[1])
        self.k_teeth = 5e7
        dev = self.fem.dev
        self.tie_u = torch.as_tensor(self.mu.meta["tie"] + o[0], device=dev)
        self.tie_l = torch.as_tensor(self.ml.meta["tie"] + o[1], device=dev)
        self.tie_w = torch.as_tensor(self.mu.meta["tie_w"], device=dev, dtype=self.fem.dt_)
        self.tie_r0 = (X0[self.tie_u] - X0[self.tie_l]).clone()
        self.chin = torch.as_tensor(self.ml.meta["chin"] + o[1], device=dev)
        self.cheek = torch.as_tensor(np.concatenate([self.mu.meta["cheek"] + o[0], self.ml.meta["cheek"] + o[1]]), device=dev)
        self.cheek_X0 = X0[self.cheek].clone()
        self.k_cheek = K_CHEEK_TOTAL / max(1, self.cheek.numel())

    def _soft_links(self, f: torch.Tensor, act):
        """입꼬리 묶음 + 이근 (선 작용자)."""
        fem = self.fem
        if self.cheek.numel():
            f.index_add_(0, self.cheek, -self.k_cheek * (fem.x[self.cheek] - self.cheek_X0))
        dx = fem.x[self.tie_u] - fem.x[self.tie_l] - self.tie_r0
        ft = -K_TIE * self.tie_w[:, None] * dx
        f.index_add_(0, self.tie_u, ft)
        f.index_add_(0, self.tie_l, -ft)
        a_men = float(act[MUSCLES.index("MEN")])
        if a_men > 0 and self.chin.numel():
            d = torch.as_tensor(MEN_DIR, device=fem.dev, dtype=fem.dt_)
            d = d / torch.linalg.norm(d)
            f.index_add_(0, self.chin, (a_men * self.line_fmax / self.chin.numel()) * d[None, :].expand(self.chin.numel(), 3))

    def _jaw(self, ang: float, pivot=(-6.0, 1.5)):
        """아래턱 회전 ang [rad] (+ 벌림) 의 아랫입술 붙는 곳 — VTL 정중면에서 턱관절 어림 둘레 (잠정)."""
        X = self.fixed_X0.clone()
        if ang != 0.0:
            px, py = pivot[0] * 1e-2, pivot[1] * 1e-2
            c, s = np.cos(-ang), np.sin(-ang)
            m = self.lower_fixed
            dx, dy = X[m, 0] - px, X[m, 1] - py
            X[m, 0] = px + c * dx - s * dy
            X[m, 1] = py + s * dx + c * dy
        return X

    def step(self, dt, act: np.ndarray, jaw: float = 0.0, p_oral: float = 0.0):
        fem = self.fem
        a = torch.as_tensor(np.asarray(act, float)[:4], device=fem.dev, dtype=fem.dt_)
        ps = torch.full((fem.surf.shape[0],), float(p_oral), device=fem.dev, dtype=fem.dt_)
        f = fem.forces(a, ps)
        # 선 작용자 (입가)
        for k, name in enumerate(MUSCLES[4:], start=4):
            if act[k] <= 0:
                continue
            d = torch.as_tensor(LINE[name], device=fem.dev, dtype=fem.dt_)
            ci = torch.as_tensor(self.corners, device=fem.dev)
            xs = fem.x[ci]
            side = torch.sign(xs[:, 2:3]).clamp(min=-1, max=1)
            dirv = d[None, :] * torch.cat([torch.ones_like(side), torch.ones_like(side), side], 1)
            dirv = dirv / torch.linalg.norm(dirv, dim=1, keepdim=True)
            f.index_add_(0, ci, act[k] * self.line_fmax * LINE_FMAX_SCALE.get(name, 1.0) / len(self.corners) * 2 * dirv)
        self._soft_links(f, act)
        # 앞니 면 (평면 x = x_inc, 아래는 수평 덮임만큼 뒤): 입술 뒷면 마디가 그 뒤로 못 간다
        bi = torch.as_tensor(self.back, device=fem.dev)
        # 아래 앞니는 아래턱과 함께 돈다 (경첩 둘레로 앞니 끝을 돌린 x)
        xl, yl = (self.G.x_inc - self.G.overjet), self.G.y_sto - 0.3
        px, py = -6.0, 1.5
        c_, s_ = np.cos(-jaw), np.sin(-jaw)
        xl_j = px + c_ * (xl - px) - s_ * (yl - py)
        xinc = torch.where(bi >= int(fem.offs[1]), torch.full_like(fem.x[bi, 0], xl_j * 1e-2),
                           torch.full_like(fem.x[bi, 0], self.G.x_inc * 1e-2))
        zb = fem.x[bi, 2]
        xinc = xinc - zb * zb / (2 * self.G.R * 1e-2)               # 치열궁: 앞니 면도 입술과 같은 곡률로 휜다
        pen = torch.clamp(xinc - fem.x[bi, 0], min=0.0)
        Ab = (self.G.W / self.G.nu) * (self.G.H_u / self.G.nw) * 1e-4
        f[bi, 0] += self.k_teeth * pen * Ab
        acc = f / fem.m[:, None]
        fem.v = fem.v + dt * acc
        fem.v[fem.fixed] = 0.0
        fem.x = fem.x + dt * fem.v
        fem.x[self.fixed_idx] = self._jaw(jaw)
        return f

    def settle(self, act, jaw=0.0, p_oral=0.0, T=0.15, ramp=0.4):
        """근육 활성 · 턱 · 입 안 압력을 T 의 앞 `ramp` 몫 동안 곧게 올린 뒤 기다린다 — 턱·근육은 유한한 빠르기로 움직인다 (한순간에 턱을
        5° 돌리면 입술 자리에서 ~1 cm 가 한 걸음에 옮겨져 요소가 뒤집혔다)."""
        dt = self.fem.dt_stable
        n = int(T / dt)
        nr = max(1, int(ramp * n))
        act = np.asarray(act, float)
        for i in range(n):
            r = min(1.0, (i + 1) / nr)
            self.step(dt, act * r, jaw * r, p_oral * r)

    def aperture(self, nz: int = 41):
        """입 벌림: 옆 z 마다 (윗입술 아랫면 최저 y − 아랫입술 윗면 최고 y) 의 양의 부분 → 넓이 [cm²], 너비 [cm], 가운데 높이 [cm], 접촉 넓이 [cm²]."""
        fem = self.fem
        x = fem.x.cpu().numpy() * 1e2
        o = fem.offs
        cu = x[self.mu.meta["contact"] + o[0]]
        cl = x[self.ml.meta["contact"] + o[1]]
        zs = np.linspace(-self.G.W / 2, self.G.W / 2, nz)
        dz = zs[1] - zs[0]
        h = np.zeros(nz)
        for k, z0 in enumerate(zs):
            mu_ = np.abs(cu[:, 2] - z0) < dz
            ml_ = np.abs(cl[:, 2] - z0) < dz
            if mu_.any() and ml_.any():
                h[k] = max(0.0, cu[mu_, 1].min() - cl[ml_, 1].max())
        area = float(h.sum() * dz)
        width = float((h > 0.02).sum() * dz)
        return dict(area=area, width=width, height=float(h[nz // 2]), contact_cm2=self.fem.contact_area[0] * 1e4 if self.fem.contact_area else 0.0,
                    contact_N=self.fem.contact_force[0] if self.fem.contact_force else 0.0, h=h, z=zs,
                    protrusion=float(x[:, 0].max()))



class LipsBatch:
    """입술 B 벌을 한 FEM 에 (서로 닿지 않는 몸 2B 개) — 대리 모형 학습 표본을 GPU 한 계산으로 (§52.527). 벌마다 활성 (9,), 턱 [rad], 입 안 압력 [Pa]."""

    def __init__(self, B: int, G: LipGeom | None = None, E_upper=45e3, E_lower=33.7e3, device="cuda", line_fmax=1.0):
        self.G = G or LipGeom()
        self.B = B
        bodies = []
        for b in range(B):
            mu = _lip_body(self.G, True, ST.Material(E=E_upper))
            ml = _lip_body(self.G, False, ST.Material(E=E_lower))
            for body in (mu, ml):
                body.muscle = np.where(body.muscle >= 0, body.muscle + 4 * b, -1)
            bodies += [mu, ml]
        self.bodies = bodies
        self.fem = ST.FEM(bodies, device=device)
        o = self.fem.offs
        du = (self.G.W / self.G.nu) * (self.G.T_u / self.G.nv) * 1e-4
        cu = np.concatenate([bodies[2 * b].meta["contact"] + o[2 * b] for b in range(B)])
        cl = np.concatenate([bodies[2 * b + 1].meta["contact"] + o[2 * b + 1] for b in range(B)])
        self.n_cpair = len(bodies[0].meta["contact"])
        self.fem.add_contact(cu, cl, np.full(len(cu), du), (0.0, 1.0, 0.0), sigma_c=3e3, delta_c=3e-4, damp=50.0)
        dev = self.fem.dev
        self.back = torch.as_tensor(np.concatenate([bodies[i].meta["back"] + o[i] for i in range(2 * B)]), device=dev)
        self.back_copy = torch.as_tensor(np.concatenate([np.full(len(bodies[i].meta["back"]), i // 2) for i in range(2 * B)]), device=dev)
        self.back_lower = torch.as_tensor(np.concatenate([np.full(len(bodies[i].meta["back"]), i % 2 == 1) for i in range(2 * B)]), device=dev)
        self.corners = torch.as_tensor(np.concatenate([bodies[i].meta["corners"] + o[i] for i in range(2 * B)]), device=dev)
        self.corner_copy = torch.as_tensor(np.concatenate([np.full(len(bodies[i].meta["corners"]), i // 2) for i in range(2 * B)]), device=dev)
        self.n_corner = len(bodies[0].meta["corners"]) * 2
        self.line_fmax = line_fmax
        self.fixed_idx = torch.nonzero(self.fem.fixed).squeeze(1)
        self.fixed_X0 = self.fem.X0[self.fixed_idx].clone()
        node_copy = np.concatenate([np.full(b_.X0.shape[0], i // 2) for i, b_ in enumerate(bodies)])
        node_lower = np.concatenate([np.full(b_.X0.shape[0], i % 2 == 1) for i, b_ in enumerate(bodies)])
        nc = torch.as_tensor(node_copy, device=dev)
        nl = torch.as_tensor(node_lower, device=dev)
        self.fixed_copy = nc[self.fixed_idx]
        self.fixed_lower = nl[self.fixed_idx]
        self.k_teeth = 5e7
        self.Ab = (self.G.W / self.G.nu) * (self.G.H_u / self.G.nw) * 1e-4
        self.tie_u = torch.as_tensor(np.concatenate([bodies[2 * b].meta["tie"] + o[2 * b] for b in range(B)]), device=dev)
        self.tie_l = torch.as_tensor(np.concatenate([bodies[2 * b + 1].meta["tie"] + o[2 * b + 1] for b in range(B)]), device=dev)
        self.tie_w = torch.as_tensor(np.concatenate([bodies[2 * b].meta["tie_w"] for b in range(B)]), device=dev, dtype=self.fem.dt_)
        self.tie_r0 = (self.fem.X0[self.tie_u] - self.fem.X0[self.tie_l]).clone()
        self.chin = torch.as_tensor(np.concatenate([bodies[2 * b + 1].meta["chin"] + o[2 * b + 1] for b in range(B)]), device=dev)
        self.chin_copy = torch.as_tensor(np.concatenate([np.full(len(bodies[2 * b + 1].meta["chin"]), b) for b in range(B)]), device=dev)
        self.n_chin = len(bodies[1].meta["chin"])
        self.cheek = torch.as_tensor(np.concatenate([bodies[i].meta["cheek"] + o[i] for i in range(2 * B)]), device=dev)
        self.cheek_X0 = self.fem.X0[self.cheek].clone()
        self.k_cheek = K_CHEEK_TOTAL / max(1, len(bodies[0].meta["cheek"]) + len(bodies[1].meta["cheek"]))

    def step(self, dt, act: torch.Tensor, jaw: torch.Tensor, p_oral: torch.Tensor):
        """act (B, 9), jaw (B,) [rad], p_oral (B,) [Pa] — torch."""
        fem = self.fem
        dev, dt_ = fem.dev, fem.dt_
        a_f = act[:, :4].reshape(-1)
        # 압력: 표면 세모가 몸 순서대로 이어 붙어 있다 (몸마다 같은 수)
        nt = fem.surf.shape[0] // (2 * self.B)
        ps = p_oral.repeat_interleave(2 * nt)
        f = fem.forces(a_f, ps)
        # 선 작용자
        xs = fem.x[self.corners]
        side = torch.sign(xs[:, 2:3])
        for k, name in enumerate(MUSCLES[4:], start=4):
            ak = act[self.corner_copy, k]
            if not bool((ak > 0).any()):
                continue
            d = torch.as_tensor(LINE[name], device=dev, dtype=dt_)
            dirv = d[None, :] * torch.cat([torch.ones_like(side), torch.ones_like(side), side], 1)
            dirv = dirv / torch.linalg.norm(dirv, dim=1, keepdim=True)
            f.index_add_(0, self.corners, (ak * self.line_fmax * LINE_FMAX_SCALE.get(name, 1.0) / self.n_corner * 2)[:, None] * dirv)
        # 볼에 매달린 입꼬리 붙는 곳
        if self.cheek.numel():
            f.index_add_(0, self.cheek, -self.k_cheek * (fem.x[self.cheek] - self.cheek_X0))
        # 입꼬리 묶음
        dx = fem.x[self.tie_u] - fem.x[self.tie_l] - self.tie_r0
        ft = -K_TIE * self.tie_w[:, None] * dx
        f.index_add_(0, self.tie_u, ft)
        f.index_add_(0, self.tie_l, -ft)
        # 이근 (턱끝 연조직을 위 · 앞으로)
        am = act[self.chin_copy, MUSCLES.index("MEN")]
        if bool((am > 0).any()) and self.n_chin:
            d = torch.as_tensor(MEN_DIR, device=dev, dtype=dt_)
            d = d / torch.linalg.norm(d)
            f.index_add_(0, self.chin, (am * self.line_fmax / self.n_chin)[:, None] * d[None, :])
        # 앞니 면 (치열궁 곡률, 아래 앞니는 턱과 함께)
        xl, yl = (self.G.x_inc - self.G.overjet), self.G.y_sto - 0.3
        px, py = -6.0, 1.5
        jb = jaw[self.back_copy]
        xl_j = px + torch.cos(-jb) * (xl - px) - torch.sin(-jb) * (yl - py)
        xinc = torch.where(self.back_lower, xl_j, torch.full_like(xl_j, self.G.x_inc)) * 1e-2
        zb = fem.x[self.back, 2]
        xinc = xinc - zb * zb / (2 * self.G.R * 1e-2)
        pen = torch.clamp(xinc - fem.x[self.back, 0], min=0.0)
        f[self.back, 0] += self.k_teeth * pen * self.Ab
        acc = f / fem.m[:, None]
        fem.v = fem.v + dt * acc
        fem.v[fem.fixed] = 0.0
        fem.x = fem.x + dt * fem.v
        # 붙는 곳: 아래턱 회전
        X = self.fixed_X0.clone()
        jf = jaw[self.fixed_copy]
        m = self.fixed_lower
        pxm, pym = px * 1e-2, py * 1e-2
        dx, dy = X[m, 0] - pxm, X[m, 1] - pym
        c_, s_ = torch.cos(-jf[m]), torch.sin(-jf[m])
        X[m, 0] = pxm + c_ * dx - s_ * dy
        X[m, 1] = pym + s_ * dx + c_ * dy
        fem.x[self.fixed_idx] = X

    def settle(self, act, jaw, p_oral, T=0.06, ramp=0.4):
        fem = self.fem
        dev, dt_ = fem.dev, fem.dt_
        act = torch.as_tensor(np.asarray(act, float), device=dev, dtype=dt_)
        jaw = torch.as_tensor(np.asarray(jaw, float), device=dev, dtype=dt_)
        po = torch.as_tensor(np.asarray(p_oral, float), device=dev, dtype=dt_)
        dt = fem.dt_stable
        n = int(T / dt)
        nr = max(1, int(ramp * n))
        for i in range(n):
            r = min(1.0, (i + 1) / nr)
            self.step(dt, act * r, jaw * r, po * r)

    def shapes(self, nz: int = 33, xs_rel=(0.0, 0.3, 0.6, 0.9)):
        """벌마다 입술 사이 틈 h(z) [cm] (입술 앞 끝에서 뒤로 xs_rel cm 의 단면 넷), 넓이, 앞 끝 x, 접촉 넓이 [cm²]."""
        fem = self.fem
        x = fem.x.cpu().numpy() * 1e2
        o = fem.offs
        out = []
        g = fem.pairs[0]["gap"].cpu().numpy() * 1e2
        A = fem.pairs[0]["A"].cpu().numpy() * 1e4
        zs = np.linspace(-self.G.W / 2, self.G.W / 2, nz)
        dz = zs[1] - zs[0]
        for b in range(self.B):
            cu = x[self.bodies[2 * b].meta["contact"] + o[2 * b]]
            cl = x[self.bodies[2 * b + 1].meta["contact"] + o[2 * b + 1]]
            front = float(max(x[o[2 * b]:o[2 * b + 2], 0].max(), 0))
            prof = []
            for xr in xs_rel:
                x0 = front - xr
                h = np.zeros(nz)
                for k, z0 in enumerate(zs):
                    mu_ = (np.abs(cu[:, 2] - z0) < dz) & (np.abs(cu[:, 0] - x0) < 0.2)
                    ml_ = (np.abs(cl[:, 2] - z0) < dz) & (np.abs(cl[:, 0] - x0) < 0.2)
                    if mu_.any() and ml_.any():
                        h[k] = max(0.0, cu[mu_, 1].min() - cl[ml_, 1].max())
                prof.append(h)
            prof = np.array(prof)
            sl = slice(b * self.n_cpair, (b + 1) * self.n_cpair)
            out.append(dict(h=prof, area=prof.sum(1) * dz, front=front, contact=float((A[sl] * (g[sl] < 0)).sum())))
        return out
