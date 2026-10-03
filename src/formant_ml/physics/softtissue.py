"""무른 조직 유한요소 (MEASUREMENTS §52.527) — 입술(다음은 혀·연구개)을 강체·기하 변수가 아니라 근육으로 움직이는 무른 몸으로.

사용자 (2026-10-02): *"입술의 움직임도 많고, 입술을 다물었을 때 걸리는 힘과 입술 변형, 입술이 닿는 면적까지 싹다 봐야 … 파열음, 순음 관련한 퀄리티가 중요하니까.
입술도 마찬가지로 강체로 볼 게 아니고 soft해야 해."*

명시적 동역학 (중앙 차분, 덩어리 질량), 4 면체 선형 요소, 전체 라그랑주:
* **재료**: 압축성 Neo-Hookean P = μ(F − F⁻ᵀ) + λ ln J F⁻ᵀ, 거의 비압축 (ν 0.49). 근거: 윗입술 E ≈ 45 kPa (압입), 아랫입술 33.7 ± 7.3 kPa (흡인,
  Luboz et al. 2014), 입가 볼 2.6–4.4 kPa.
* **점성**: Kelvin–Voigt σ_v = η (L + Lᵀ), L = Ḟ F⁻¹ (η 는 잠정 — 연조직 전단 점성 1–10 Pa·s 범위).
* **근육**: Hill 형 능동 응력 σ_a = a σ_max f_L(λ_f) 를 섬유 방향 d 로 — P_a = σ_a (F d ⊗ d)/λ_f, f_L = exp(−((λ_f − 1)/0.45)²), σ_max ≈ 0.2 MPa
  (골격근 최대 등척 응력 0.2–0.3 MPa 의 어림, 잠정).
* **경계**: 뼈에 붙은 마디는 운동학적으로 (턱 회전 등).
* **표면 하중**: 입 안 압력 등을 표면 세모에 법선 힘으로.
* **접촉 · 점착**: 두 표면(같은 매개 격자를 가진 위·아래 입술의 맞닿는 면) 사이 틈 g 로 벌점 접촉 압력 k_c max(0, −g) + 감쇠, 한 번 닿은 쌍은 침
  막의 응집 법칙 t = σ_c (1 − g/δ_c) (0 < g < δ_c) 로 끌어당기다 끊긴다. σ_c 는 모세관 압력 ~ 2γ/h_막 (침 γ ≈ 0.05 N/m, 막 수십 µm → 2–5 kPa;
  말소리 [p] 의 입술 사이 압력 1–3 kPa 와 같은 크기 — Hinton & Luschei 1992), δ_c ~ 균열 전선 간격 300 µm (Abkarian & Stone 2020). 잠정.
단위: SI (m, kg, s, Pa) — 음향 모듈(CGS) 과 맞출 때 바꾼다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Material:
    E: float = 40e3          # Pa
    nu: float = 0.49
    rho: float = 1050.0      # kg/m³
    eta: float = 3.0         # Pa·s (Kelvin–Voigt, 잠정)

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


@dataclass
class Body:
    """4 면체 무른 몸. X0 (N,3) 쉼 자리 [m], tets (M,4), mat, fixed (N,) bool, fiber (M,3) 단위 벡터 · muscle (M,) int (−1 = 없음)."""
    X0: np.ndarray
    tets: np.ndarray
    mat: Material
    fixed: np.ndarray
    fiber: np.ndarray | None = None
    muscle: np.ndarray | None = None
    surf_tris: np.ndarray | None = None          # 바깥 표면 세모 (K,3) — 압력 하중
    meta: dict = field(default_factory=dict)


def tetrahedral_nodal_forces(P, volume, reference_inverse, tets, n_nodes):
    """Assemble first-Piola stresses [Pa] into nodal forces [N] (SI tensors)."""
    H = -volume[:, None, None] * P @ reference_inverse.transpose(1, 2)
    force = P.new_zeros((n_nodes, 3))
    for k in range(3):
        force.index_add_(0, tets[:, k + 1], H[:, :, k])
    force.index_add_(0, tets[:, 0], -H.sum(2))
    return force


def triangle_area_vectors(x, triangles):
    """Current oriented triangle area vectors [m²]; winding defines outward."""
    a, b, c = x[triangles[:, 0]], x[triangles[:, 1]], x[triangles[:, 2]]
    return 0.5 * torch.cross(b - a, c - a, dim=1)


def triangle_pressure_forces(x, triangles, pressure):
    """Consistent constant-pressure traction: inward positive, [Pa] -> [N]."""
    force = torch.zeros_like(x)
    nodal = -(pressure[:, None] * triangle_area_vectors(x, triangles)) / 3.0
    for k in range(3):
        force.index_add_(0, triangles[:, k], nodal)
    return force


class FEM:
    """여러 몸을 한꺼번에 (마디를 이어 붙여 한 배열로). 접촉 쌍은 같은 수의 마디 목록 두 개 (같은 매개 격자 위 짝)."""

    def __init__(self, bodies: list, device="cuda", dtype=torch.float64, sigma_max: float = 0.2e6):
        self.dev, self.dt_ = torch.device(device), dtype
        T = lambda x, d=dtype: torch.as_tensor(np.ascontiguousarray(x), device=self.dev, dtype=d)
        offs = np.cumsum([0] + [b.X0.shape[0] for b in bodies])
        self.offs = offs
        X0 = np.concatenate([b.X0 for b in bodies])
        tets = np.concatenate([b.tets + o for b, o in zip(bodies, offs[:-1])])
        self.N, self.M = X0.shape[0], tets.shape[0]
        self.X0 = T(X0)
        self.x = self.X0.clone()
        self.v = torch.zeros_like(self.x)
        self.tets = T(tets, torch.long)
        Dm = (self.X0[self.tets[:, 1:]] - self.X0[self.tets[:, :1]]).transpose(1, 2)        # (M,3,3) 열 = 모서리
        self.V0 = torch.abs(torch.det(Dm)) / 6.0
        self.Dm_inv = torch.linalg.inv(Dm)
        mu = np.concatenate([np.full(b.tets.shape[0], b.mat.mu) for b in bodies])
        lam = np.concatenate([np.full(b.tets.shape[0], b.mat.lam) for b in bodies])
        eta = np.concatenate([np.full(b.tets.shape[0], b.mat.eta) for b in bodies])
        rho = np.concatenate([np.full(b.tets.shape[0], b.mat.rho) for b in bodies])
        self.mu, self.lam, self.eta = T(mu), T(lam), T(eta)
        m = torch.zeros(self.N, device=self.dev, dtype=dtype)
        m.index_add_(0, self.tets.reshape(-1), (T(rho) * self.V0 / 4.0).repeat_interleave(4))
        self.m = m
        self.fixed = T(np.concatenate([b.fixed for b in bodies]), torch.bool)
        fib = [b.fiber if b.fiber is not None else np.zeros((b.tets.shape[0], 3)) for b in bodies]
        mus = [b.muscle if b.muscle is not None else -np.ones(b.tets.shape[0], int) for b in bodies]
        self.fiber = T(np.concatenate(fib))
        self.muscle = T(np.concatenate(mus), torch.long)
        self.sigma_max = sigma_max
        st = [b.surf_tris + o for b, o in zip(bodies, offs[:-1]) if b.surf_tris is not None]
        self.surf = T(np.concatenate(st), torch.long) if st else None
        self.pairs = []                            # (iA (K,), iB (K,), 넓이 (K,), 법선 B→A (3,), 상태)
        # 안정 시간 간격: 요소의 가장 짧은 높이 (3V / 맞은편 면 넓이) / 압축파 속도 × 0.4. 꼭짓점 0 의 모서리만 보면 입가의 얇은 요소에서 너무 컸다.
        P4 = self.X0[self.tets]                                                               # (M,4,3)
        hmin = None
        for k in range(4):
            o = [j for j in range(4) if j != k]
            a_, b_, c_ = P4[:, o[0]], P4[:, o[1]], P4[:, o[2]]
            Af = 0.5 * torch.linalg.norm(torch.cross(b_ - a_, c_ - a_, dim=1), dim=1)
            hk = 3.0 * self.V0 / Af.clamp_min(1e-30)
            hmin = hk if hmin is None else torch.minimum(hmin, hk)
        cp = float(torch.sqrt((self.lam.max() + 2 * self.mu.max()) / torch.as_tensor(rho.min())))
        self.h_min = float(hmin.min())
        self.dt_stable = 0.4 * self.h_min / cp

    def add_contact(self, iA, iB, area, normal, k_c=None, sigma_c=3e3, delta_c=3e-4, g0_frac=0.1, mu_f=0.3, v_eps=1e-3, damp=0.0):
        """접촉 쌍 — 마디 iA (위 몸 표면) 와 iB (아래 몸 표면) 를 짝으로, 틈 g = (x_A − x_B)·n (n: B 에서 A 쪽 단위 벡터).
        침 막 응집 (닿은 적 있는 쌍): 쌍선형 법칙 — g ≤ g0 = g0_frac·δ_c 에서 σ_c g/g0 로 오르고, g0 < g < δ_c 에서 σ_c (δ_c − g)/(δ_c − g0) 로
        무르다 δ_c 에서 끊긴다. 지나간 최대 틈 g_max 아래로 돌아오면 원점으로 곧게 (비가역). 미끄럼: 정규화 쿨롱 마찰 μ_f |법선 힘| (닿거나 붙은 쌍).
        k_c 기본 5e7 Pa/m (2 kPa 에서 침투 0.04 mm)."""
        n = torch.as_tensor(np.asarray(normal, float) / np.linalg.norm(normal), device=self.dev, dtype=self.dt_)
        K = len(iA)
        st = dict(iA=torch.as_tensor(iA, device=self.dev), iB=torch.as_tensor(iB, device=self.dev),
                  A=torch.as_tensor(area, device=self.dev, dtype=self.dt_), n=n, k=k_c or 5e7, sc=sigma_c, dc=delta_c, g0=g0_frac * delta_c,
                  mu=mu_f, veps=v_eps, damp=damp,
                  stuck=torch.zeros(K, device=self.dev, dtype=torch.bool), gmax=torch.zeros(K, device=self.dev, dtype=self.dt_))
        self.pairs.append(st)
        return len(self.pairs) - 1

    def forces(self, act: torch.Tensor | None = None, p_surf: torch.Tensor | None = None):
        """마디 힘 (N,3). act: 근육별 활성 (n_muscle,) ∈ [0,1]. p_surf: 표면 세모별 압력 (K,) [Pa] (바깥 법선 반대로 미는 쪽 +)."""
        x, v = self.x, self.v
        Ds = (x[self.tets[:, 1:]] - x[self.tets[:, :1]]).transpose(1, 2)
        F = Ds @ self.Dm_inv
        J = torch.det(F).clamp_min(0.2)
        Finv = torch.linalg.inv(F)
        FinvT = Finv.transpose(1, 2)
        P = self.mu[:, None, None] * (F - FinvT) + (self.lam * torch.log(J))[:, None, None] * FinvT
        # 점성 (Kelvin–Voigt)
        Vs = (v[self.tets[:, 1:]] - v[self.tets[:, :1]]).transpose(1, 2)
        Fd = Vs @ self.Dm_inv
        Lv = Fd @ Finv
        sig_v = self.eta[:, None, None] * (Lv + Lv.transpose(1, 2))
        P = P + J[:, None, None] * sig_v @ FinvT
        # 근육
        if act is not None and act.numel():
            has = self.muscle >= 0
            a = torch.zeros(self.M, device=self.dev, dtype=self.dt_)
            a[has] = act[self.muscle[has]]
            Fd_ = (F @ self.fiber[:, :, None])[..., 0]
            lf = torch.linalg.norm(Fd_, dim=1).clamp_min(1e-6)
            sa = a * self.sigma_max * torch.exp(-((lf - 1.0) / 0.45) ** 2)
            P = P + (sa / lf)[:, None, None] * Fd_[:, :, None] * self.fiber[:, None, :]
        f = tetrahedral_nodal_forces(P, self.V0, self.Dm_inv, self.tets, self.N)
        if p_surf is not None and self.surf is not None:
            nA = triangle_area_vectors(x, self.surf)
            fs = -(p_surf[:, None] * nA) / 3.0
            for k in range(3):
                f.index_add_(0, self.surf[:, k], fs)
        # 접촉 · 점착 · 마찰
        self.contact_force, self.contact_area, self.adhesion_force = [], [], []
        for st in self.pairs:
            xa, xb = x[st["iA"]], x[st["iB"]]
            va, vb = v[st["iA"]], v[st["iB"]]
            n_ = st["n"]
            g = (xa - xb) @ n_
            vrel = va - vb
            gd = vrel @ n_
            pen = torch.clamp(-g, min=0.0)
            touching = pen > 0
            pc = st["k"] * pen + st["damp"] * torch.clamp(-gd, min=0.0) * touching
            st["stuck"] |= touching
            # 쌍선형 응집 (비가역)
            gp = torch.clamp(g, min=0.0)
            def env(gg):
                return torch.where(gg <= st["g0"], st["sc"] * gg / st["g0"],
                                   torch.clamp(st["sc"] * (st["dc"] - gg) / (st["dc"] - st["g0"]), min=0.0))
            load = gp >= st["gmax"]
            st["gmax"] = torch.where(st["stuck"] & load, gp, st["gmax"])
            alive = st["stuck"] & (st["gmax"] < st["dc"]) & (g > 0)
            coh = torch.where(alive, torch.where(load, env(gp), env(st["gmax"]) * gp / st["gmax"].clamp_min(1e-12)), torch.zeros_like(g))
            tn = (pc - coh) * st["A"]                                                         # + 면 A 를 n 쪽으로
            fa = tn[:, None] * n_[None, :]
            # 정규화 쿨롱 마찰 (닿거나 붙은 쌍)
            vt = vrel - gd[:, None] * n_[None, :]
            fric_on = touching | alive
            fmag = st["mu"] * torch.abs(tn) * fric_on
            fa = fa - fmag[:, None] * vt / torch.sqrt((vt * vt).sum(1, keepdim=True) + st["veps"] ** 2)
            f.index_add_(0, st["iA"], fa)
            f.index_add_(0, st["iB"], -fa)
            self.contact_force.append(float((pc * st["A"]).sum()))
            self.contact_area.append(float((st["A"] * touching).sum()))
            self.adhesion_force.append(float((coh * st["A"]).sum()))
            st["gap"] = g
        return f

    def step(self, dt: float, act=None, p_surf=None, x_fixed: torch.Tensor | None = None):
        f = self.forces(act, p_surf)
        a = f / self.m[:, None]
        self.v = self.v + dt * a
        self.v[self.fixed] = 0.0
        self.x = self.x + dt * self.v
        if x_fixed is not None:
            self.x[self.fixed] = x_fixed
        return f


def box_tets(nx, ny, nz, size):
    """직육면체 [0,size] 를 hex 격자 → 4 면체 (hex 하나에 5 개가 아니라 6 개 — 대칭 좋은 나눔). 반환 X (N,3), tets (M,4), 격자 번호 함수."""
    xs = [np.linspace(0, size[a], n + 1) for a, n in enumerate((nx, ny, nz))]
    X = np.stack(np.meshgrid(*xs, indexing="ij"), -1).reshape(-1, 3)
    idx = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                v0, v1, v2, v3, v4, v5, v6, v7 = c[0], c[4], c[6], c[2], c[1], c[5], c[7], c[3]
                tets += [[v0, v1, v2, v6], [v0, v2, v3, v6], [v0, v3, v7, v6], [v0, v7, v4, v6], [v0, v4, v5, v6], [v0, v5, v1, v6]]
    tets = np.array(tets)
    # 방향 맞추기 (부피 +)
    D = X[tets[:, 1:]] - X[tets[:, :1]]
    neg = np.linalg.det(D) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return X, tets, idx
