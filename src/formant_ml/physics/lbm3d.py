"""성도 흐름 + 소리 — 3D 격자 볼츠만 (D3Q19, GPU torch) (MEASUREMENTS §52.528).

사용자 (2026-10-02): *"제트 기류가 나온다고 해서 끝나는 게 아니라 그 기류가 연구개 뒤쪽에서 산란하겠지. 그러면서 파형도 차이가 생길 거고."* 선형 음향
(`fdtd3d`) 에는 흐름이 없다 — 성문 제트·소용돌이·굽이(인두 → 구강)에서의 산란과 그 소리(소용돌이 소리, 흐름–음향 상호작용)를 잴 수 없다.

* 압축성 격자 볼츠만은 흐름과 소리를 함께 푼다 (실제 음속: 격자 음속 1/√3 = c dt/dx → dt = dx/(√3 c)). 성문 제트 ~30 m/s 는 마하 0.08.
* 충돌: 정규화 BGK (비평형을 2 차 모멘트로 사영) + Smagorinsky LES (C_s 0.17) — 실제 공기 점성에서 τ − ½ ≈ 3e−4 라 BGK 는 불안정하다.
* 벽: 중간 되튐 (no-slip). 성문: 첫 띠 칸들에 주어진 속도(성문 유량 / 넓이, 성도 축 방향)의 평형 분포. 입 밖: 바깥 공기 + 흡수층(평형으로 끌어당김).
* 비교: 같은 성문 유량을 선형 3D 음향에 넣은 결과와 입 앞 압력·입 유량의 스펙트럼 차 = 흐름이 만든 몫.
SI 단위 (m, s, kg).
"""
from __future__ import annotations

import math

import numpy as np
import torch

# D3Q19
E = np.array([[0, 0, 0],
              [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
              [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]])
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
CS2 = 1.0 / 3.0


class LBM:
    def __init__(self, solid: np.ndarray, dx: float, c_phys: float = 357.0, nu_phys: float = 1.65e-5, rho0: float = 1.12,
                 sponge: np.ndarray | None = None, cs_smag: float = 0.17, device="cuda", dtype=torch.float32):
        self.dev, self.dt_ = torch.device(device), dtype
        self.dx = dx
        self.dt = dx / (math.sqrt(3.0) * c_phys)
        self.c = c_phys
        self.rho0 = rho0
        self.u_scale = dx / self.dt                       # 격자 속도 1 = 이 [m/s]
        self.nu_lat = nu_phys * self.dt / dx ** 2
        self.tau0 = 0.5 + 3.0 * self.nu_lat
        self.cs = cs_smag
        sh = solid.shape
        self.shape = sh
        self.solid = torch.as_tensor(solid, device=self.dev)
        self.fluid = ~self.solid
        self.e = torch.as_tensor(E, device=self.dev, dtype=dtype)
        self.w = torch.as_tensor(W, device=self.dev, dtype=dtype)
        self.f = self.w.view(19, 1, 1, 1).expand(19, *sh).clone()
        self.sponge = None if sponge is None else torch.as_tensor(sponge, device=self.dev, dtype=dtype)
        self.inlet = None

    def set_inlet(self, mask: np.ndarray, direction):
        d = np.asarray(direction, float)
        d /= np.linalg.norm(d)
        self.inlet = torch.as_tensor(mask, device=self.dev)
        self.inlet_dir = torch.as_tensor(d, device=self.dev, dtype=self.dt_)

    def macro(self):
        rho = self.f.sum(0)
        u = torch.einsum("qd,qxyz->dxyz", self.e, self.f) / rho
        return rho, u

    def feq(self, rho, u):
        eu = torch.einsum("qd,dxyz->qxyz", self.e, u)
        uu = (u * u).sum(0)
        return self.w.view(19, 1, 1, 1) * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uu)

    def step(self, u_in_lat: float = 0.0):
        f = self.f
        rho, u = self.macro()
        if self.inlet is not None:
            u = torch.where(self.inlet, self.inlet_dir.view(3, 1, 1, 1) * u_in_lat, u)
        fe = self.feq(rho, u)
        fneq = f - fe
        # 정규화: 비평형 2 차 모멘트 Π = Σ e e f_neq → f_neq ≈ w (9/2) Q:Π, Q = e e − c_s² I
        Pi = torch.einsum("qa,qb,qxyz->abxyz", self.e, self.e, fneq)
        Pmag = torch.sqrt(2.0 * (Pi * Pi).sum((0, 1))) + 1e-12
        # Smagorinsky: τ_eff = ½ (τ0 + √(τ0² + 18 √2 C_s² |Π| / ρ))
        tau = 0.5 * (self.tau0 + torch.sqrt(self.tau0 ** 2 + 18.0 * math.sqrt(2.0) * self.cs ** 2 * Pmag / rho))
        ee = torch.einsum("qa,qb->qab", self.e, self.e) - CS2 * torch.eye(3, device=self.dev, dtype=self.dt_)
        freg = 4.5 * self.w.view(19, 1, 1, 1) * torch.einsum("qab,abxyz->qxyz", ee, Pi)
        fpost = fe + (1.0 - 1.0 / tau) * freg
        if self.sponge is not None:
            # 흡수층: 쉼 평형(ρ0 = 1, u = 0) 쪽으로 끌어당김
            f0 = self.w.view(19, 1, 1, 1) * torch.ones_like(rho)
            fpost = fpost + self.sponge * (f0 - fpost)
        if self.inlet is not None:
            fpost = torch.where(self.inlet, fe, fpost)
        # 흐르기 + 중간 되튐
        out = torch.empty_like(fpost)
        for q in range(19):
            sx, sy, sz = (int(v) for v in E[q])
            out[q] = torch.roll(fpost[q], shifts=(sx, sy, sz), dims=(0, 1, 2))
        # 고체 칸으로 들어간 분포는 반대 방향으로 되돌린다 (중간 되튐): 고체 칸에서 온 값 대신 같은 칸의 반대 방향 사후 분포
        for q in range(1, 19):
            sx, sy, sz = (int(v) for v in E[q])
            from_solid = torch.roll(self.solid, shifts=(sx, sy, sz), dims=(0, 1, 2))
            out[q] = torch.where(from_solid & self.fluid, fpost[OPP[q]], out[q])
        out[:, self.solid] = self.w.view(19, 1).expand(19, int(self.solid.sum()))
        self.f = out

    def pressure(self, rho):
        """음향 압력 [Pa] = (ρ − 1) c_s² × ρ0 (u_scale)² — 격자 밀도 요동 → 물리."""
        return (rho - 1.0) * CS2 * self.rho0 * self.u_scale ** 2
