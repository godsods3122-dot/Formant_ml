"""격자 볼츠만 D3Q19 — numba CUDA 묶은 커널 (MEASUREMENTS §52.528). 분포는 평형 가중치에서 뺀 편차로 담는다. `lbm3d.LBM` 와 같은 물리를 한 커널로:
당기기 흐르기 + 중간 되튐 + 정규화 BGK + Smagorinsky LES + 성문 입구(속도 평형) + 흡수층(쉼 평형으로 끌어당김). 분포는 f[q·N + 칸], 버퍼 둘.
torch 의 `lbm3d.LBM` 는 연산을 쪼개 칸 8640 개에 3.3 ms/걸음 — 성도 크기(~10M 칸)에서 몇 시간이 걸려 묶었다.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from numba import cuda, float32, int32

from .lbm3d import E, W, OPP

EX = tuple(int(v) for v in E[:, 0])
EY = tuple(int(v) for v in E[:, 1])
EZ = tuple(int(v) for v in E[:, 2])
# float32 사본 — 정수·파이썬 실수와 섞이면 numba 가 float64 로 올린다 (이 GPU 의 float64 는 float32 의 1/64; 1천만 칸 30 ms/걸음이었다)
FX = tuple(np.float32(v) for v in E[:, 0])
FY = tuple(np.float32(v) for v in E[:, 1])
FZ = tuple(np.float32(v) for v in E[:, 2])
WQ = tuple(np.float32(v) for v in W)
OQ = tuple(int(v) for v in OPP)
F0, F1, F3, F45, F15, FH, F13, F2 = (np.float32(v) for v in (0.0, 1.0, 3.0, 4.5, 1.5, 0.5, 1.0 / 3.0, 2.0))


@cuda.jit(fastmath=True)
def _collide_stream(fa, fb, kind, sponge, nx, ny, nz, tau0, cs2smag, u_in, dx_, dy_, dz_, drho_src):
    """kind: 0 유체, 1 고체, 2 성문 음원 칸. 버퍼는 **편차** g = f − w (§52.528) — 계산도 모두 편차 꼴로 (f = w + g 로 되돌리면 float32 정밀도를 잃는다).
    음원 칸은 보통 유체로 충돌한 뒤 질량 Δρ (= U dt / dx³ / 칸 수) 와 운동량 Δρ·u_제트 를 더한다 — 부피 유량 원천 + 제트 운동량 (뒤는 단단한 벽)."""
    c = cuda.grid(1)
    N = nx * ny * nz
    if c >= N:
        return
    kd = kind[c]
    if kd == 1:
        return
    k = c % nz
    j = (c // nz) % ny
    i = c // (ny * nz)
    gin = cuda.local.array(19, float32)
    for q in range(19):
        si = i - EX[q]
        sj = j - EY[q]
        sk = k - EZ[q]
        if si < 0 or si >= nx or sj < 0 or sj >= ny or sk < 0 or sk >= nz:
            gin[q] = fa[OQ[q] * N + c]
        else:
            s = (si * ny + sj) * nz + sk
            if kind[s] == 1:
                gin[q] = fa[OQ[q] * N + c]          # 중간 되튐 (w_q = w_반대 라 편차끼리)
            else:
                gin[q] = fa[q * N + s]
    drho = F0
    jx = F0
    jy = F0
    jz = F0
    for q in range(19):
        drho += gin[q]
        jx += gin[q] * FX[q]
        jy += gin[q] * FY[q]
        jz += gin[q] * FZ[q]
    rho = F1 + drho
    ux = jx / rho
    uy = jy / rho
    uz = jz / rho
    uu = ux * ux + uy * uy + uz * uz
    pxx = F0
    pyy = F0
    pzz = F0
    pxy = F0
    pxz = F0
    pyz = F0
    geq = cuda.local.array(19, float32)
    for q in range(19):
        eu = FX[q] * ux + FY[q] * uy + FZ[q] * uz
        geq[q] = WQ[q] * (drho + rho * (F3 * eu + F45 * eu * eu - F15 * uu))     # f_eq − w
        d = gin[q] - geq[q]
        pxx += d * FX[q] * FX[q]
        pyy += d * FY[q] * FY[q]
        pzz += d * FZ[q] * FZ[q]
        pxy += d * FX[q] * FY[q]
        pxz += d * FX[q] * FZ[q]
        pyz += d * FY[q] * FZ[q]
    pm = math.sqrt(F2 * (pxx * pxx + pyy * pyy + pzz * pzz + F2 * (pxy * pxy + pxz * pxz + pyz * pyz))) + float32(1e-20)
    tau = FH * (tau0 + math.sqrt(tau0 * tau0 + cs2smag * pm / rho))
    om = F1 - F1 / tau
    sp = sponge[c]
    for q in range(19):
        reg = F45 * WQ[q] * ((FX[q] * FX[q] - F13) * pxx + (FY[q] * FY[q] - F13) * pyy + (FZ[q] * FZ[q] - F13) * pzz
                             + F2 * (FX[q] * FY[q] * pxy + FX[q] * FZ[q] * pxz + FY[q] * FZ[q] * pyz))
        gp = geq[q] + om * reg
        if kd == 2:
            gp = gp + WQ[q] * drho_src * (F1 + F3 * u_in * (FX[q] * dx_ + FY[q] * dy_ + FZ[q] * dz_))
        if sp > F0:
            gp = gp - sp * gp                       # 쉼 평형 (편차 0) 쪽으로
        fb[q * N + c] = gp


@cuda.jit(fastmath=True)
def _probe(f, idx, out, col, N):
    """칸 idx 의 밀도 · 운동량 x·y·z → out[:, col] (4 행 × 칸)."""
    p = cuda.grid(1)
    if p >= idx.size:
        return
    c = idx[p]
    rho = F0
    mx = F0
    my = F0
    mz = F0
    for q in range(19):
        v = f[q * N + c]                          # 편차 g = f − w — 합이 ρ − 1, 운동량은 Σ e g (Σ e w = 0)
        rho += v
        mx += v * FX[q]
        my += v * FY[q]
        mz += v * FZ[q]
    out[0, p, col] = rho
    out[1, p, col] = mx
    out[2, p, col] = my
    out[3, p, col] = mz


class LBMCuda:
    def __init__(self, kind: np.ndarray, dx: float, c_phys: float = 357.0, nu_phys: float = 1.65e-5, rho0: float = 1.12,
                 sponge: np.ndarray | None = None, cs_smag: float = 0.17, inlet_dir=(0.0, 1.0, 0.0)):
        self.shape = kind.shape
        self.N = int(np.prod(kind.shape))
        self.dx = dx
        self.dt = dx / (math.sqrt(3.0) * c_phys)
        self.u_scale = dx / self.dt
        self.rho0 = rho0
        self.nu_lat = nu_phys * self.dt / dx ** 2
        self.tau0 = 0.5 + 3.0 * self.nu_lat
        self.cs2smag = 18.0 * math.sqrt(2.0) * cs_smag ** 2
        d = np.asarray(inlet_dir, float)
        self.dir = d / np.linalg.norm(d)
        self.kind = cuda.to_device(np.ascontiguousarray(kind.ravel().astype(np.int32)))
        sp = np.zeros(self.N, np.float32) if sponge is None else np.ascontiguousarray(sponge.ravel().astype(np.float32))
        self.sponge = cuda.to_device(sp)
        # 편차 저장 (§52.528): 버퍼는 g = f − w — 밀도 ≈ 1 위의 작은 요동을 float32 로 바로 담는다 (f 를 그대로 담으면 고역 바닥이 −68 dB re Pa²/Hz 에 깔렸다)
        g0 = np.zeros(19 * self.N, np.float32)
        self.fa = cuda.to_device(g0)
        self.fb = cuda.to_device(g0.copy())
        self.tpb = 128
        self.blocks = (self.N + self.tpb - 1) // self.tpb

    def set_source(self, n_cells: int, area_m2: float):
        """성문 음원 칸 수와 틈 넓이 — step 의 입력을 부피 유량 [m³/s] 으로 받는다."""
        self.n_src, self.A_src = int(n_cells), float(area_m2)

    def step(self, U_phys: float):
        """U_phys: 성문 부피 유량 [m³/s]. 칸마다 Δρ = U dt / (dx³ n_칸), 제트 속도 U / A_틈 (격자 단위)."""
        U = float(U_phys)
        drho = np.float32(U * self.dt / (self.dx ** 3 * max(getattr(self, "n_src", 1), 1)))
        u = np.float32(U / max(getattr(self, "A_src", 1.0), 1e-12) / self.u_scale)
        nx, ny, nz = self.shape
        _collide_stream[self.blocks, self.tpb](self.fa, self.fb, self.kind, self.sponge, nx, ny, nz, np.float32(self.tau0), np.float32(self.cs2smag), u,
                                               np.float32(self.dir[0]), np.float32(self.dir[1]), np.float32(self.dir[2]), drho)
        self.fa, self.fb = self.fb, self.fa

    def run(self, u_in: np.ndarray, probe_cells: np.ndarray, every: int = 1):
        """u_in (n,) [m/s] 걸음마다. probe_cells: 평탄 칸 번호. 반환 (4, P, n//every): 밀도 · 운동량 (격자 단위)."""
        idx = cuda.to_device(np.ascontiguousarray(probe_cells.astype(np.int64)))
        n = len(u_in)
        m = (n + every - 1) // every
        out = cuda.device_array((4, len(probe_cells), m), np.float32)
        pb = (len(probe_cells) + 127) // 128
        for t in range(n):
            self.step(u_in[t])
            if t % every == 0 and t // every < m:
                _probe[pb, 128](self.fa, idx, out, t // every, self.N)
        cuda.synchronize()
        return out.copy_to_host()

    def pressure(self, drho_lat):
        """음향 압력 [Pa] — 탐침의 밀도 편차 (ρ − 1, 격자) → (ρ − 1) c_s² ρ0 (dx/dt)²."""
        return drho_lat / 3.0 * self.rho0 * self.u_scale ** 2
