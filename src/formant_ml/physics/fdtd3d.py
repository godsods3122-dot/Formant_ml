"""3 차원 성도 음향 — 엇갈린 직교 격자 유한 차분 시간 영역 (GPU, torch) (MEASUREMENTS §52.526).

선형 음향 (CGS): ∂u/∂t = −∇p/ρ, ∂p/∂t = −ρc² ∇·u. 압력은 칸 가운데, 속도는 면 가운데 (Yee). 칸 사이 면의 흐름은 열린 비율 a ∈ [0, 1] × h² × u
(`tract3d.Grid`) — 격자보다 좁은 협착의 넓이를 담는다. 고체 쪽 면은 벽 면이고 벽마다 두 흐름이 공기 칸에서 빠져나간다:

* **경계층 (점성·열)**: 국소 반응 어드미턴스 v = √(jω)·(√ν + (γ−1)√κ)/(ρc²) · p — 관 벽을 따라가는 평면파의 Kirchhoff 감쇠와 같은 꼴
  (Re β = √(ω/2)(√ν + (γ−1)√κ)/c). √(jω) 는 확산 표현 Σ w_m (x − φ_m) (극 넷, `tube_td.diffusive` 와 같은 꼴, 이 표본률로 맞춘다).
  단단한 벽(치아·경구개)과 무른 벽 모두.
* **무른 벽**: 단위 넓이 질량–저항–강성 m ξ̈ + r ξ̇ + k ξ = p (조직별, `TISSUE_WALL`). 사용자: *"경구개, 치아를 제외하면 다 부드럽다는 것도 고려해야
  하고, 진동이 성도벽으로부터 빼앗기는 에너지, 성도벽이 진동할 때 나는 소리도 고려해야 해"*. 벽 속도는 기록해 벽 방사(다음 단계)에 쓴다.

바깥 공기의 상자 끝은 흡수층(σ 이 2 차로 커지는 감쇠, 압력·속도에 같은 σ — 수직 입사에서 반사 없음). 음원: 첫 띠(성문 끝) 칸들에 부피 유량.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import air as _air
from .tract3d import Grid, T_ID, TISSUE

#: 조직별 벽 (단위 넓이) — (두께 [cm], 공명 [Hz], Q). 질량 m = ρ_조직 × 두께, 강성 k = m (2π f)², 저항 r = 2π f m / Q.
#: Hanna, Smith & Wolfe (2016, JASA 139:2924) Table III 여성 3 명의 전역 등가 벽 공명은 19 Hz, Q = 1.0 이다 (닫힌 성문, 입술 임피던스).
#: Q ≈ 0.5 는 같은 표의 Ishizaka (1975) 비교값이다. 아래 Q = 0.5 와 국소 두께는 잠정 모형값이지 Hanna 의 조직별 측정값이 아니다.
#: 조직마다의 두께는 해부 문헌 어림 (볼 1.2, 연구개 0.9, 입술 1.0, 혀 2.0 (두꺼운 근육 — 표면 진동의 실효 깊이), 후두개 0.3 (양쪽이
#: 공기), 인두 0.6 (척추 앞 연조직 — 뼈에 받쳐 더 단단: 공명 40 Hz), 입 바닥·잇몸 1.0). **잠정값** — 조직별 실측(초음파 두께, 외부 가진)으로
#: 바꿔야 하고, 3D 모형 전체가 Hanna 의 여성 R0·R0′·대역폭을 내는지로 검증한다.
TISSUE_WALL = {
    "tongue": (2.0, 19.0, 0.5),
    "velum": (0.9, 19.0, 0.5),
    "pharynx": (0.6, 40.0, 0.5),
    "cheek": (1.2, 19.0, 0.5),
    "lip": (1.0, 19.0, 0.5),
    "epiglottis": (0.3, 19.0, 0.5),
    "jaw_cover": (1.0, 19.0, 0.5),
}
RHO_TISSUE = 1.05      # g/cm³


def diffusive_weights(fs: float, poles_hz=(30.0, 330.0, 2600.0, 2.0e4, 2.0e5), fmin=50.0, fmax=20000.0):
    """√(jω) ≈ Σ w_m (x − φ_m), φ_m ← φ_m + a_m (x − φ_m) 의 (a_m, w_m) — 비음수 최소제곱 (상대 오차)."""
    from scipy.optimize import nnls
    dt = 1.0 / fs
    f = np.geomspace(fmin, fmax, 500)
    z = np.exp(2j * np.pi * f * dt)
    tgt = np.sqrt(2j * np.pi * f)
    a = 1.0 - np.exp(-2.0 * np.pi * np.asarray(poles_hz) * dt)
    H = np.stack([(z - 1.0) / (z - 1.0 + am) for am in a], 1) / np.abs(tgt)[:, None]
    t = tgt / np.abs(tgt)
    w, _ = nnls(np.vstack([H.real, H.imag]), np.concatenate([t.real, t.imag]))
    Hf = (np.stack([(z - 1.0) / (z - 1.0 + am) for am in a], 1) @ w)
    err = np.abs(Hf / tgt - 1.0)
    return a, w, float(err.max())


@dataclass
class Result:
    fs: float
    p_rec: np.ndarray        # (n_rec, T) 수신점 압력 [dyn/cm²]
    p_in: np.ndarray         # (T,) 음원 칸 평균 압력
    u_src: np.ndarray        # (T,) 넣은 부피 유량 [cm³/s]
    meta: dict


class Sim:
    def __init__(self, g: Grid, T_c: float = 35.5, rh: float = 1.0, courant: float = 0.5, sponge_cells: int = 20,
                 sponge_sigma: float | None = None, walls: str = "soft", boundary_layer: bool = True, device: str = "cuda",
                 dtype=torch.float32):
        a = _air.props(T_c, rh).cgs()
        self.rho, self.c = a["RHO"], a["C_SOUND"]
        nu = a["MU"] / a["RHO"]
        kappa = a["LAMBDA_TH"] / (a["RHO"] * a["CP"])
        self.g = g
        h = g.h
        self.dt = courant * h / (self.c * math.sqrt(3.0))
        self.fs = 1.0 / self.dt
        dev = torch.device(device)
        self.dev, self.dtype = dev, dtype
        T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev, dtype=dtype)
        self.air = T(g.air.astype(np.float32))
        self.ax, self.ay, self.az = T(g.ax), T(g.ay), T(g.az)
        self.p = torch.zeros(g.shape, device=dev, dtype=dtype)
        nx, ny, nz = g.shape
        self.ux = torch.zeros((nx + 1, ny, nz), device=dev, dtype=dtype)
        self.uy = torch.zeros((nx, ny + 1, nz), device=dev, dtype=dtype)
        self.uz = torch.zeros((nx, ny, nz + 1), device=dev, dtype=dtype)
        # ---- 흡수층: **분할 PML** (Berenger) — p = p_x + p_y + p_z, ∂p_a/∂t + σ_a p_a = −ρc² ∂(a·u_a)/∂a, ∂u_a/∂t + σ_a u_a = −∂p/∂a /ρ.
        # 연속 매질에서 모든 각도·주파수에 반사가 없다. 예전의 압력 전체 감쇠(스펀지)는 비스듬한 파·저역에서 정합이 깨져 250 Hz 아래에서 상자가
        # 닫힌 공동처럼 굴었다 (상반성 G 가 저역에서 평평). σ_a(a) = σ_max (깊이/L)², σ_max = 12 c/L (∫σ = 4c → 왕복 e^−8), 바깥 공기 칸에만.
        self.pml = sponge_cells
        smax = sponge_sigma if sponge_sigma is not None else (12.0 * self.c / (max(sponge_cells, 1) * h))
        ext = g.exterior.astype(np.float32)
        sig_c, sig_f = [], []
        for ax_ in range(3):
            n = g.shape[ax_]
            d = np.minimum(np.arange(n) + 0.5, n - 0.5 - np.arange(n))                 # 칸 가운데의 끝까지 깊이 [칸]
            prof = np.clip((sponge_cells - d) / max(sponge_cells, 1), 0.0, 1.0) ** 2 * smax if sponge_cells > 0 else np.zeros(n)
            df = np.minimum(np.arange(n + 1), n - np.arange(n + 1)).astype(float)       # 면
            proff = np.clip((sponge_cells - df) / max(sponge_cells, 1), 0.0, 1.0) ** 2 * smax if sponge_cells > 0 else np.zeros(n + 1)
            shp = [1, 1, 1]
            shp[ax_] = n
            sc = prof.reshape(shp) * ext
            shf = [1, 1, 1]
            shf[ax_] = n + 1
            ef = np.zeros([n + 1 if a == ax_ else g.shape[a] for a in range(3)], np.float32)
            sl = [slice(None)] * 3
            sl[ax_] = slice(1, -1)
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[ax_] = slice(0, -1)
            sl1[ax_] = slice(1, None)
            ef[tuple(sl)] = np.maximum(ext[tuple(sl0)], ext[tuple(sl1)])
            sf = proff.reshape(shf) * ef
            sig_c.append(T(np.exp(-sc * self.dt)))
            sig_f.append(T(np.exp(-sf * self.dt)))
        self.dpx, self.dpy, self.dpz = sig_c
        self.damp_x, self.damp_y, self.damp_z = sig_f
        self.px = torch.zeros_like(self.p)
        self.py = torch.zeros_like(self.p)
        self.pz = torch.zeros_like(self.p)
        self.openx, self.openy, self.openz = (self.ax > 0).to(dtype), (self.ay > 0).to(dtype), (self.az > 0).to(dtype)
        # ---- 벽 면 목록: (공기 칸 번호, 조직)
        cells, tis, cosw, wpos, wdir = [], [], [], [], []
        for axis, (W, Cw) in enumerate(((g.wx, g.cx), (g.wy, g.cy), (g.wz, g.cz))):
            idx = np.argwhere(W > 0)
            if idx.size == 0:
                continue
            t = W[tuple(idx.T)]
            # 공기 칸: 면 (i) 은 칸 i−1 과 i 사이
            lo = idx.copy()
            lo[:, axis] -= 1
            hi = idx.copy()
            okL = (lo[:, axis] >= 0)
            okH = (hi[:, axis] < g.shape[axis])
            airL = np.zeros(len(idx), bool)
            airH = np.zeros(len(idx), bool)
            airL[okL] = g.air[tuple(lo[okL].T)]
            airH[okH] = g.air[tuple(hi[okH].T)]
            cell = np.where(airL[:, None], lo, hi)
            keep = airL ^ airH
            cells.append(np.ravel_multi_index(tuple(cell[keep].T), g.shape))
            tis.append(t[keep])
            cosw.append(Cw[tuple(idx[keep].T)])
            fc = g.origin + (idx[keep] + 0.5) * h
            fc[:, axis] -= 0.5 * h
            wpos.append(fc)
            d = np.zeros((int(keep.sum()), 3))
            d[:, axis] = np.where(airL[keep], 1.0, -1.0)          # 공기 → 벽 (바깥) 방향
            wdir.append(d)
        cells = np.concatenate(cells) if cells else np.zeros(0, int)
        tis = np.concatenate(tis) if tis else np.zeros(0, np.int8)
        cosw = np.concatenate(cosw) if cosw else np.zeros(0, np.float32)
        self.wall_pos = np.concatenate(wpos) if wpos else np.zeros((0, 3))
        self.wall_dir = np.concatenate(wdir) if wdir else np.zeros((0, 3))
        self.cut = bool(g.meta.get("cut", False))
        if self.cut:                                   # 절단 칸 (`tract3d_cut`): 벽은 칸마다, 넓이는 고운 경계 면에서
            W_ = g.meta["walls"]
            cells, tis = np.asarray(W_["cell"]), np.asarray(W_["tis"])
            cosw = (np.asarray(W_["area"]) / (h * h)).astype(np.float32)
            self.wall_pos, self.wall_dir = np.asarray(W_["pos"]), np.asarray(W_["dir"])
        self.n_wall = len(cells)
        self.wall_cell = torch.as_tensor(cells, device=dev, dtype=torch.long)
        self.wall_tis = tis
        self.wall_area = {TISSUE[t]: float(cosw[tis == t].sum() * h * h) for t in np.unique(tis)}
        self.wall_area_stair = {TISSUE[t]: float((tis == t).sum() * h * h) for t in np.unique(tis)}
        self.wall_cos = torch.as_tensor(cosw, device=dev, dtype=dtype)
        # 무른 벽 계수 (단단·배플·성문 끝은 움직이지 않는다)
        m = np.zeros(len(cells), np.float64)
        r = np.zeros(len(cells), np.float64)
        k = np.zeros(len(cells), np.float64)
        soft = np.zeros(len(cells), bool)
        if walls == "soft":
            for name, (th, f0, Q) in TISSUE_WALL.items():
                sel = tis == T_ID[name]
                mm = RHO_TISSUE * th
                m[sel], k[sel], r[sel] = mm, mm * (2 * math.pi * f0) ** 2, 2 * math.pi * f0 * mm / Q
                soft[sel] = True
        self.soft = torch.as_tensor(soft, device=dev)
        dt = self.dt
        den = m / dt + r / 2
        self.w_c1 = T(np.where(soft, (m / dt - r / 2) / np.where(soft, den, 1), 0))
        self.w_c2 = T(np.where(soft, 1.0 / np.where(soft, den, 1), 0))
        self.w_k = T(k)
        self.vw = torch.zeros(len(cells), device=dev, dtype=torch.float64 if dtype == torch.float64 else dtype)
        self.xw = torch.zeros_like(self.vw)
        # 경계층
        self.bl = boundary_layer
        if boundary_layer:
            aa, ww, err = diffusive_weights(self.fs)
            self.bl_a = [float(x) for x in aa]
            self.bl_w = [float(x) for x in ww]
            self.bl_err = err
            self.bl_C = (math.sqrt(nu) + (a["GAMMA"] - 1.0) * math.sqrt(kappa)) / (self.rho * self.c ** 2)
            self.phi = [torch.zeros(len(cells), device=dev, dtype=dtype) for _ in aa]
        # 음원 칸
        self.inlet = torch.as_tensor(np.ravel_multi_index(np.nonzero(g.inlet), g.shape), device=dev, dtype=torch.long)
        self.kp = self.dt * self.rho * self.c ** 2 / h
        self.ku = self.dt / (self.rho * h)
        if self.cut:
            vf = np.asarray(g.meta["vf"], np.float32)
            veff = stable_volume(vf, g.ax, g.ay, g.az, courant)
            self.veff = veff
            self.inv_v = T(np.where(veff > 0, 1.0 / np.maximum(veff, 1e-6), 0.0))
            vin = vf[g.inlet]
            self.v_inlet = float(vin.sum()) * h ** 3
            # 음원 칸마다 dp = dt ρc² u · (V_칸/ΣV) / (V_eff,칸 h³)
            self.src_w = T((vin / vin.sum()) / np.maximum(veff[g.inlet], 1e-6) * vin.sum())
        else:
            self.inv_v = None
            self.v_inlet = float(g.inlet.sum()) * h ** 3
            self.src_w = None
        self.wall_iv = self.inv_v.view(-1)[self.wall_cell] if (self.cut and self.n_wall) else None

    def set_flux_faces(self, faces: list):
        """면 유량을 기록할 면들 — [(축, 면 번호 (n, 3), 부호 (n,)), …]. 유량 = 부호 · a·u·h² [cm³/s] (부호 +1 = +축 방향이 바깥)."""
        self.flux_sets = []
        for axis, idx, sign in faces:
            shp = (self.ux, self.uy, self.uz)[axis].shape
            fi = torch.as_tensor(np.ravel_multi_index(tuple(np.asarray(idx).T), shp), device=self.dev)
            w = (self.ax, self.ay, self.az)[axis].reshape(-1)[fi] * self.g.h ** 2 * torch.as_tensor(np.asarray(sign, np.float32), device=self.dev)
            self.flux_sets.append((axis, fi, w))
        self.n_flux = sum(len(x[1]) for x in self.flux_sets)

    def set_wall_groups(self, group: np.ndarray, n_groups: int):
        """무른 벽 면마다 무리 번호 (−1 = 기록 안 함) — 걸음마다 무리별 벽 부피 유량 Σ v·넓이 [cm³/s] (바깥으로 +) 을 기록한다."""
        group = np.asarray(group)
        m = group >= 0
        self.wg_sel = torch.as_tensor(np.flatnonzero(m), device=self.dev)
        self.wg_id = torch.as_tensor(group[m], device=self.dev, dtype=torch.long)
        self.wg_n = int(n_groups)
        self.wg_area = (self.wall_cos[self.wg_sel] * self.g.h ** 2)

    def step(self, u_src: float):
        p, h = self.p, self.g.h
        self.ux[1:-1] -= self.ku * (p[1:] - p[:-1])
        self.uy[:, 1:-1] -= self.ku * (p[:, 1:] - p[:, :-1])
        self.uz[:, :, 1:-1] -= self.ku * (p[:, :, 1:] - p[:, :, :-1])
        self.ux *= self.openx * self.damp_x
        self.uy *= self.openy * self.damp_y
        self.uz *= self.openz * self.damp_z
        fx, fy, fz = self.ax * self.ux, self.ay * self.uy, self.az * self.uz
        if self.inv_v is None:
            self.px -= self.kp * (fx[1:] - fx[:-1])
            self.py -= self.kp * (fy[:, 1:] - fy[:, :-1])
            self.pz -= self.kp * (fz[:, :, 1:] - fz[:, :, :-1])
        else:
            self.px -= self.kp * self.inv_v * (fx[1:] - fx[:-1])
            self.py -= self.kp * self.inv_v * (fy[:, 1:] - fy[:, :-1])
            self.pz -= self.kp * self.inv_v * (fz[:, :, 1:] - fz[:, :, :-1])
        # 벽: 칸 압력 → 벽 속도 (바깥으로 +) → 칸에서 빠지는 흐름 (벽·음원은 흡수층 밖이라 성분 하나에 넣는다)
        if self.n_wall:
            pw = p.view(-1)[self.wall_cell]
            vout = torch.zeros_like(pw)
            if self.soft.any():
                vw_new = self.w_c1 * self.vw + self.w_c2 * (pw - self.w_k * self.xw)
                self.vw = vw_new
                self.xw = self.xw + self.dt * vw_new
                vout = vout + vw_new
            if self.bl:
                acc = torch.zeros_like(pw)
                for m_, (am, wm) in enumerate(zip(self.bl_a, self.bl_w)):
                    acc = acc + wm * (pw - self.phi[m_])
                    self.phi[m_] = self.phi[m_] + am * (pw - self.phi[m_])
                vout = vout + self.bl_C * acc
            kw = self.kp * self.wall_cos if self.wall_iv is None else self.kp * self.wall_cos * self.wall_iv
            self.px.view(-1).index_add_(0, self.wall_cell, -kw * vout)
        if u_src != 0.0:
            k_ = self.dt * self.rho * self.c ** 2 * u_src / self.v_inlet
            if self.src_w is None:
                self.px.view(-1).index_add_(0, self.inlet, torch.full((self.inlet.numel(),), k_, device=self.dev, dtype=self.dtype))
            else:
                self.px.view(-1).index_add_(0, self.inlet, k_ * self.src_w)
        self.px *= self.air * self.dpx
        self.py *= self.air * self.dpy
        self.pz *= self.air * self.dpz
        torch.add(self.px, self.py, out=self.p)
        self.p += self.pz

    def _step_t(self, u_t: torch.Tensor):
        """한 걸음 — 제자리 연산만 (CUDA 그래프에 담을 수 있게). u_t: 0 차원 텐서 [cm³/s]."""
        p = self.p
        self.ux[1:-1].sub_(self.ku * (p[1:] - p[:-1]))
        self.uy[:, 1:-1].sub_(self.ku * (p[:, 1:] - p[:, :-1]))
        self.uz[:, :, 1:-1].sub_(self.ku * (p[:, :, 1:] - p[:, :, :-1]))
        self.ux.mul_(self.mx)
        self.uy.mul_(self.my)
        self.uz.mul_(self.mz)
        kv = self.kpv
        self.px.sub_(kv * (self.ax[1:] * self.ux[1:] - self.ax[:-1] * self.ux[:-1]))
        self.py.sub_(kv * (self.ay[:, 1:] * self.uy[:, 1:] - self.ay[:, :-1] * self.uy[:, :-1]))
        self.pz.sub_(kv * (self.az[:, :, 1:] * self.uz[:, :, 1:] - self.az[:, :, :-1] * self.uz[:, :, :-1]))
        if self.n_wall:
            pw = p.view(-1)[self.wall_cell]
            vout = torch.zeros_like(pw)
            if self.has_soft:
                vw_new = self.w_c1 * self.vw + self.w_c2 * (pw - self.w_k * self.xw)
                self.vw.copy_(vw_new)
                self.xw.add_(self.dt * vw_new)
                vout = vout + vw_new
            if self.bl:
                acc = torch.zeros_like(pw)
                for m_, (am, wm) in enumerate(zip(self.bl_a, self.bl_w)):
                    d_ = pw - self.phi[m_]
                    acc = acc + wm * d_
                    self.phi[m_].add_(am * d_)
                vout = vout + self.bl_C * acc
            self.px.view(-1).index_add_(0, self.wall_cell, -self.kw * vout)
        self.px.view(-1).index_add_(0, self.inlet, (self.src_coef * u_t) * self.src_vec)
        self.px.mul_(self.mpx)
        self.py.mul_(self.mpy)
        self.pz.mul_(self.mpz)
        torch.add(self.px, self.py, out=self.p)
        self.p.add_(self.pz)

    def _prep_t(self):
        """`_step_t` 가 쓰는 묶은 계수 (한 번)."""
        if getattr(self, "_prepped", False):
            return
        self.mx, self.my, self.mz = self.openx * self.damp_x, self.openy * self.damp_y, self.openz * self.damp_z
        self.kpv = self.kp * (self.inv_v if self.inv_v is not None else torch.ones_like(self.p))
        self.mpx, self.mpy, self.mpz = self.air * self.dpx, self.air * self.dpy, self.air * self.dpz
        self.has_soft = bool(self.soft.any().item()) if self.n_wall else False
        if self.n_wall:
            self.kw = self.kp * self.wall_cos if self.wall_iv is None else self.kp * self.wall_cos * self.wall_iv
        self.src_coef = self.dt * self.rho * self.c ** 2 / self.v_inlet
        self.src_vec = self.src_w if self.src_w is not None else torch.ones(self.inlet.numel(), device=self.dev, dtype=self.dtype)
        self._prepped = True

    def _state(self):
        st = [self.p, self.px, self.py, self.pz, self.ux, self.uy, self.uz]
        if self.n_wall:
            st += [self.vw, self.xw]
            if self.bl:
                st += list(self.phi)
        return st

    @torch.no_grad()
    def run_graph(self, u_src: np.ndarray, rec_points: list, block: int = 128) -> Result:
        """`run` 과 같은 결과를 CUDA 그래프로 (block 걸음을 한 그래프에) — 파이썬 호출 부담을 없앤다."""
        self._prep_t()
        idx = torch.as_tensor([np.ravel_multi_index(self.g.index(q), self.g.shape) for q in rec_points], device=self.dev)
        n = len(u_src)
        nb = (n + block - 1) // block
        u_all = torch.zeros(nb * block, device=self.dev, dtype=self.dtype)
        u_all[:n] = torch.as_tensor(np.asarray(u_src), device=self.dev, dtype=self.dtype)
        u_blk = torch.zeros(block, device=self.dev, dtype=self.dtype)
        o_blk = torch.zeros((len(rec_points), block), device=self.dev, dtype=self.dtype)
        pin_blk = torch.zeros(block, device=self.dev, dtype=self.dtype)
        fs_ = getattr(self, "flux_sets", None)
        fl_blk = torch.zeros((self.n_flux, block), device=self.dev, dtype=self.dtype) if fs_ else None
        wg = getattr(self, "wg_sel", None)
        wg_blk = torch.zeros((self.wg_n, block), device=self.dev, dtype=self.dtype) if wg is not None else None

        def body():
            for k in range(block):
                self._step_t(u_blk[k])
                pf = self.p.view(-1)
                o_blk[:, k] = pf[idx]
                pin_blk[k] = pf[self.inlet].mean()
                if fl_blk is not None:
                    k0 = 0
                    for axis, fi, w in fs_:
                        fl_blk[k0:k0 + fi.numel(), k] = (self.ux, self.uy, self.uz)[axis].reshape(-1)[fi] * w
                        k0 += fi.numel()
                if wg_blk is not None:
                    acc = torch.zeros(self.wg_n, device=self.dev, dtype=self.dtype)
                    acc.index_add_(0, self.wg_id, self.vw[wg] * self.wg_area)
                    wg_blk[:, k] = acc

        state = [t.clone() for t in self._state()]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        for t, t0 in zip(self._state(), state):
            t.copy_(t0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        for t, t0 in zip(self._state(), state):
            t.copy_(t0)
        out = torch.zeros((len(rec_points), nb * block), device=self.dev, dtype=torch.float64)
        pin = torch.zeros(nb * block, device=self.dev, dtype=torch.float64)
        flux = torch.zeros((self.n_flux, nb * block), device=self.dev, dtype=torch.float32) if fl_blk is not None else None
        wgo = torch.zeros((self.wg_n, nb * block), device=self.dev, dtype=torch.float32) if wg_blk is not None else None
        for b_ in range(nb):
            u_blk.copy_(u_all[b_ * block:(b_ + 1) * block])
            graph.replay()
            out[:, b_ * block:(b_ + 1) * block] = o_blk
            pin[b_ * block:(b_ + 1) * block] = pin_blk
            if flux is not None:
                flux[:, b_ * block:(b_ + 1) * block] = fl_blk
            if wgo is not None:
                wgo[:, b_ * block:(b_ + 1) * block] = wg_blk
        meta = dict(n_wall=self.n_wall, wall_area=self.wall_area)
        if flux is not None:
            meta["flux"] = flux[:, :n].cpu().numpy()
        if wgo is not None:
            meta["wall_groups"] = wgo[:, :n].cpu().numpy()
        return Result(self.fs, out[:, :n].cpu().numpy(), pin[:n].cpu().numpy(), np.asarray(u_src), meta)

    @torch.no_grad()
    def run(self, u_src: np.ndarray, rec_points: list, record_every: int = 1) -> Result:
        idx = torch.as_tensor([np.ravel_multi_index(self.g.index(q), self.g.shape) for q in rec_points], device=self.dev)
        n = len(u_src)
        out = torch.zeros((len(rec_points), n), device=self.dev, dtype=torch.float64)
        pin = torch.zeros(n, device=self.dev, dtype=torch.float64)
        fs_ = getattr(self, "flux_sets", None)
        flux = torch.zeros((self.n_flux, n), device=self.dev, dtype=torch.float32) if fs_ else None
        wg = getattr(self, "wg_sel", None)
        wgo = torch.zeros((self.wg_n, n), device=self.dev, dtype=torch.float32) if wg is not None else None
        for t in range(n):
            self.step(float(u_src[t]))
            pf = self.p.view(-1)
            out[:, t] = pf[idx].double()
            pin[t] = pf[self.inlet].double().mean()
            if flux is not None:
                k0 = 0
                for axis, fi, w in fs_:
                    flux[k0:k0 + fi.numel(), t] = (self.ux, self.uy, self.uz)[axis].reshape(-1)[fi] * w
                    k0 += fi.numel()
            if wgo is not None:
                acc = torch.zeros(self.wg_n, device=self.dev, dtype=self.vw.dtype)
                acc.index_add_(0, self.wg_id, self.vw[wg] * self.wg_area)
                wgo[:, t] = acc
        meta = dict(n_wall=self.n_wall, wall_area=self.wall_area)
        if flux is not None:
            meta["flux"] = flux.cpu().numpy()
        if wgo is not None:
            meta["wall_groups"] = wgo.cpu().numpy()
        return Result(self.fs, out.cpu().numpy(), pin.cpu().numpy(), np.asarray(u_src), meta)


def gauss_pulse(fs: float, fmax: float = 20000.0, n: int = 1000):
    """대역 제한 부피 유량 펄스 (가우스, −40 dB @ fmax) [cm³/s] — 넓이 1 cm³ 가 되게."""
    sig = math.sqrt(2 * math.log(100.0)) / (2 * math.pi * fmax)
    t0 = 5 * sig
    t = np.arange(n) / fs
    u = np.exp(-0.5 * ((t - t0) / sig) ** 2)
    return u / (u.sum() / fs), t0


def zero_net_pulse(fs: float, fmax: float = 20000.0, n: int = 1000, sigma2_ms: float = 1.5):
    """순 부피 0 의 펄스 — 좁은 가우스(−40 dB @ fmax) − 같은 넓이의 넓은 가우스(σ₂). 닫힌 상자·흡수층에 남는 정압(넣은 질량)이 저역 전달을
    흐리지 않게 (상반성 계산의 150 Hz 아래가 평평했다). ~1/(2πσ₂) ≈ 100 Hz 아래는 음원이 약해 쓰지 않는다."""
    u1, t0 = gauss_pulse(fs, fmax, n)
    s2 = sigma2_ms * 1e-3
    t = np.arange(n) / fs
    tc = max(t0, 4 * s2)
    u1 = np.roll(u1, int((tc - t0) * fs))
    u2 = np.exp(-0.5 * ((t - tc) / s2) ** 2)
    u2 *= u1.sum() / u2.sum()
    return u1 - u2, tc


def stable_volume(vf: np.ndarray, ax: np.ndarray, ay: np.ndarray, az: np.ndarray, courant: float, iters: int = 6) -> np.ndarray:
    """절단 칸의 실효 부피 — 안정에 꼭 필요한 만큼만 공기 몫 V 를 늘린다 (§52.526). 엇갈린 격자 도약 갱신은 (c dt/h)² λ_max ≤ 4, 대칭화한
    연산자 V^{-½} A V^{-½} 의 게르시고린 상한 B_i = Σ_f a_f (1/V_i + 1/√(V_i V_j)) 로 B_i ≤ 12/κ² (κ = 쿠랑 수 / (1/√3)). 꽉 찬 칸은 B = 12.
    예전의 V_eff = max(V, 면 열림 최대) 는 경계 칸 부피를 지나치게 늘려 원통 공명이 2–3 % 낮았다."""
    Bmax = 12.0 / courant ** 2
    V = np.where(vf > 0, vf, 0.0).astype(np.float64)
    A = [np.asarray(ax, np.float64), np.asarray(ay, np.float64), np.asarray(az, np.float64)]
    for _ in range(iters):
        sa = np.zeros_like(V)
        sb = np.zeros_like(V)
        Vs = np.sqrt(np.maximum(V, 1e-12))
        for axis in range(3):
            a_ = A[axis]
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[axis], sl_hi[axis] = slice(0, -1), slice(1, None)
            a_lo, a_hi = a_[tuple(sl_lo)], a_[tuple(sl_hi)]          # 칸의 아래·위 면
            sa += a_lo + a_hi
            # 이웃 칸의 √V
            nb_lo = np.zeros_like(V)
            nb_hi = np.zeros_like(V)
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[axis], s1[axis] = slice(1, None), slice(0, -1)
            nb_lo[tuple(s0)] = Vs[tuple(s1)]
            nb_hi[tuple(s1)] = Vs[tuple(s0)]
            sb += a_lo / np.maximum(nb_lo, 1e-6) * (nb_lo > 0) + a_hi / np.maximum(nb_hi, 1e-6) * (nb_hi > 0)
        # B(V) = sa/V + sb/√V ≤ Bmax → √V ≥ (sb + √(sb² + 4 Bmax sa)) / (2 Bmax)
        root = (sb + np.sqrt(sb * sb + 4 * Bmax * sa)) / (2 * Bmax)
        Vn = np.where(vf > 0, np.maximum(vf, root ** 2), 0.0)
        if np.allclose(Vn, V, rtol=1e-4, atol=1e-7):
            V = Vn
            break
        V = Vn
    return V.astype(np.float32)
