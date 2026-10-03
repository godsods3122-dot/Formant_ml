"""절단 칸 격자 (MEASUREMENTS §52.526) — 앞니·입술 둘레의 1–2 mm 틈.

1 mm 칸 가운데만 보는 격자(`tract3d.build_grid`)는 앞니 사이 0.15 cm² 틈(높이 ~1.5 mm)을 놓쳐 성도가 끊겼다 (m7 의 ɨ · iː · ɐ · ʌ 틀, 성문 쪽 공기가
x ≈ 4.1 cm 앞니에서 멈춤). 여기서는 칸마다 고운 표본(h/sub, 기본 sub = 4 → 64 개)으로:

* 공기 몫 V (칸 안 공기 부피 / h³). 공기 칸 = V > 0.
* 면 열림 a = 면 양쪽 고운 칸이 모두 공기인 고운 기둥의 몫.
* 실효 부피 V_eff = max(V, 칸 면 열림의 최대) — 압력 갱신을 V_eff 로 나눈다. a/V_eff ≤ 1 이라 쿠랑 조건이 그대로다 (작은 칸 문제를 피하는 대가로
  그런 칸의 부피를 조금 늘린다).
* 벽은 칸마다: 고운 경계 면의 넓이 × |n·e| (σ 1.5 고운 칸으로 고른 장의 법선) 를 공기 쪽 굵은 칸에 모은다. 조직은 가장 가까운 고체 굵은 칸의 조직
  (`build_grid` 의 면 조직), 바깥 법선은 칸 안 경계 면들의 합.
"""
from __future__ import annotations

import numpy as np

from .tract3d import Grid, _fill_surf, build_grid


def _fine_occupancy(cp: dict, lo: np.ndarray, shape_f: tuple, hf: float, ep, ed, device="cuda"):
    """고운 격자(간격 hf, 원점 lo, 모양 shape_f) 의 (성도 공기, 바깥 공기, 첫 띠) — torch bool."""
    import torch
    dev = torch.device(device)
    dt = torch.float32
    F = torch.zeros(shape_f, dtype=torch.bool, device=dev)
    Fin = torch.zeros(shape_f, dtype=torch.bool, device=dev)
    cen, nor = cp["center"], cp["normal"]
    zs = cp["z"]
    dz = zs[1] - zs[0]
    U = torch.as_tensor(np.nan_to_num(cp["upper"], nan=-1e3), device=dev, dtype=dt)
    L = torch.as_tensor(np.nan_to_num(cp["lower"], nan=1e3), device=dev, dtype=dt)
    valid = torch.as_tensor(np.isfinite(cp["upper"]) & np.isfinite(cp["lower"]), device=dev)
    zf = torch.as_tensor(lo[2] + (np.arange(shape_f[2]) + 0.5) * hf, device=dev, dtype=dt)
    kf = (zf - float(zs[0])) / float(dz)
    k0 = torch.clamp(torch.floor(kf).long(), 0, len(zs) - 2)
    wz = torch.clamp(kf - k0, 0.0, 1.0)
    zin = (kf >= 0) & (kf <= len(zs) - 1)
    cr = lambda ax_, ay_, bx, by: ax_ * by - ay_ * bx
    for i in range(cen.shape[0] - 1):
        lo_i = np.nanmin(np.concatenate([cp["lower"][i], cp["lower"][i + 1]]))
        up_i = np.nanmax(np.concatenate([cp["upper"][i], cp["upper"][i + 1]]))
        if not np.isfinite(lo_i):
            continue
        pts = np.stack([cen[i] + lo_i * nor[i], cen[i] + up_i * nor[i], cen[i + 1] + lo_i * nor[i + 1], cen[i + 1] + up_i * nor[i + 1]])
        i0 = max(0, int(np.floor((pts[:, 0].min() - 0.05 - lo[0]) / hf)))
        i1 = min(shape_f[0], int(np.ceil((pts[:, 0].max() + 0.05 - lo[0]) / hf)) + 1)
        j0 = max(0, int(np.floor((pts[:, 1].min() - 0.05 - lo[1]) / hf)))
        j1 = min(shape_f[1], int(np.ceil((pts[:, 1].max() + 0.05 - lo[1]) / hf)) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        X = torch.as_tensor(lo[0] + (np.arange(i0, i1) + 0.5) * hf, device=dev, dtype=dt)[:, None]
        Y = torch.as_tensor(lo[1] + (np.arange(j0, j1) + 0.5) * hf, device=dev, dtype=dt)[None, :]
        c0, c1, n0, n1 = (torch.as_tensor(np.asarray(v, float), device=dev, dtype=dt) for v in (cen[i], cen[i + 1], nor[i], nor[i + 1]))
        dc, dn = c1 - c0, n1 - n0
        d0x, d0y = X - c0[0], Y - c0[1]
        a = -cr(dc[0], dc[1], dn[0], dn[1])
        b = cr(d0x, d0y, dn[0], dn[1]) - cr(dc[0], dc[1], n0[0], n0[1])
        c = cr(d0x, d0y, n0[0], n0[1])
        if abs(float(a)) < 1e-12:
            tau = -c / torch.where(torch.abs(b) < 1e-12, torch.full_like(b, 1e-12), b)
        else:
            disc = torch.sqrt(torch.clamp(b * b - 4 * a * c, min=0.0))
            t1, t2 = (-b + disc) / (2 * a), (-b - disc) / (2 * a)
            tau = torch.where(torch.abs(t1 - 0.5) < torch.abs(t2 - 0.5), t1, t2)
        inside = (tau >= 0) & (tau <= 1)
        tau = torch.clamp(tau, 0.0, 1.0)
        Ax, Ay = c0[0] + tau * dc[0], c0[1] + tau * dc[1]
        Bx, By = n0[0] + tau * dn[0], n0[1] + tau * dn[1]
        s = ((X - Ax) * Bx + (Y - Ay) * By) / torch.sqrt(Bx * Bx + By * By)

        def prof(P_):
            p0 = (1 - wz) * P_[i, k0] + wz * P_[i, k0 + 1]
            p1 = (1 - wz) * P_[i + 1, k0] + wz * P_[i + 1, k0 + 1]
            return (1 - tau)[..., None] * p0 + tau[..., None] * p1

        vv = valid[i, k0] & valid[i, k0 + 1] & valid[i + 1, k0] & valid[i + 1, k0 + 1] & zin
        a3 = inside[..., None] & vv & (s[..., None] > prof(L)) & (s[..., None] < prof(U))
        F[i0:i1, j0:j1] |= a3
        if i == 0:
            Fin[i0:i1, j0:j1] |= a3
    Xf = torch.as_tensor(lo[0] + (np.arange(shape_f[0]) + 0.5) * hf, device=dev, dtype=dt)[:, None, None]
    Yf = torch.as_tensor(lo[1] + (np.arange(shape_f[1]) + 0.5) * hf, device=dev, dtype=dt)[None, :, None]
    rel = (Xf - float(ep[0])) * float(ed[0]) + (Yf - float(ep[1])) * float(ed[1])
    E = (rel > 0).expand(shape_f) & ~F
    return F, E, Fin


def _pool(B, sub):
    import torch.nn.functional as Fnn
    return Fnn.avg_pool3d(B.float()[None, None], sub)[0, 0]


def build_grid_cut(cp: dict, h: float = 0.1, sub: int = 4, ext_len: float = 4.0, ext_half: float = 5.0, margin: float = 0.3,
                   surface_names: tuple | None = None, x_hard: float = 0.0, device: str = "cuda") -> Grid:
    """절단 칸 격자. Grid.meta 에 vf · veff · walls(칸별 벽: cell, area, tis, pos, dir) 를 더한다 — `fdtd3d.Sim` 이 쓴다."""
    import torch
    import torch.nn.functional as Fnn
    from scipy.ndimage import distance_transform_edt
    g = build_grid(cp, h=h, ext_len=ext_len, ext_half=ext_half, margin=margin, surface_names=surface_names, x_hard=x_hard)
    lo = g.origin
    nx, ny, nz = g.shape
    hf = h / sub
    Ft, Fe, Fin = _fine_occupancy(_fill_surf(cp), lo, (nx * sub, ny * sub, nz * sub), hf, g.exit_point, g.exit_dir, device)
    Fa = Ft | Fe
    vf_t, vf_e, vf = _pool(Ft, sub), _pool(Fe, sub), _pool(Fa, sub)
    air = vf > 0
    inlet = (_pool(Fin, sub) > 0) & air

    def aperture(axis):
        n = Fa.shape[axis]
        idx_lo = torch.arange(sub - 1, n - 1, sub, device=Fa.device)          # 고운 sub·i − 1, i = 1 … n/sub − 1
        both = (Fa.index_select(axis, idx_lo) & Fa.index_select(axis, idx_lo + 1)).float()
        k = [sub, sub, sub]
        k[axis] = 1
        ap = Fnn.avg_pool3d(both[None, None], tuple(k))[0, 0]
        shp = list(air.shape)
        shp[axis] += 1
        out = torch.zeros(shp, device=Fa.device)
        sl = [slice(None)] * 3
        sl[axis] = slice(1, -1)
        out[tuple(sl)] = ap
        return out

    A = [aperture(a) for a in range(3)]
    amax = torch.zeros_like(vf)
    for axis in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis], sl1[axis] = slice(0, -1), slice(1, None)
        amax = torch.maximum(amax, torch.maximum(A[axis][tuple(sl0)], A[axis][tuple(sl1)]))
    veff = torch.where(air, torch.maximum(vf, amax), torch.zeros_like(vf))
    # ---- 벽: 고운 경계 면 × |n·e| 를 공기 쪽 굵은 칸에
    wk = torch.exp(-0.5 * (torch.arange(-3, 4, device=Fa.device, dtype=torch.float32) / 1.5) ** 2)
    wk = wk / wk.sum()
    S = Fa.float()[None, None]
    for axis in range(3):
        shp = [1, 1, 1, 1, 1]
        shp[2 + axis] = 7
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = 3
        S = Fnn.conv3d(Fnn.pad(S, pad, mode="replicate"), wk.view(shp))
    S = S[0, 0]
    gr = torch.gradient(S)
    gn = torch.sqrt(gr[0] ** 2 + gr[1] ** 2 + gr[2] ** 2) + 1e-9
    warea = torch.zeros_like(vf)
    wn = [torch.zeros_like(vf) for _ in range(3)]
    for axis in range(3):
        n = Fa.shape[axis]
        A0 = Fa.narrow(axis, 0, n - 1)
        A1 = Fa.narrow(axis, 1, n - 1)
        cosv = torch.abs(gr[axis].narrow(axis, 0, n - 1) + gr[axis].narrow(axis, 1, n - 1)) / (gn.narrow(axis, 0, n - 1) + gn.narrow(axis, 1, n - 1))
        aw = torch.clamp(cosv, 0.05, 1.0) * hf * hf
        for m_, sgn, padlo in ((A0 & ~A1, 1.0, False), (A1 & ~A0, -1.0, True)):
            val = aw * m_.float()
            p6 = [0, 0, 0, 0, 0, 0]
            j = 2 * (2 - axis)
            if padlo:
                p6[j] = 1          # 공기 쪽이 높은 쪽 칸 (i+1) — 앞에 한 칸 덧대 고운 칸 번호를 맞춘다
            else:
                p6[j + 1] = 1      # 공기 쪽이 낮은 쪽 칸 (i)
            val = Fnn.pad(val, p6)
            cs = _pool(val, sub) * sub ** 3
            warea += cs
            wn[axis] += sgn * cs
    warea = warea * air.float()
    # ---- 조직: 가장 가까운 고체 굵은 칸 (build_grid 의 면 조직을 고체 칸으로 옮긴 표)
    solid = ~g.air
    tis_cell = np.zeros(g.shape, np.int8)
    for axis, W in enumerate((g.wx, g.wy, g.wz)):
        idx = np.argwhere(W > 0)
        for side in (-1, 0):
            c_ = idx.copy()
            c_[:, axis] += side
            ok = (c_[:, axis] >= 0) & (c_[:, axis] < g.shape[axis])
            cc, ii = c_[ok], idx[ok]
            s_ok = solid[tuple(cc.T)]
            tis_cell[tuple(cc[s_ok].T)] = W[tuple(ii[s_ok].T)]
    labeled = tis_cell > 0
    _, ind = distance_transform_edt(~labeled, return_indices=True)
    wa = warea.cpu().numpy()
    cells = np.argwhere(wa > 1e-8)
    tis = tis_cell[tuple(ind[:, cells[:, 0], cells[:, 1], cells[:, 2]])]
    nvec = np.stack([w_.cpu().numpy()[tuple(cells.T)] for w_ in wn], 1)
    nvec /= np.linalg.norm(nvec, axis=1, keepdims=True) + 1e-12
    g.air = air.cpu().numpy()
    g.exterior = ((vf_e > 0.5 * vf) & air).cpu().numpy()
    g.inlet = inlet.cpu().numpy()
    g.ax, g.ay, g.az = (a_.cpu().numpy().astype(np.float32) for a_ in A)
    g.meta.update(vf=vf.cpu().numpy().astype(np.float32), veff=veff.cpu().numpy().astype(np.float32),
                  walls=dict(cell=np.ravel_multi_index(tuple(cells.T), g.shape), area=wa[tuple(cells.T)].astype(np.float64),
                             tis=tis.astype(np.int8), pos=g.origin + (cells + 0.5) * h, dir=nvec),
                  vol_tract=float(vf_t.sum().item() * h ** 3), n_air=int(air.sum().item()), cut=True, sub=sub)
    del Ft, Fe, Fin, Fa, S, gr
    torch.cuda.empty_cache()
    return g
