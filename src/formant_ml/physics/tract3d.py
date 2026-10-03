"""3 차원 성도 기하 — VTL 단면 윤곽에서 직교 격자의 공기 칸·열린 면·벽 조직으로 (MEASUREMENTS §52.526).

사용자 (2026-10-02): *"3D로 전달함수를 구성해야 할 거야 … 성도가 기역자로 굽어있어서, 그 부분 잘 고려해야 할 거야 … 경구개, 치아를 제외하면 다
부드럽다는 것도 고려해야 하고"*. 1D 관은 성도를 곧은 관으로 펴고 단면을 넓이 하나로 뭉친다 — 가로 모드(너비 3–4 cm 에서 ~5–6 kHz 위), ㄱ 자
굽이, 단면 모양, 벽 조직의 차이를 모두 잃는다.

기하의 근원은 VTL 의 3 차원 표면 모형(혀·치아·입술·윗벽·아랫벽·옆벽·후두개·목젖, `vtl_api.VTL.cross_profiles`)이다. VTL 은 성문→입술 중심선의
단면 129 개마다 정중 시상면의 자름 선(점 c_i, 단위 법선 n_i)과 옆 방향 z 의 평면으로 표면을 자르고, 옆 표본 96 개(7 cm)마다 아래·위 윤곽 s ∈ [l, u]
사이를 공기로 둔다 (면적 함수가 이 적분). 여기서는 이웃 단면 사이(띠)를 쌍선형으로 이어 3 차원 공기 영역을 세우고 직교 격자에 올린다:

* 칸 가운데가 공기면 공기 칸. 칸 사이 면은 면 위 2×2 표본의 공기 몫이 **열린 비율** — 격자보다 좁은 협착의 넓이를 칸 크기에 묶이지 않고 담는다.
* 공기 칸과 닿은 단단한 쪽 면은 **벽 면**이고, 그 벽을 이룬 VTL 표면 번호로 조직을 매긴다: 치아·경구개(윗벽 중 x ≥ 0, VTL 의 경구개 갈비는
  x = 0 … 경구개 길이)는 단단, 혀·연구개(목젖 포함)·인두·후두 벽·볼(옆벽)·입술·후두개·턱 쪽 아랫벽은 무름.
* 입술 끝 단면 너머는 바깥 공기. 머리는 지금 입술 끝 단면을 지나는 평면(무한 배플)으로 두고, 그 뒤 성도 밖은 고체다 — 머리 모양은 다음 단계.
* 성문 끝(단면 0)의 앞은 고체 (닫힌 성문). 음원은 첫 띠의 공기 칸들에 부피 유량으로 넣는다.

좌표: VTL 정중 시상면 x (앞), y (위), 옆 z [cm]. 격자 축은 이 셋과 나란하다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 조직 번호 (벽 면) — 0 은 벽 아님.
TISSUE = ("none", "rigid", "tongue", "velum", "pharynx", "cheek", "lip", "epiglottis", "jaw_cover", "glottis_end", "baffle")
T_ID = {n: i for i, n in enumerate(TISSUE)}

#: VTL SurfaceIndex → 조직 (UPPER_COVER 는 자리로 다시 가른다).
_SURF = {"UPPER_TEETH": "rigid", "LOWER_TEETH": "rigid", "TONGUE": "tongue", "UVULA": "velum", "EPIGLOTTIS": "epiglottis",
         "UPPER_LIP": "lip", "LOWER_LIP": "lip", "LEFT_COVER": "cheek", "RIGHT_COVER": "cheek", "LOWER_COVER": "jaw_cover",
         "UPPER_COVER": "upper_cover", "RADIATION": "baffle"}


def surface_tissue(names: tuple, sid: np.ndarray, x: np.ndarray, y: np.ndarray, x_hard: float = 0.0,
                   y_velum: float = -1.6, y_throat: float = -1.4) -> np.ndarray:
    """VTL 표면 번호 + 자리 → 조직 번호. 윗벽: x ≥ `x_hard` 경구개(단단), 그 뒤 y ≥ `y_velum` 연구개, 아래는 인두 뒷벽.
    아랫벽: y ≥ `y_throat` 는 턱 쪽(입 바닥·잇몸), 아래는 인두 앞벽·후두."""
    out = np.zeros(sid.shape, np.int8)
    for k, nm in enumerate(names):
        m = sid == k
        if not m.any():
            continue
        t = _SURF.get(nm)
        if t is None:
            continue
        if t == "upper_cover":
            hard = m & (x >= x_hard)
            vel = m & (x < x_hard) & (y >= y_velum)
            out[hard] = T_ID["rigid"]
            out[vel] = T_ID["velum"]
            out[m & ~hard & ~vel] = T_ID["pharynx"]
        elif t == "jaw_cover":
            out[m & (y >= y_throat)] = T_ID["jaw_cover"]
            out[m & (y < y_throat)] = T_ID["pharynx"]
        else:
            out[m] = T_ID[t]
    return out


@dataclass
class Grid:
    h: float                      # 칸 크기 [cm]
    origin: np.ndarray            # 칸 (0,0,0) 의 모서리 [cm] (x, y, z)
    air: np.ndarray               # (nx, ny, nz) bool
    ax: np.ndarray                # (nx+1, ny, nz) x 면 열린 비율
    ay: np.ndarray
    az: np.ndarray
    wx: np.ndarray                # (nx+1, ny, nz) int8 x 면 벽 조직 (공기 칸 쪽 벽만, 0 = 벽 아님)
    wy: np.ndarray
    wz: np.ndarray
    inlet: np.ndarray             # (nx, ny, nz) bool — 음원 칸 (첫 띠)
    exterior: np.ndarray          # bool — 바깥 공기 (입술 끝 평면 너머)
    exit_point: np.ndarray        # 입술 끝 중심 (x, y) [cm]
    exit_dir: np.ndarray          # 입술 끝 접선 (x, y)
    cx: np.ndarray                # (nx+1, ny, nz) 벽 면의 넓이 몫 |n·e_x| (계단 면 → 실제 표면 넓이)
    cy: np.ndarray
    cz: np.ndarray
    area_1d: np.ndarray           # 같은 윤곽의 1D 면적 함수 (129,) [cm²]
    pos_1d: np.ndarray            # 중심선 거리 (129,) [cm]
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.air.shape

    def centers(self, axis: int) -> np.ndarray:
        return self.origin[axis] + (np.arange(self.air.shape[axis]) + 0.5) * self.h

    def index(self, p) -> tuple:
        """점 (x, y, z) [cm] 이 든 칸 번호."""
        return tuple(int(np.floor((p[a] - self.origin[a]) / self.h)) for a in range(3))


def _strip_eval(cp: dict, i: int, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """띠 i (단면 i … i+1) 안의 점들: (안에 듦, 공기, s − l, u − s, 옆 끝 밖, 표본 번호 k, τ) — 점은 1 차원 배열."""
    c0, c1 = cp["center"][i], cp["center"][i + 1]
    n0, n1 = cp["normal"][i], cp["normal"][i + 1]
    d0x, d0y = X - c0[0], Y - c0[1]
    dc = c1 - c0
    dn = n1 - n0
    cr = lambda ax_, ay_, bx, by: ax_ * by - ay_ * bx
    # (d0 − τ dc) × (n0 + τ dn) = 0 → a τ² + b τ + c = 0
    a = -cr(dc[0], dc[1], dn[0], dn[1])
    b = cr(d0x, d0y, dn[0], dn[1]) - cr(dc[0], dc[1], n0[0], n0[1])
    c = cr(d0x, d0y, n0[0], n0[1])
    if abs(a) < 1e-12:
        tau = -c / np.where(np.abs(b) < 1e-12, 1e-12, b)
    else:
        disc = np.sqrt(np.clip(b * b - 4 * a * c, 0.0, None))
        t1 = (-b + disc) / (2 * a)
        t2 = (-b - disc) / (2 * a)
        tau = np.where(np.abs(t1 - 0.5) < np.abs(t2 - 0.5), t1, t2)
    inside = (tau >= 0.0) & (tau <= 1.0)
    tau = np.clip(tau, 0.0, 1.0)
    Ax = c0[0] + tau * dc[0]
    Ay = c0[1] + tau * dc[1]
    Bx = n0[0] + tau * dn[0]
    By = n0[1] + tau * dn[1]
    Bn = np.sqrt(Bx * Bx + By * By)
    s = ((X - Ax) * Bx + (Y - Ay) * By) / Bn
    zs = cp["z"]
    dz = zs[1] - zs[0]
    kf = (Z - zs[0]) / dz
    k0 = np.clip(np.floor(kf).astype(int), 0, len(zs) - 2)
    w = np.clip(kf - k0, 0.0, 1.0)
    out_lat = (kf < 0) | (kf > len(zs) - 1)

    def prof(P):
        p0 = (1 - w) * P[i, k0] + w * P[i, k0 + 1]
        p1 = (1 - w) * P[i + 1, k0] + w * P[i + 1, k0 + 1]
        return (1 - tau) * p0 + tau * p1

    lo = prof(cp["lower"])
    up = prof(cp["upper"])
    valid = np.isfinite(lo) & np.isfinite(up) & ~out_lat
    air = inside & valid & (s > lo) & (s < up)
    k = np.clip(np.rint(kf).astype(int), 0, len(zs) - 1)
    return inside, air, np.where(valid, s - lo, np.nan), np.where(valid, up - s, np.nan), out_lat, k, tau


def _air_points(cp: dict, X, Y, Z, exit_point, exit_dir, baffle: bool):
    """점들의 공기 여부 + 벽 표면 정보. 반환 (air, interior, surf_id, inlet)."""
    n = X.size
    air = np.zeros(n, bool)
    inlet = np.zeros(n, bool)
    sid = np.full(n, -1, np.int16)
    best = np.full(n, np.inf)
    cen, nor = cp["center"], cp["normal"]
    for i in range(cp["center"].shape[0] - 1):
        lo_i = np.nanmin(np.concatenate([cp["lower"][i], cp["lower"][i + 1]]))
        up_i = np.nanmax(np.concatenate([cp["upper"][i], cp["upper"][i + 1]]))
        if not np.isfinite(lo_i):
            continue
        pts = np.stack([cen[i] + lo_i * nor[i], cen[i] + up_i * nor[i], cen[i + 1] + lo_i * nor[i + 1], cen[i + 1] + up_i * nor[i + 1]])
        mx = 0.3
        sel = np.flatnonzero((X >= pts[:, 0].min() - mx) & (X <= pts[:, 0].max() + mx) & (Y >= pts[:, 1].min() - mx) &
                             (Y <= pts[:, 1].max() + mx) & (np.abs(Z) <= 3.6))
        if sel.size == 0:
            continue
        ins, a, dl, du, olat, k, tau = _strip_eval(cp, i, X[sel], Y[sel], Z[sel])
        air[sel] |= a
        if i == 0:
            inlet[sel] |= a
        # 벽 표면: 띠 안의 고체 점 — 가까운 윤곽 쪽 (아래/위) 의 표면 번호
        m = ins & ~a
        if m.any():
            dist = np.where(np.isfinite(dl), np.minimum(np.abs(dl), np.abs(du)), 0.5)
            upper_side = np.where(np.isfinite(dl), np.abs(du) < np.abs(dl), True)
            ii = np.where(tau < 0.5, i, i + 1)
            sv = np.where(upper_side, cp["usurf"][ii, k], cp["lsurf"][ii, k])
            # 옆 끝(표본 없음): 가장 가까운 유효 표본의 표면 — 열의 양 끝 유효 표본
            better = m & (dist < best[sel])
            idx = sel[better]
            sid[idx] = sv[better]
            best[idx] = dist[better]
    rel = (X - exit_point[0]) * exit_dir[0] + (Y - exit_point[1]) * exit_dir[1]
    ext = (rel > 0) if baffle else np.zeros(n, bool)
    ext &= ~air
    return air, ext, sid, inlet


def _fill_surf(cp: dict) -> dict:
    """옆 끝(윤곽 없음) 표본의 벽 표면 번호를 같은 단면의 가장 가까운 유효 표본 것으로 — 옆벽의 조직을 매기려고."""
    cp = dict(cp)
    for key, prof in (("usurf", "upper"), ("lsurf", "lower")):
        S = cp[key].copy()
        for i in range(S.shape[0]):
            ok = np.flatnonzero(np.isfinite(cp[prof][i]) & (S[i] >= 0))
            if ok.size == 0:
                continue
            j = np.arange(S.shape[1])
            near = ok[np.abs(j[:, None] - ok[None, :]).argmin(1)]
            S[i] = S[i, near]
        cp[key] = S
    return cp


def build_grid(cp: dict, h: float = 0.1, ext_len: float = 4.0, ext_half: float = 4.0, margin: float = 0.3,
               surface_names: tuple | None = None, x_hard: float = 0.0) -> Grid:
    """단면 윤곽 → 격자. `ext_len`: 입술 끝 너머 바깥 공기 길이 [cm], `ext_half`: 그 옆·위아래 반폭."""
    from ..engine.vtl_api import VTL
    names = surface_names or VTL.SURFACES
    cp = _fill_surf(cp)
    cen, nor = cp["center"], cp["normal"]
    ep = cen[-1]
    ed = cen[-1] - cen[-3]
    ed = ed / np.linalg.norm(ed)
    # 성도의 끝점들
    P = []
    for i in range(cen.shape[0]):
        for s in (np.nanmin(cp["lower"][i]), np.nanmax(cp["upper"][i])):
            if np.isfinite(s):
                P.append(cen[i] + s * nor[i])
    P = np.array(P)
    e_perp = np.array([-ed[1], ed[0]])
    for a in (-ext_half, ext_half):
        for L in (0.0, ext_len):
            P = np.vstack([P, ep + L * ed + a * e_perp])
    lo = np.array([P[:, 0].min() - margin, P[:, 1].min() - margin, -max(3.6, ext_half) - margin])
    hi = np.array([P[:, 0].max() + margin, P[:, 1].max() + margin, max(3.6, ext_half) + margin])
    nx, ny, nz = (int(np.ceil((hi[a] - lo[a]) / h)) for a in range(3))
    xc = lo[0] + (np.arange(nx) + 0.5) * h
    yc = lo[1] + (np.arange(ny) + 0.5) * h
    zc = lo[2] + (np.arange(nz) + 0.5) * h

    def evalp(x, y, z):
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        a, e, s, inl = _air_points(cp, X.ravel(), Y.ravel(), Z.ravel(), ep, ed, True)
        sh = X.shape
        return a.reshape(sh), e.reshape(sh), s.reshape(sh), inl.reshape(sh), X, Y

    a_c, e_c, s_c, inl, Xc, Yc = evalp(xc, yc, zc)
    air = a_c | e_c
    # 바깥 공기의 끝: 입술 끝에서 ext_len 넘으면 고체(흡수층은 풀이기가 둔다) — 상자 끝이 곧 흡수층
    # 면 열린 비율: 면 위 2×2 표본
    q = np.array([0.25, 0.75])

    def face_frac(axis):
        cs = [xc, yc, zc]
        edges = [lo[a] + np.arange(n_ + 1) * h for a, n_ in enumerate((nx, ny, nz))]
        acc = None
        for qa in q:
            for qb in q:
                coords = []
                j = 0
                for a in range(3):
                    if a == axis:
                        coords.append(edges[a][1:-1])
                    else:
                        coords.append(lo[a] + (np.arange((nx, ny, nz)[a]) + (qa if j == 0 else qb)) * h)
                        j += 1
                fa, fe, _, _, _, _ = evalp(*coords)
                v = (fa | fe).astype(np.float32)
                acc = v if acc is None else acc + v
        return acc / 4.0

    def full(axis, inner):
        shp = list(air.shape)
        shp[axis] += 1
        out = np.zeros(shp, np.float32)
        sl = [slice(None)] * 3
        sl[axis] = slice(1, -1)
        out[tuple(sl)] = inner
        return out

    fr = [full(a, face_frac(a)) for a in range(3)]
    A = []
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1)
        sl1[a] = slice(1, None)
        both = np.zeros_like(fr[a], bool)
        inner = air[tuple(sl0)] & air[tuple(sl1)]
        sl = [slice(None)] * 3
        sl[a] = slice(1, -1)
        both[tuple(sl)] = inner
        A.append(np.where(both, np.maximum(fr[a], 0.0), 0.0).astype(np.float32))
    # 벽 조직: 공기 칸과 고체 칸 사이 면 — 고체 칸의 표면 번호 + 자리
    tis_cell = surface_tissue(names, s_c, Xc, Yc, x_hard=x_hard)
    tis_cell[(s_c < 0) & ~air] = T_ID["baffle"]
    W = []
    for a in range(3):
        shp = list(air.shape)
        shp[a] += 1
        w = np.zeros(shp, np.int8)
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1)
        sl1[a] = slice(1, None)
        aL, aR = air[tuple(sl0)], air[tuple(sl1)]
        tL, tR = tis_cell[tuple(sl0)], tis_cell[tuple(sl1)]
        inner = np.zeros(aL.shape, np.int8)
        m1 = aL & ~aR
        m2 = aR & ~aL
        inner[m1] = tR[m1]
        inner[m2] = tL[m2]
        inner[(m1 | m2) & (inner == 0)] = T_ID["baffle"]
        sl = [slice(None)] * 3
        sl[a] = slice(1, -1)
        w[tuple(sl)] = inner
        W.append(w)
    # 계단 벽의 넓이 보정: 실제 표면 법선 n (공기 몫을 σ = 1.2 칸으로 고른 장의 기울기) 과 면 축의 |cos| — 계단 면들의 넓이 합이 실제 표면
    # 넓이가 된다 (구에서 계단 면 넓이는 실제의 1.5 배). 경계층·벽 손실이 넓이에 비례하므로 필요하다.
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(air.astype(np.float32), 1.2)
    gr = np.gradient(sm)
    gn = np.sqrt(gr[0] ** 2 + gr[1] ** 2 + gr[2] ** 2) + 1e-9
    C = []
    for a_ in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a_] = slice(0, -1)
        sl1[a_] = slice(1, None)
        num = np.abs(gr[a_][tuple(sl0)] + gr[a_][tuple(sl1)])
        den = gn[tuple(sl0)] + gn[tuple(sl1)]
        cos = np.clip(num / den, 0.05, 1.0)
        shp = list(air.shape)
        shp[a_] += 1
        cc = np.ones(shp, np.float32)
        sl = [slice(None)] * 3
        sl[a_] = slice(1, -1)
        cc[tuple(sl)] = cos
        C.append(cc)
    # 성문 끝: 음원 칸의 아래(단면 0 앞) 벽은 성문 끝
    area = np.nansum(np.clip(cp["upper"] - cp["lower"], 0.0, None), 1) * (cp["z"][1] - cp["z"][0])
    g = Grid(h=h, origin=lo, air=air, ax=A[0], ay=A[1], az=A[2], wx=W[0], wy=W[1], wz=W[2],
             cx=C[0], cy=C[1], cz=C[2], inlet=inl & a_c,
             exterior=e_c, exit_point=ep, exit_dir=ed, area_1d=area, pos_1d=cp["pos"].copy(),
             meta=dict(n_air=int(air.sum()), n_tract=int(a_c.sum()), vol_tract=float(a_c.sum() * h ** 3)))
    return g


def mesh_wall_area(g: Grid, sigma: float = 0.6) -> None:
    """벽 면마다 실제 표면 넓이 [칸 넓이 h² 몫] 을 다시 매긴다 (§52.526) — 공기 몫을 σ 칸으로 고른 장의 0.5 등치면(marching cubes)을
    세모로 나눠, 세모마다 가장 가까운 벽 면에 그 넓이를 더한다. 기울기 법선의 |cos| (`build_grid`) 는 원통에서 실제 넓이의 1.17 배였다.
    결과는 g.cx/cy/cz 를 덮어쓴다 (벽 면의 넓이 = h² × c)."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    sm = gaussian_filter(g.air.astype(np.float32), sigma)
    pad = np.pad(sm, 1, constant_values=0.0)
    verts, faces, _, _ = marching_cubes(pad, 0.5, spacing=(g.h, g.h, g.h))
    verts = verts - g.h + g.origin + 0.5 * g.h          # 칸 가운데 좌표로
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    cen = tri.mean(1)
    pts, ref = [], []
    for a, W in enumerate((g.wx, g.wy, g.wz)):
        idx = np.argwhere(W > 0)
        c = g.origin + (idx + 0.5) * g.h
        c[:, a] -= 0.5 * g.h
        pts.append(c)
        ref.append(np.column_stack([np.full(len(idx), a), idx]))
    pts = np.concatenate(pts)
    ref = np.concatenate(ref)
    d, j = cKDTree(pts).query(cen, k=1)
    keep = d < 1.5 * g.h                                # 상자 끝·흡수층의 가짜 면은 버린다
    acc = np.bincount(j[keep], weights=area[keep], minlength=len(pts)) / (g.h * g.h)
    for a, C in enumerate((g.cx, g.cy, g.cz)):
        C[:] = 1.0
        m = ref[:, 0] == a
        C[tuple(ref[m, 1:].T)] = acc[m].astype(np.float32)
    g.meta["mesh_area"] = float(area[keep].sum())


def _min_area_ts(x: float) -> float:
    """VTL `tongueSideParamToMinArea_cm2` — 혀 옆 변수 → 옆 통로 최소 면적 [cm²] (0.2–0.4: 마찰음 버팀 0–0.15, −0.2–−0.4: 설측음 0–0.20)."""
    if x > 0.2:
        return min(0.15, 0.15 * (x - 0.2) / 0.2)
    if x < -0.2:
        return min(0.20, 0.20 * (x + 0.2) / -0.2)
    return 0.0


def vtl_floors(cp: dict, incisor_pos: float, ts2: float, ts3: float, slit_w: float = 1.0, lat_w: float = 0.3) -> dict:
    """VTL 이 1D 면적 함수에 덧씌우는 최소 면적을 3D 윤곽에 통로로 낸다 (§52.526) — **잠정**: VTL 3D 표면에 없는 이 사이 틈·구강 전정(앞니 둘레
    −0.5…+0.3 cm 에서 0.15 cm²)과 혀 옆 통로(혀 뒤 TS2, 혀끝 TS3). 앞니 둘레는 가운데 틈(너비 `slit_w`), 혀 옆은 공기 영역 양 가장자리의 옆 통로
    (너비 `lat_w`). 실제 이 사이 틈·전정 기하로 바꿔야 한다. 반환: 고친 cp (+ 'floor_added' 칸별 더한 면적)."""
    cp = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cp.items()}
    z = cp["z"]
    dz = z[1] - z[0]
    U, L = cp["upper"], cp["lower"]
    pos = cp["pos"]
    area = np.nansum(np.clip(U - L, 0.0, None), 1) * dz
    from ..engine.vtl_api import VTL
    mid = [len(z) // 2 - 1, len(z) // 2]
    lsm = cp["lsurf"][:, mid]
    tongue = np.flatnonzero((lsm == VTL.SURFACES.index("TONGUE")).any(1))
    lip = np.flatnonzero((lsm == VTL.SURFACES.index("LOWER_LIP")).any(1) | (lsm == VTL.SURFACES.index("RADIATION")).any(1))
    tip_r = pos[tongue.max()] if tongue.size else 0.0
    lip_l = pos[lip.min()] if lip.size else 1e6
    tip_l = tip_r - 2.0
    added = np.zeros(len(pos))
    for i in range(len(pos)):
        need = 0.0
        kind = None
        if incisor_pos - 0.5 <= pos[i] <= incisor_pos + 0.3 and area[i] < 0.15:
            need, kind = 0.15 - area[i], "slit"
        mts = _min_area_ts(ts2) if (pos[i] <= tip_l and ts2 >= 0.0) else (_min_area_ts(ts3) if tip_l <= pos[i] <= lip_l else 0.0)
        if area[i] + need < mts:
            need, kind = mts - area[i], kind or "lateral"
        if need <= 1e-6:
            continue
        ok = np.isfinite(U[i]) & np.isfinite(L[i])
        # 기준 높이: 이 단면의 유효 표본 가운데, 없으면 이웃 단면
        ref_i = i
        while not ok.any() and ref_i > 0:
            ref_i -= 1
            ok = np.isfinite(U[ref_i]) & np.isfinite(L[ref_i])
        if not ok.any():
            continue
        s0 = float(np.nanmedian(0.5 * (U[ref_i][ok] + L[ref_i][ok])))
        if kind == "slit" or not (np.isfinite(U[i]) & np.isfinite(L[i])).any():
            sel = np.abs(z) <= slit_w / 2
        else:
            okk = np.flatnonzero(np.isfinite(U[i]) & np.isfinite(L[i]))
            lo_k, hi_k = okk.min(), okk.max()
            nk = max(1, int(round(lat_w / dz)))
            sel = np.zeros(len(z), bool)
            sel[max(0, lo_k - nk):lo_k + 1] = True
            sel[hi_k:min(len(z), hi_k + nk + 1)] = True
        w = sel.sum() * dz
        dh = need / w
        for k in np.flatnonzero(sel):
            if np.isfinite(U[i, k]) and np.isfinite(L[i, k]) and U[i, k] > L[i, k]:
                c = 0.5 * (U[i, k] + L[i, k])
                half = 0.5 * (U[i, k] - L[i, k]) + 0.5 * dh
            else:
                c = s0 if not (np.isfinite(U[i, k]) and np.isfinite(L[i, k])) else 0.5 * (U[i, k] + L[i, k])
                half = 0.5 * dh
            U[i, k], L[i, k] = c + half, c - half
        added[i] = need
    cp["floor_added"] = added
    return cp


#: 앞니 너비 [cm] (가운데부터: 중절치, 측절치, 견치) — Wheeler 치아 해부학 머리 크라운 근원심 폭 (위 8.5 · 6.5 · 7.5, 아래 5.0 · 5.5 · 7.0 mm).
TOOTH_W_UPPER = (0.85, 0.65, 0.75)
TOOTH_W_LOWER = (0.50, 0.55, 0.70)


def teeth_embrasures(cp: dict, incisor_pos: float, depth: float = 0.12, width: float = 0.10, window=(-0.6, 0.4)) -> dict:
    """앞니 사이의 V 자 틈 (절단 공극) 을 단면 윤곽의 치아 경계에 판다 (§52.527) — 이가 맞물려도 바람이 지나는 실제 길. VTL 1D 의 "앞니 둘레 0.15 cm²"
    하한(`vtl_floors`) 을 기하로 바꾼다. 틈 자리: 이웃 이의 경계 (가운데 0, ±중절치, ±중절치+측절치, …). 틈 모양: 끝에서 깊이 `depth`, 폭 `width` 의
    세모 (잠정 — 화자 치아 계측으로 바꿀 것). 앞니 둘레 (`window`, 중심선 위치 기준) 의 단면에서 윗 경계가 위 이(UPPER_TEETH) 인 표본은 위로, 아랫 경계가
    아래 이(LOWER_TEETH) 인 표본은 아래로 판다."""
    from ..engine.vtl_api import VTL
    cp = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cp.items()}
    cpf = _fill_surf(cp)
    z = cp["z"]
    U, L = cp["upper"], cp["lower"]
    ut, lt = VTL.SURFACES.index("UPPER_TEETH"), VTL.SURFACES.index("LOWER_TEETH")

    def notch(widths):
        edges = [0.0]
        acc = 0.0
        for w_ in widths:
            acc += w_
            edges += [acc, -acc]
        prof = np.zeros_like(z)
        for e in edges:
            prof = np.maximum(prof, depth * np.clip(1.0 - np.abs(z - e) / (0.5 * width), 0.0, 1.0))
        return prof

    nu, nl = notch(TOOTH_W_UPPER), notch(TOOTH_W_LOWER)
    sel = (cp["pos"] >= incisor_pos + window[0]) & (cp["pos"] <= incisor_pos + window[1])
    added = np.zeros(len(cp["pos"]))
    dz = z[1] - z[0]
    for i in np.flatnonzero(sel):
        mu_ = cpf["usurf"][i] == ut
        ml_ = cpf["lsurf"][i] == lt
        # 닫힌 표본(VTL 이 지운 곳 — 이가 맞물림)도 틈을 낸다: 맞물린 높이 s0 = 이 단면 유효 표본의 가운데, 없으면 이웃 단면
        ok = np.isfinite(U[i]) & np.isfinite(L[i])
        j = i
        while not ok.any() and j > 0:
            j -= 1
            ok = np.isfinite(U[j]) & np.isfinite(L[j])
        s0 = float(np.nanmedian(0.5 * (U[j][ok] + L[j][ok]))) if ok.any() else 0.0
        for k in range(len(z)):
            if nu[k] <= 0 and nl[k] <= 0:
                continue
            up_ok, lo_ok = np.isfinite(U[i, k]), np.isfinite(L[i, k])
            if up_ok and lo_ok:
                if mu_[k] and nu[k] > 0:
                    U[i, k] += nu[k]
                    added[i] += nu[k] * dz
                if ml_[k] and nl[k] > 0:
                    L[i, k] -= nl[k]
                    added[i] += nl[k] * dz
            elif (mu_[k] or ml_[k]) and abs(z[k]) < 2.5:
                U[i, k], L[i, k] = s0 + nu[k], s0 - nl[k]
                added[i] += (nu[k] + nl[k]) * dz
    cp["emb_added"] = added
    return cp
