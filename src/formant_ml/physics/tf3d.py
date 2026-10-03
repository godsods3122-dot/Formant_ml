"""성문 → 20 cm 마이크 3D 전달 (MEASUREMENTS §52.526) — 성도 내부 3D (`fdtd3d`) × 머리·몸통 상반성 표면 전달 (`head3d`).

p_mic(ω) = Σ_입 G(ω, 표면 점) Q_면(ω) + Σ_벽 G(ω, 피부 점) Q_벽(ω):
* 입: 성도 내부 3D 의 바깥 공기가 시작되는 x 면들을 지나는 면 유량 (입 단면). 그 자리의 G 는 머리 모형 입 둘레 표면 점 (y, z 가장 가까운 것).
* 벽: 무른 벽 면마다 바깥 법선으로 몸 모형을 나갈 때의 피부 점 — 볼·입술·혀(턱 밑)·입 바닥·인두 옆·앞벽(목). 인두 뒷벽(척추에 막힘)·후두개(양쪽
  공기)·연구개(비강 쪽)는 피부 방사에서 뺀다. 조직이 질량처럼 함께 움직인다고 본다 (얇은 벽의 저역 근사 — 두꺼운 혀·인두는 상한).
같은 물리의 1D 사슬(`chain_1d`)과 그 먼 장 단극 jωρU_L/(4πr) (지금 관 모형 출력의 꼴) 을 함께 낸다.
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np

from . import air as AIR
from . import fdtd3d as F3
from . import head3d as HD
from . import tract3d as T3


def chain_1d(cp: dict, f: np.ndarray, r_mic: float = 20.0, T_c: float = 35.5):
    """같은 윤곽·같은 물리(경계층, 조직별 무른 벽)의 1D 손실 관 — (U_L/U_g, 먼 장 단극 p/U_g at r_mic, 입력 임피던스)."""
    from scipy.special import j1, struve
    from ..engine.vtl_api import VTL
    a_ = AIR.props(T_c, 1.0).cgs()
    rho, c = a_["RHO"], a_["C_SOUND"]
    nu, kap, gam = a_["MU"] / rho, a_["LAMBDA_TH"] / (rho * a_["CP"]), a_["GAMMA"]
    dz = cp["z"][1] - cp["z"][0]
    A = np.maximum(np.nansum(np.clip(cp["upper"] - cp["lower"], 0.0, None), 1) * dz, 1e-3)
    cpf = T3._fill_surf(cp)
    n = A.size
    S_t = {k: np.zeros(n) for k in F3.TISSUE_WALL}
    S_tot = np.zeros(n)
    for i in range(n):
        U_, L_ = cp["upper"][i], cp["lower"][i]
        ok = np.isfinite(U_) & np.isfinite(L_) & (U_ - L_ > 0)
        for prof, sk in ((U_, "usurf"), (L_, "lsurf")):
            seg = np.nan_to_num(np.sqrt(dz ** 2 + np.diff(np.where(ok, prof, np.nan)) ** 2))
            x = cp["center"][i, 0] + np.where(ok, prof, 0) * cp["normal"][i, 0]
            y = cp["center"][i, 1] + np.where(ok, prof, 0) * cp["normal"][i, 1]
            tis = T3.surface_tissue(VTL.SURFACES, cpf[sk][i], x, y)
            S_tot[i] += seg.sum()
            for k in S_t:
                S_t[k][i] += seg[tis[:-1] == T3.T_ID[k]].sum()
    dl = np.diff(cp["pos"])
    w = 2 * np.pi * np.maximum(f, 1.0)
    jw = 1j * w
    Ybl = np.sqrt(jw) * (np.sqrt(nu) + (gam - 1) * np.sqrt(kap)) / (rho * c ** 2)
    M = np.array([[np.ones_like(w), 0 * w], [0 * w, np.ones_like(w)]], dtype=complex)
    for i in range(n - 1):
        Am = 0.5 * (A[i] + A[i + 1])
        Ysh = jw * Am / (rho * c ** 2) + 0.5 * (S_tot[i] + S_tot[i + 1]) * Ybl
        for k, (th, f0, Q) in F3.TISSUE_WALL.items():
            Ss = 0.5 * (S_t[k][i] + S_t[k][i + 1])
            if Ss > 0:
                mm = F3.RHO_TISSUE * th
                Ysh = Ysh + Ss / (jw * mm + 2 * np.pi * f0 * mm / Q + mm * (2 * np.pi * f0) ** 2 / jw)
        Zs = jw * rho / Am
        gm, Zc = np.sqrt(Zs * Ysh), np.sqrt(Zs / Ysh)
        ch, sh = np.cosh(gm * dl[i]), np.sinh(gm * dl[i])
        M = np.einsum("ijf,jkf->ikf", M, np.array([[ch, Zc * sh], [sh / Zc, ch]]))
    aL = np.sqrt(A[-1] / np.pi)
    x2 = 2 * w / c * aL
    Zr = rho * c / A[-1] * ((1 - 2 * j1(x2) / x2) + 1j * 2 * struve(1, x2) / x2)
    UL = 1.0 / (M[1, 0] * Zr + M[1, 1])
    return UL, jw * rho * UL / (4 * np.pi * r_mic), (M[0, 0] * Zr + M[0, 1]) * UL


def mouth_faces(g):
    """입 단면 = 성도 공기 칸과 바깥 공기 칸이 맞닿는 모든 면 (방향 무관) — [(축, 면 번호, 부호)], 면 가운데 좌표. 처음에는 "바깥이 시작되는
    x 면" 하나로 잡았다가 입술 끝이 기운 틀에서 실제 입을 지나지 않아 입 유량이 0 이 되었다 (§52.526)."""
    tract = g.air & ~g.exterior
    sets, fcs = [], []
    for axis, A_ in enumerate((g.ax, g.ay, g.az)):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis], sl1[axis] = slice(0, -1), slice(1, None)
        lo_t, hi_t = tract[tuple(sl0)], tract[tuple(sl1)]
        lo_e, hi_e = g.exterior[tuple(sl0)], g.exterior[tuple(sl1)]
        for m_, sg in ((lo_t & hi_e, 1.0), (lo_e & hi_t, -1.0)):
            ii = np.argwhere(m_)
            if ii.size == 0:
                continue
            fidx = ii.copy()
            fidx[:, axis] += 1                                   # 면 i+1 = 칸 i 와 i+1 사이
            keep = A_[tuple(fidx.T)] > 0
            fidx = fidx[keep]
            sets.append((axis, fidx, np.full(len(fidx), sg)))
            c_ = g.origin + (fidx + 0.5) * g.h
            c_[:, axis] -= 0.5 * g.h
            fcs.append(c_)
    return sets, (np.concatenate(fcs) if fcs else np.zeros((0, 3)))


def glottis_to_mic(cp: dict, recip: dict, h: float = 0.1, ms: float = 40.0, ext_half: float = 5.0, pml: int = 15,
                   fs_out: float = 96000.0, verbose: bool = True, cut: bool = True) -> dict:
    """3D 성문 → 마이크. recip: `head3d` 상반성 결과 (np.load). 반환: f, Hm (전체), Hmouth, Hwall{조직}, Hmono (같은 입 유량의 먼 장 단극),
    H1 (1D 사슬의 먼 장 단극), UL3 (3D 입 유량/U_g), ir (96 kHz 성문→마이크 임펄스 응답), meta."""
    from scipy.ndimage import gaussian_filter
    from scipy.signal import resample_poly
    from scipy.spatial import cKDTree
    t0 = time.time()
    from . import tract3d_cut as TC
    if cut:
        g = TC.build_grid_cut(cp, h=h, ext_len=4.0, ext_half=ext_half)
        sim = F3.Sim(g, courant=0.7, sponge_cells=pml)
    else:
        g = T3.build_grid(cp, h=h, ext_len=4.0, ext_half=ext_half)
        sim = F3.Sim(g, courant=0.9, sponge_cells=pml)
    ep = g.exit_point
    sh_c = recip["origin"] + (recip["idx"] + 0.5) * float(recip["h"])
    reg = recip["reg"]
    # 입 단면 = 성도 공기 칸과 바깥 공기 칸이 맞닿는 모든 면 (방향 무관). 처음에는 "바깥이 시작되는 x 면" 하나로 잡았다가 입술 끝이 기운 틀에서
    # 실제 입을 지나지 않아 입 유량이 0 이 되었다 (§52.526).
    sets, fc3 = mouth_faces(g)
    sim.set_flux_faces(sets)
    mouth_area = float(sum(((g.ax, g.ay, g.az)[a_][tuple(ix.T)]).sum() for a_, ix, _ in sets) * h * h)
    if False:
     tract = g.air & ~g.exterior
     sets, fcs = [], []
    mouth_pts = np.flatnonzero(reg == 0)
    # 입 단면을 머리 모형의 입 자리로: 입술 끝 차이만큼 옮겨 가장 가까운 입 둘레 표면 점 (y, z)
    dy = float(recip["mic"][1]) - ep[1]
    _, jm = cKDTree(sh_c[mouth_pts][:, 1:]).query(fc3[:, 1:] + np.array([dy, 0.0]), k=1)
    mouth_g = mouth_pts[jm]
    tis = sim.wall_tis
    if getattr(sim, "cut", False):
        nrm = sim.wall_dir.copy()                     # 절단 칸: 고운 경계 면에서 모은 바깥 법선
    else:
        grd = np.stack(np.gradient(gaussian_filter(g.air.astype(np.float32), 1.2)), -1)
        ci = np.clip(((sim.wall_pos - g.origin) / g.h).astype(int), 0, np.array(g.shape) - 1)
        nrm = -grd[tuple(ci.T)]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    use = np.isin(tis, [T3.T_ID[t] for t in ("cheek", "lip", "pharynx", "tongue", "jaw_cover")])
    use &= ~((tis == T3.T_ID["pharynx"]) & (nrm[:, 0] < -0.5))
    ep_head = np.array([float(recip["mic"][0]) - 20.0, float(recip["mic"][1])])
    shift = np.array([ep_head[0] - ep[0], ep_head[1] - ep[1], 0.0])
    pos = sim.wall_pos + shift
    alive = use.copy()
    exitp = np.full((len(tis), 3), np.nan)
    for _ in range(200):
        pos[alive] += 0.1 * nrm[alive]
        inside, _x = HD.body_indicator(pos[alive, 0], pos[alive, 1], pos[alive, 2], ep_head)
        ii = np.flatnonzero(alive)
        exitp[ii[~inside]] = pos[ii[~inside]]
        alive[ii[~inside]] = False
        if not alive.any():
            break
    ok = np.isfinite(exitp[:, 0])
    _, js = cKDTree(sh_c).query(exitp[ok], k=1)
    uniq, gid = np.unique(js, return_inverse=True)
    gg = np.full(len(tis), -1)
    gg[np.flatnonzero(ok)] = gid
    sim.set_wall_groups(gg, len(uniq))
    gtis = np.zeros(len(uniq), int)
    gtis[gid] = tis[ok]
    N = int(ms * 1e-3 * sim.fs)
    u, _ = F3.gauss_pulse(sim.fs, 20000.0, N)
    r = sim.run_graph(u, [(ep[0] + 3.0, ep[1], 0.0)])
    fr = Fraction(fs_out / sim.fs).limit_denominator(4000)
    rs = lambda x: resample_poly(x, fr.numerator, fr.denominator, axis=-1)
    fs = sim.fs * fr.numerator / fr.denominator
    Qf, Qw, ud = rs(r.meta["flux"].astype(np.float64)), rs(r.meta["wall_groups"].astype(np.float64)), rs(u)
    Pg, qg = recip["P"].astype(np.float64), recip["q"].astype(np.float64)
    nf = 1 << int(np.ceil(np.log2(Qf.shape[1] + Pg.shape[1])))
    f = np.fft.rfftfreq(nf, 1 / fs)
    Qg = np.fft.rfft(qg, nf)
    need = np.unique(np.concatenate([mouth_g, uniq]))
    Gd = {int(k): v for k, v in zip(need, np.fft.rfft(Pg[need], nf) / Qg)}
    Ud = np.fft.rfft(ud, nf)
    QF = np.fft.rfft(Qf, nf)
    Pmouth = sum(Gd[int(k)] * QF[j] for j, k in enumerate(mouth_g))
    QW = np.fft.rfft(Qw, nf)
    Hwall = {}
    for t in np.unique(gtis):
        sel = np.flatnonzero(gtis == t)
        Hwall[T3.TISSUE[t]] = sum(Gd[int(uniq[j])] * QW[j] for j in sel) / Ud
    Hmouth = Pmouth / Ud
    Hm = Hmouth + sum(Hwall.values()) if Hwall else Hmouth
    rho = AIR.props(35.5, 1.0).cgs()["RHO"]
    UL3 = np.fft.rfft(Qf.sum(0), nf) / Ud
    Hmono = 1j * 2 * np.pi * f * rho * UL3 / (4 * np.pi * 20.0)
    UL1, H1, Zin1 = chain_1d(cp, f)
    # 성문 → 마이크 임펄스 응답 (96 kHz, 20 kHz 위는 음원이 약해 자른다)
    Hc = Hm * (f < 19000)
    ir = np.fft.irfft(Hc, nf)[: int(0.04 * fs)]
    meta = dict(n_mouth=len(fc3), mouth_area=mouth_area, n_skin=len(uniq), sec=time.time() - t0,
                vol=g.meta["vol_tract"], exit_point=ep.tolist())
    if verbose:
        print(f"    3D {N} 걸음, {meta['sec']:.0f} s, 입 단면 {meta['mouth_area']:.2f} cm², 성도 부피 {meta['vol']:.1f} cm³")
    return dict(f=f, Hm=Hm, Hmouth=Hmouth, Hwall=Hwall, Hmono=Hmono, H1=H1, UL3=UL3, UL1=UL1, ir=ir, fs=fs, meta=meta)


def peaks_bw(f, H, fmin=150.0, fmax=6000.0, prom=3.0, n=6):
    """봉우리 주파수와 −3 dB 대역폭."""
    from scipy.signal import find_peaks
    db = 20 * np.log10(np.abs(H) + 1e-30)
    m = (f > fmin) & (f < fmax)
    idx = np.flatnonzero(m)
    pk, _ = find_peaks(db[m], prominence=prom)
    out = []
    for p in pk[:n]:
        i = idx[p]
        top = db[i] - 3
        lo, hi = i, i
        while lo > 0 and db[lo] > top:
            lo -= 1
        while hi < len(f) - 1 and db[hi] > top:
            hi += 1
        out.append((f[i], f[hi] - f[lo]))
    return out
