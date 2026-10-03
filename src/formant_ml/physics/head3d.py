"""머리·목·몸통과 20 cm 마이크까지의 방사 — 상반성으로 구한 표면→마이크 전달 (MEASUREMENTS §52.526).

사용자 (2026-10-02): *"가장 중요한 건, 측정이 화자랑 딱 붙어있는 게 아니니, 물리적 거리도 고려해야 해. 화자로부터 20cm 정도로 가정하자."* 1D 관의 출력은
입술·콧구멍 유량의 시간 미분(먼 장 단극) 하나였다 — 거리·근거리 항(kr ≈ 1 이 20 cm 에서 ~280 Hz)·머리의 회절·몸통 반사·벽 진동의 방사가 없다.

**상반성**: 단단한 몸 바깥의 선형 음장에서 점 B 의 부피 유량 Q_B 가 점 A 에 내는 압력 p_A 는 Q_A 가 B 에 내는 압력과 같다 (p_A/Q_B = p_B/Q_A).
마이크 자리에 점 음원을 한 번 두고 몸 표면 바로 앞 점들(입·콧구멍·볼·목·턱 밑)의 압력을 받으면 G(ω, 표면 점) = 표면 점의 부피 유량 → 마이크 압력
이다. 입의 방사 = Σ G · (입 단면의 면 유량), 벽 방사 = Σ G · (벽 속도 × 넓이), 코 = G · 콧구멍 유량.

몸 모형 (잠정, 인체 계측 어림 — 실제 머리 스캔으로 바꿀 것): 머리 = 타원체 (앞뒤 18.3, 위아래 22, 너비 15.2 cm, 젊은 한국 여성 머리 길이·너비
어림), 앞면이 입술 끝에 닿게; 목 = 지름 10 cm 원기둥; 몸통 = 어깨 너비 36·두께 20 cm 상자 (목 밑 12 cm 부터); 코 = 입술 위 쐐기. 피부는
공기에 비해 음향적으로 단단하다 (특성 임피던스 ~1.5 MRayl ≫ 415 Rayl) — 반사면.
좌표는 VTL (x 앞, y 위, z 옆) [cm].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 몸 치수 [cm] — 잠정 (근거: 인체 계측 어림; 실측 머리 모양으로 바꿀 것).
HEAD = dict(ax=9.15, ay=11.0, az=7.6, y0=3.5, neck_r=5.0, neck_x=-2.5, neck_top=-3.0, torso_top=-15.0,
            torso_half_w=18.0, torso_x0=-12.0, torso_x1=8.0, nose_len=2.0, nose_half_w=1.6, nose_y0=0.6, nose_y1=4.6)


def body_indicator(X, Y, Z, exit_point, H=HEAD):
    """몸 안이면 True. 머리 타원체의 앞면이 입술 끝 높이에서 입술 끝 x 에 오게 가운데 x0 를 둔다."""
    yl = exit_point[1]
    x0 = exit_point[0] - H["ax"] * np.sqrt(max(1e-6, 1.0 - ((yl - H["y0"]) / H["ay"]) ** 2))
    head = ((X - x0) / H["ax"]) ** 2 + ((Y - H["y0"]) / H["ay"]) ** 2 + (Z / H["az"]) ** 2 <= 1.0
    neck = ((X - H["neck_x"]) ** 2 + Z ** 2 <= H["neck_r"] ** 2) & (Y <= H["neck_top"]) & (Y >= H["torso_top"] - 1)
    torso = (Y <= H["torso_top"]) & (np.abs(Z) <= H["torso_half_w"]) & (X >= H["torso_x0"]) & (X <= H["torso_x1"])
    # 코: 입술 위, 얼굴 앞에서 nose_len 만큼 나온 쐐기 (아래가 가장 높이 나온다)
    xf = exit_point[0]
    t = np.clip((H["nose_y1"] - Y) / (H["nose_y1"] - H["nose_y0"]), 0.0, 1.0)
    nose = (Y >= H["nose_y0"]) & (Y <= H["nose_y1"]) & (np.abs(Z) <= H["nose_half_w"] * (0.4 + 0.6 * t)) & \
           (X <= xf - 0.3 + H["nose_len"] * t) & (X >= xf - 3.0)
    return head | neck | torso | nose, x0


@dataclass
class Exterior:
    h: float
    origin: np.ndarray
    body: np.ndarray       # (nx, ny, nz) bool
    mic: np.ndarray        # (3,)
    x0: float


def build_exterior(exit_point, h=0.2, mic_dist=20.0, mic_dir=(1.0, 0.0), margin=9.0, H=HEAD) -> Exterior:
    ep = np.asarray(exit_point, float)
    md = np.asarray(mic_dir, float) / np.linalg.norm(mic_dir)
    mic = np.array([ep[0] + mic_dist * md[0], ep[1] + mic_dist * md[1], 0.0])
    lo = np.array([-16.0, H["torso_top"] - 10.0, -H["torso_half_w"] - margin])
    hi = np.array([mic[0] + margin, H["y0"] + H["ay"] + margin, H["torso_half_w"] + margin])
    n = np.ceil((hi - lo) / h).astype(int)
    xs, ys, zs = (lo[a] + (np.arange(n[a]) + 0.5) * h for a in range(3))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    body, x0 = body_indicator(X, Y, Z, ep, H)
    return Exterior(h=h, origin=lo, body=body, mic=mic, x0=x0)


def surface_points(ext: Exterior, region: str, exit_point, **kw):
    """몸 표면 바로 앞(첫 공기 칸) 점들 — region: 'mouth' (입술 끝 둘레 판), 'face', 'neck', 'all'. 반환 (칸 번호 (n, 3), 바깥 법선 (n, 3))."""
    from scipy.ndimage import binary_dilation, gaussian_filter
    b = ext.body
    shell = binary_dilation(b, iterations=1) & ~b
    idx = np.argwhere(shell)
    c = ext.origin + (idx + 0.5) * ext.h
    sm = gaussian_filter(b.astype(np.float32), 1.5)
    g = np.stack(np.gradient(sm), -1)[tuple(idx.T)]
    nrm = -g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    ep = np.asarray(exit_point)
    if region == "mouth":
        r = kw.get("radius", 2.5)
        m = (np.abs(c[:, 1] - ep[1]) <= r) & (np.abs(c[:, 2]) <= r) & (c[:, 0] > ep[0] - 1.0) & (nrm[:, 0] > 0.5)
    elif region == "face":
        m = (c[:, 0] > ext.x0) & (c[:, 1] > -6.0) & (c[:, 1] < 6.0)
    elif region == "neck":
        m = (c[:, 1] < -2.0) & (c[:, 1] > -15.0)
    else:
        m = np.ones(len(c), bool)
    return idx[m], nrm[m]


def reciprocity(exit_point, h: float = 0.2, ms: float = 30.0, pml: int = 25, verbose: bool = True, stl: str | None = None) -> dict:
    """마이크(입술 끝 앞 20 cm)에 순 부피 0 펄스 → 몸 표면 바로 앞 점(입·얼굴·목)의 압력 (96 kHz). `tf3d.glottis_to_mic` 의 recip 입력."""
    import time
    import torch
    from fractions import Fraction
    from scipy.signal import resample_poly
    from . import fdtd3d as F3
    from . import tract3d as T3
    ep = np.asarray(exit_point, float)
    ext = scan_exterior(stl, ep, h=h) if stl else build_exterior(ep, h=h)
    air = ~ext.body

    def faces(axis):
        shp = list(air.shape)
        shp[axis] += 1
        out = np.zeros(shp, np.float32)
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl = [slice(None)] * 3
        sl0[axis], sl1[axis], sl[axis] = slice(0, -1), slice(1, None), slice(1, -1)
        out[tuple(sl)] = (air[tuple(sl0)] & air[tuple(sl1)]).astype(np.float32)
        return out

    F = [faces(a) for a in range(3)]
    Z = [np.zeros_like(f, dtype=np.int8) for f in F]
    inlet = np.zeros(air.shape, bool)
    inlet[tuple(int(np.floor((ext.mic[a] - ext.origin[a]) / h)) for a in range(3))] = True
    g = T3.Grid(h=h, origin=ext.origin, air=air, ax=F[0], ay=F[1], az=F[2], wx=Z[0], wy=Z[1], wz=Z[2],
                cx=np.ones_like(F[0]), cy=np.ones_like(F[1]), cz=np.ones_like(F[2]), inlet=inlet, exterior=air,
                exit_point=ep, exit_dir=np.array([1.0, 0.0]), area_1d=np.zeros(1), pos_1d=np.zeros(1))
    sim = F3.Sim(g, courant=0.9, walls="rigid", boundary_layer=False, sponge_cells=pml)
    idx, nrm, reg = [], [], []
    for k, r in enumerate(("mouth", "face", "neck")):
        i_, n_ = surface_points(ext, r, ep)
        idx.append(i_); nrm.append(n_); reg.append(np.full(len(i_), k))
    idx, nrm, reg = np.concatenate(idx), np.concatenate(nrm), np.concatenate(reg)
    flat = torch.as_tensor(np.ravel_multi_index(tuple(idx.T), air.shape), device=sim.dev)
    N = int(ms * 1e-3 * sim.fs)
    q, _ = F3.zero_net_pulse(sim.fs, 20000.0, N)
    buf = torch.zeros((len(idx), N), device=sim.dev, dtype=torch.float32)
    t0 = time.time()
    rec_pts = [tuple(ext.origin + (i_ + 0.5) * h) for i_ in idx]
    r = sim.run_graph(q, rec_pts)
    buf = torch.as_tensor(r.p_rec, dtype=torch.float32)
    fr = Fraction(96000.0 / sim.fs).limit_denominator(2000)
    P = resample_poly(buf.cpu().numpy().astype(np.float64), fr.numerator, fr.denominator, axis=1).astype(np.float32)
    qd = resample_poly(q, fr.numerator, fr.denominator)
    if verbose:
        print(f"    상반성: 격자 {air.shape}, {N} 걸음 {time.time() - t0:.0f} s, 표면 점 {len(idx)}")
    return dict(idx=idx, nrm=nrm, reg=reg, P=P, q=qd, fs=sim.fs * fr.numerator / fr.denominator, origin=ext.origin, h=h, mic=ext.mic, x0=ext.x0)


# ============================================================================================================================
# 실제 머리·몸통 스캔 (§52.526) — SONICOM HRTF 자료 (Imperial College, CC BY 4.0 / MIT, 3D 스캔 0.5 mm 점 구름 → 물 샐 틈 없는 STL).
# 동아시아(중국계) 18–24 세 여성 스캔을 쓴다 (`third_party/sonicom`). 스캔 축: +x 위, −y 앞 (정중선 앞 윤곽에 코끝 · 인중 · 입술 · 턱), z 옆, mm.
# ============================================================================================================================


def scan_landmarks(V: np.ndarray, front: int | None = -1) -> dict:
    """스캔 꼭짓점 (mm) 에서 앞 방향 · 정중선 · 코끝 · 입술 사이(stomion) 를 찾는다. 위 = +x.
    코는 좁고 날카로운 돌출이다 — 정중선 윤곽에서 넓게(80 mm) 고른 성분을 뺀 나머지의 봉우리가 큰 쪽이 앞, 그 봉우리가 코끝 (뒤통수의 넓은 볼록을
    코로 잡지 않게). 입술 사이 = 코끝 아래 20–45 mm 에서 윗입술 · 아랫입술 봉우리 사이의 오목 (나머지의 최소)."""
    from scipy.ndimage import median_filter
    top = V[:, 0].max()
    head = V[V[:, 0] > top - 300]
    zc = float(np.median(head[:, 2]))
    mid = head[np.abs(head[:, 2] - zc) < 3]
    xs = np.arange(top - 300, top, 2.0)
    best = None
    # SONICOM 스캔은 모두 얼굴이 −y (정중 단면 10 개를 눈으로 확인, §52.526) — 올림머리의 날카로운 돌출을 코로 잡지 않게 앞을 고정한다.
    for sgn in ((1, -1) if front is None else (front,)):
        pr = np.array([(mid[np.abs(mid[:, 0] - x0) < 1.5][:, 1] * sgn).max() if (np.abs(mid[:, 0] - x0) < 1.5).any() else np.nan for x0 in xs])
        prf = np.where(np.isfinite(pr), pr, np.nanmin(pr))
        res = prf - median_filter(prf, size=41, mode="nearest")
        win = (xs > top - 230) & (xs < top - 80)
        i = int(np.argmax(np.where(win, res, -np.inf)))
        if best is None or res[i] > best[0]:
            best = (res[i], sgn, i, pr, res)
    prom, sgn, i_nose, pr, res = best
    x_nose = xs[i_nose]
    # 코끝 아래로 해부 순서: 인중(골) → 윗입술(봉우리) → 입술 사이(골). 1 mm 간격 윤곽의 극점 (돌출 ≥ 1 mm) — P0156 에서 코끝 152 · 인중 160 ·
    # 윗입술 179 · 입술 사이 187 · 아랫입술 195 mm (꼭대기 아래) 로 확인. 못 찾으면 코끝 − 35 mm.
    from scipy.signal import find_peaks
    xf = np.arange(x_nose - 60, x_nose + 1, 1.0)
    pf = np.array([(mid[np.abs(mid[:, 0] - x0) < 0.75][:, 1] * sgn).max() if (np.abs(mid[:, 0] - x0) < 0.75).any() else np.nan for x0 in xf])
    pf = np.where(np.isfinite(pf), pf, np.nanmin(pf))
    pk, _ = find_peaks(pf, prominence=1.0)
    tr, _ = find_peaks(-pf, prominence=1.0)
    order = sorted([(xf[i], "p") for i in pk] + [(xf[i], "t") for i in tr], reverse=True)    # 위에서 아래로
    want, x_st = ["t", "p", "t"], None
    k = 0
    for xv, kind in order:
        if xv >= x_nose - 2:
            continue
        if kind == want[k]:
            k += 1
            if k == 3:
                x_st = xv
                break
    if x_st is None:
        x_st = x_nose - 35.0
    i_st = int(np.argmin(np.abs(xs - x_st)))
    return dict(top=top, zc=zc, front=sgn, x_nose=x_nose, y_nose=pr[i_nose] * sgn, x_stom=float(x_st), y_stom=float(pr[i_st] * sgn),
                nose_prom=prom, profile=(xs, pr))


def scan_exterior(stl: str, exit_point, h: float = 0.2, mic_dist: float = 20.0, margin: float = 9.0) -> Exterior:
    """스캔 몸을 VTL 좌표 (cm; x 앞, y 위, z 옆) 로 옮겨 입술 사이를 입술 끝 (exit_point) 에 맞추고 h 격자에 채운다."""
    import trimesh
    m = trimesh.load(stl)
    L = scan_landmarks(m.vertices)
    sg = L["front"]
    # 스캔 (mm) → VTL (cm): X = sg·y (앞이 +), Y = x (위), Z = z − zc. 입술 사이 → exit_point
    M = np.zeros((4, 4))
    M[0, 1] = sg * 0.1
    M[1, 0] = 0.1
    M[2, 2] = 0.1
    M[3, 3] = 1.0
    if np.linalg.det(M[:3, :3]) < 0:              # 손잡이(오른손 좌표) 유지
        M[2, 2] = -0.1
    st = M[:3, :3] @ np.array([L["x_stom"], L["y_stom"], L["zc"]])
    M[:3, 3] = np.array([exit_point[0], exit_point[1], 0.0]) - st
    m.apply_transform(M)
    ep = np.asarray(exit_point, float)
    mic = np.array([ep[0] + mic_dist, ep[1], 0.0])
    b = m.bounds
    lo = np.array([b[0, 0] - 2.0, b[0, 1] - 2.0, b[0, 2] - 2.0])
    hi = np.array([max(b[1, 0], mic[0]) + margin, b[1, 1] + 2.0, b[1, 2] + 2.0])
    vox = m.voxelized(pitch=h).fill()
    n = np.ceil((hi - lo) / h).astype(int)
    body = np.zeros(n, bool)
    pts = vox.points                              # 찬 복셀 가운데 (VTL cm)
    ijk = np.floor((pts - lo) / h).astype(int)
    ok = np.all((ijk >= 0) & (ijk < n), 1)
    body[tuple(ijk[ok].T)] = True
    from scipy.ndimage import binary_closing
    body = binary_closing(body, iterations=1)
    ext = Exterior(h=h, origin=lo, body=body, mic=mic, x0=float(ep[0] - 9.0))
    ext.landmarks = L
    ext.transform = M
    return ext
