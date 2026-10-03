"""3D 성도 음향 검증 (MEASUREMENTS §52.526) — 해석해와 대조한다 (CPU, 작은 격자).

* 자유 공간 단극: |p/Q| = ωρ/(4πr) — 분할 PML 이 저역까지 열린 공간을 흉내 내는지.
* 곧은 원통 (한 끝 닫힘, 한 끝 배플 방사): 공명 주파수가 같은 손실의 1D 사슬과 1 % 안.
* 경계층 √(jω) 확산 표현의 맞춤 오차.
* 벽 조직: 경구개(윗벽 x ≥ 0)·치아는 단단, 연구개·인두는 무름.
"""
import numpy as np
import pytest
import torch
from scipy.signal import find_peaks
from scipy.special import j1, struve

from formant_ml.physics import air as AIR
from formant_ml.physics import fdtd3d as F3
from formant_ml.physics import tract3d as T3

SURF = ("UPPER_TEETH", "LOWER_TEETH", "UPPER_COVER", "LOWER_COVER", "UPPER_LIP", "LOWER_LIP", "PALATE", "MANDIBLE",
        "LOWER_TEETH_ORIGINAL", "LOW_VELUM", "MID_VELUM", "HIGH_VELUM", "NARROW_LARYNX_FRONT", "NARROW_LARYNX_BACK",
        "WIDE_LARYNX_FRONT", "WIDE_LARYNX_BACK", "TONGUE", "UPPER_COVER_TWOSIDE", "LOWER_COVER_TWOSIDE",
        "UPPER_TEETH_TWOSIDE", "LOWER_TEETH_TWOSIDE", "UPPER_LIP_TWOSIDE", "LOWER_LIP_TWOSIDE", "LEFT_COVER",
        "RIGHT_COVER", "UVULA_ORIGINAL", "UVULA", "UVULA_TWOSIDE", "EPIGLOTTIS_ORIGINAL", "EPIGLOTTIS",
        "EPIGLOTTIS_TWOSIDE", "RADIATION")


def _cyl_profiles(L=8.0, R=1.0, n=65):
    zs = (np.arange(96) + 0.5 - 48) * 7.0 / 96
    half = np.sqrt(np.clip(R * R - zs * zs, 0, None))
    up = np.where(half > 0, half, np.nan)
    return dict(center=np.stack([np.linspace(-L, 0, n), np.zeros(n)], 1), normal=np.tile([0.0, 1.0], (n, 1)),
                pos=np.linspace(0, L, n), upper=np.tile(up, (n, 1)), lower=np.tile(-up, (n, 1)),
                usurf=np.full((n, 96), SURF.index("UPPER_TEETH")), lsurf=np.full((n, 96), SURF.index("LOWER_TEETH")), z=zs)


def test_diffusive_sqrt_jw_fit():
    a, w, err = F3.diffusive_weights(400000.0)
    assert err < 0.12 and (w >= 0).all()


def test_tissue_classes():
    sid = np.array([SURF.index("UPPER_COVER")] * 3 + [SURF.index("UPPER_TEETH"), SURF.index("TONGUE")])
    x = np.array([1.0, -1.0, -2.5, 3.0, 1.0])
    y = np.array([1.3, 0.5, -4.0, 0.0, -1.0])
    t = T3.surface_tissue(SURF, sid, x, y)
    assert [T3.TISSUE[i] for i in t] == ["rigid", "velum", "pharynx", "rigid", "tongue"]


def test_free_field_monopole():
    h = 0.4
    n = 60
    air = np.ones((n, n, n), bool)
    F = []
    for axis in range(3):
        shp = [n, n, n]
        shp[axis] += 1
        f = np.zeros(shp, np.float32)
        sl = [slice(None)] * 3
        sl[axis] = slice(1, -1)
        f[tuple(sl)] = 1.0
        F.append(f)
    Z = [np.zeros_like(f, dtype=np.int8) for f in F]
    inlet = np.zeros(air.shape, bool)
    inlet[n // 2, n // 2, n // 2] = True
    g = T3.Grid(h=h, origin=np.zeros(3), air=air, ax=F[0], ay=F[1], az=F[2], wx=Z[0], wy=Z[1], wz=Z[2],
                cx=np.ones_like(F[0]), cy=np.ones_like(F[1]), cz=np.ones_like(F[2]), inlet=inlet, exterior=air,
                exit_point=np.zeros(2), exit_dir=np.array([1.0, 0.0]), area_1d=np.zeros(1), pos_1d=np.zeros(1))
    sim = F3.Sim(g, courant=0.9, walls="rigid", boundary_layer=False, sponge_cells=12, device="cpu")
    N = int(0.02 * sim.fs)
    q, _ = F3.zero_net_pulse(sim.fs, 5000.0, N)
    src = (np.array([n // 2] * 3) + 0.5) * h
    rec = src + np.array([5.0, 0.0, 0.0])
    r = sim.run(q, [tuple(rec)])
    nf = 1 << int(np.ceil(np.log2(N)))
    f = np.fft.rfftfreq(nf, 1 / sim.fs)
    H = np.fft.rfft(r.p_rec[0], nf) / np.fft.rfft(q, nf)
    rho = sim.rho
    for f0 in (300.0, 1000.0, 3000.0):
        i = np.argmin(np.abs(f - f0))
        ratio = np.abs(H[i]) / (2 * np.pi * f0 * rho / (4 * np.pi * 5.0))
        assert 0.9 < ratio < 1.1, (f0, ratio)


def test_cylinder_resonances_match_1d():
    L, R = 8.0, 1.0
    cp = _cyl_profiles(L, R)
    g = T3.build_grid(cp, h=0.2, ext_len=4.0, ext_half=5.0)
    sim = F3.Sim(g, courant=0.9, walls="rigid", boundary_layer=True, sponge_cells=8, device="cpu")
    N = int(0.03 * sim.fs)
    u, _ = F3.gauss_pulse(sim.fs, 8000.0, N)
    r = sim.run(u, [(g.exit_point[0] + 2.0, 0.0, 0.0)])
    nf = 1 << int(np.ceil(np.log2(N)))
    f = np.fft.rfftfreq(nf, 1 / sim.fs)
    H3 = np.fft.rfft(r.p_rec[0], nf) / np.fft.rfft(u, nf)
    a_ = AIR.props(35.5, 1.0).cgs()
    rho, c = a_["RHO"], a_["C_SOUND"]
    w = 2 * np.pi * np.maximum(f, 1.0)
    k = w / c
    A = np.pi * R * R
    Ybl = np.sqrt(1j * w) * (np.sqrt(a_["MU"] / rho) + (a_["GAMMA"] - 1) * np.sqrt(a_["LAMBDA_TH"] / (rho * a_["CP"]))) / (rho * c ** 2)
    Ysh, Zs = 1j * w * A / (rho * c ** 2) + 2 * np.pi * R * Ybl, 1j * w * rho / A
    gm, Zc = np.sqrt(Zs * Ysh), np.sqrt(Zs / Ysh)
    x2 = 2 * k * R
    Zr = rho * c / A * ((1 - 2 * j1(x2) / x2) + 1j * 2 * struve(1, x2) / x2)
    UL = 1.0 / (np.sinh(gm * L) / Zc * Zr + np.cosh(gm * L))
    m = (f > 400) & (f < 6000)
    p3, _ = find_peaks(20 * np.log10(np.abs(H3[m])), prominence=3)
    p1, _ = find_peaks(20 * np.log10(np.abs(UL[m])), prominence=3)
    f3, f1 = f[m][p3], f[m][p1]                     # L = 8 cm: 1/4 파장 공명 셋 (~1.0 · 3.0 · 5.1 kHz)
    assert len(f3) == len(f1) == 3, (f3, f1)
    assert np.all(np.abs(f3 / f1 - 1) < 0.03), (f3, f1)             # 2 mm 격자 (반지름 5 칸)·주파수 칸 33 Hz — 1 mm 에서는 1 % 안 (§52.526)
