"""3D 성대 유한요소(`physics.vocal_fold_solid`)에서 모드 3 성대의 화자 상수를 잰다 (MEASUREMENTS §52.537).

사용자: *"내가 13만원이나 내고 만든 모델이니만큼 무조건 써야 해 … 1-3시간 안에 적합이 끝날 수 있도록 개편해봐"*. 결합 3D 참조 모형은 0.21 µs 한
걸음에 15–26 s 라 적합 고리에 넣을 수 없다. 대신 그 **고체**가 적합에 쓰는 성대를 정한다:

1. 근육 자세 격자 (늘어남 ε × 갑상피열근 활성) 마다 3D 고체의 정적 평형 · 접선 강성 · 점성 · 질량에서 길이 첫 조화 sin(πx/L) 모드의
   진동수 · 감쇠비 · 상하 모양, 내측 균일 압력에 대한 정적 순응도, 갑상피열근 불룩함을 잰다 (`physics.fold_modal`).
2. 모드 3 (보–막 N 줄 덮개 + 몸체, `beam_membrane.ns_coefs`) 의 화자 배율 6 개 (길이 · 두께 · 층간 결합 · 감쇠 · 불룩함 · 전단) 를 같은 양에
   최소제곱으로 맞춘다. 같은 조직 법칙(`fold_rules`, serry2026)을 쓰므로 ε · 활성의 뜻이 같다.
3. 결과를 `copyfit --vf-ns-fem` 이 읽는 JSON 으로 쓴다 — 적합은 이 상수를 고정한다.

    python scripts/fem_fold_calib.py --out profiles/vf/fem_ns.json

이것은 작은 진폭의 선형화 일치다. 접촉 · 흐름 · 자기 진동의 일치는 아니다 (그것은 결합 참조 모형의 짧은 구간 대조로 따로 본다).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import scipy.linalg as sla
from scipy.optimize import least_squares
import torch

from formant_ml.physics import beam_membrane as BM
from formant_ml.physics.fold_modal import FoldLinearizer, build_solid

NAMES = ("len", "thick", "kc", "zeta", "bulge", "shear")
BOUNDS = {"len": (0.5, 2.0), "thick": (0.3, 2.0), "kc": (0.05, 20.0), "zeta": (0.01, 0.5),
          "bulge": (0.01, 10.0), "shear": (0.05, 20.0)}


def mode3(eps, ta, sc, nstrip=6):
    """모드 3 (접촉 없음, 긴장 배율 1) 의 첫 두 모드: 진동수 [Hz], 감쇠비, 상하 모양 (아래 → 위), 순응도 [m/Pa], 불룩함 [m, + 안쪽]."""
    saved = BM.STATIC_BULGE
    BM.STATIC_BULGE = True
    try:
        co = BM.ns_coefs(eps, ta, nstrip=nstrip, scale_len=sc["len"], scale_thick=sc["thick"], k_scale=sc["kc"],
                         zeta=sc["zeta"], shear_scale=sc["shear"])
    finally:
        BM.STATIC_BULGE = saved
    cT, cG, cK, cB, ms, mb, fB, L, b, bet, cC, ceta, cF = co
    fB = fB * sc["bulge"]
    N, nb = nstrip, nstrip
    Rr = float(BM.vf_static_ns()[15])
    K, C, M = np.zeros((N + 1, N + 1)), np.zeros((N + 1, N + 1)), np.zeros(N + 1)
    for k in range(N):
        M[k] = ms / N
        prof = Rr + (1.0 - Rr) * (k + 0.5) / N
        K[k, k] += cT / N
        C[k, k] += bet * cT / N
        kk, cc = cK * prof / N, cC / N
        for (i, j, s) in ((k, k, 1), (nb, nb, 1), (k, nb, -1), (nb, k, -1)):
            K[i, j] += s * kk
            C[i, j] += s * cc
        if k + 1 < N:
            for (i, j, s) in ((k, k, 1), (k + 1, k + 1, 1), (k, k + 1, -1), (k + 1, k, -1)):
                K[i, j] += s * cG
                C[i, j] += s * ceta
    M[nb] = mb
    K[nb, nb] += cB
    C[nb, nb] += cF
    w2, V = sla.eigh(K, np.diag(M))
    freq = np.sqrt(np.maximum(w2[:2], 0.0)) / (2 * np.pi)
    zeta = np.array([V[:, i] @ C @ V[:, i] / (2 * math.sqrt(max(w2[i], 1e-30)) * (V[:, i] @ (M * V[:, i]))) for i in range(2)])
    prof = []
    for i in range(2):
        v = V[:N, i]
        prof.append(v / np.abs(v).max() * np.sign(v[np.argmax(np.abs(v))]))
    dy = b / N
    f = np.zeros(N + 1)
    f[:N] = 10.0 * dy * 2.0 * L / math.pi            # 1 Pa = 10 dyn/cm², sin 모양의 일반화 힘
    comp = np.linalg.solve(K, f)[:N] / 100.0         # cm → m (+ 바깥)
    fs = np.zeros(N + 1)
    fs[nb] = fB
    bulge = -np.linalg.solve(K, fs)[:N] / 100.0
    return freq, zeta, np.asarray(prof), comp, bulge


def fem_table(eps_grid, ta_grid, mesh):
    solid = build_solid(**mesh)
    rows = []
    for ta in ta_grid:
        lin = FoldLinearizer(solid)
        for eps in eps_grid:
            t0 = time.time()
            lin.solve(eps, ta)
            m = lin.summary()
            if len(m.freq_hz) < 2:
                raise ValueError(f"fewer than two AP-first-harmonic FEM modes at eps {eps}, ta {ta}")
            rows.append(m)
            print(f"  3D 고체 ε {eps:+.2f} 활성 {ta:.2f}: 모드 {m.freq_hz[0]:6.1f} · {m.freq_hz[1]:6.1f} Hz, 감쇠비 {m.damping_ratio[0]:.3f}, "
                  f"순응도 {1e9 * m.compliance_m_per_pa.mean():.0f} µm/kPa, 불룩 {1e6 * m.bulge_m.mean():+.1f} µm  ({time.time() - t0:.1f} s)",
                  flush=True)
    return rows


EPS_SEARCH = (-0.3, 1.2)


def matched_eps(f_target, ta, sc, nstrip):
    """모드 3 의 첫 모드가 `f_target` 이 되는 늘어남 (이분법). 적합에서 음높이는 f0 → ε 표가 정하므로 두 모형은 ε 가 아니라
    **같은 음높이**에서 견준다. 범위 안에 해가 없으면 가까운 끝."""
    lo, hi = EPS_SEARCH
    f_lo, f_hi = mode3(lo, ta, sc, nstrip)[0][0], mode3(hi, ta, sc, nstrip)[0][0]
    if f_target <= f_lo:
        return lo
    if f_target >= f_hi:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mode3(mid, ta, sc, nstrip)[0][0] < f_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def residuals(logs, rows, nstrip, weights):
    sc = {n: math.exp(v) for n, v in zip(NAMES, logs)}
    out = []
    for m in rows:
        e = matched_eps(m.freq_hz[0], m.ta, sc, nstrip)
        f, z, prof, comp, bulge = mode3(e, m.ta, sc, nstrip)
        zc = (np.arange(nstrip) + 0.5) / nstrip
        zf = np.linspace(0.0, 1.0, m.medial_profile.shape[1])
        out.append(weights["freq"] * (math.log(f[0]) - math.log(m.freq_hz[0])))
        out.append(weights["ratio"] * (math.log(f[1] / f[0]) - math.log(m.freq_hz[1] / m.freq_hz[0])))
        out.append(weights["zeta"] * (z[0] - m.damping_ratio[0]))
        for i in range(2):
            target = np.interp(zc, zf, m.medial_profile[i])
            err = min(np.linalg.norm(prof[i] - target), np.linalg.norm(prof[i] + target))
            out.append(weights["shape"] * err / math.sqrt(nstrip))
        out.append(weights["comp"] * (math.log(comp.mean()) - math.log(np.interp(zc, zf, m.compliance_m_per_pa).mean())))
        if m.ta > 0:
            fb = np.interp(zc, zf, m.bulge_m).mean()
            out.append(weights["bulge"] * (math.log(max(bulge.mean(), 1e-12)) - math.log(max(fb, 1e-12))))
    return np.asarray(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps", default="0.1,0.2,0.3,0.4,0.5,0.6", help="늘어남 격자")
    ap.add_argument("--ta", default="0,0.5", help="갑상피열근 활성 격자")
    ap.add_argument("--n-ap", type=int, default=8)
    ap.add_argument("--n-si", type=int, default=6)
    ap.add_argument("--cells", default="2,2,2", help="덮개 · 인대 · 갑상피열근 층의 칸 수")
    ap.add_argument("--nstrip", type=int, default=6, help="모드 3 덮개 줄 수 (tube_td.VF_NSTRIP)")
    ap.add_argument("--geometry", choices=("fem", "fit"), default="fem",
                    help="fem: 길이·두께 배율을 3D 고체의 기하(길이·SI 높이 대 모드 3 여성 기준)로 고정하고 나머지 넷만 맞춘다. fit: 여섯 모두 맞춘다")
    ap.add_argument("--out", default="profiles/vf/fem_ns.json")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    eps_grid = [float(v) for v in a.eps.split(",")]
    ta_grid = [float(v) for v in a.ta.split(",")]
    mesh = dict(n_ap=a.n_ap, n_si=a.n_si, cells_per_layer=tuple(int(v) for v in a.cells.split(",")))
    print(f"3D 성대 유한요소 선형화: 격자 {mesh}, ε {eps_grid}, 활성 {ta_grid}", flush=True)
    rows = fem_table(eps_grid, ta_grid, mesh)
    weights = dict(freq=1.0, ratio=3.0, zeta=1.0, shape=0.3, comp=0.5, bulge=0.3)
    x_all = np.log([1.0, 1.0, 3.0, 0.1, 0.1, 1.0])
    fixed = {}
    if a.geometry == "fem":
        L0 = BM.SERRY_MALE["L0"] * BM.FEMALE_LEN
        b0 = BM.SERRY_MALE["b0"] * BM.FEMALE_THICK
        fixed = {"len": rows[0].length_m / L0, "thick": rows[0].height_m / b0}
        print(f"  기하 고정: 3D 길이 {1e3 * rows[0].length_m:.2f} mm / 모드 3 {1e3 * L0:.3f} mm → len {fixed['len']:.4g}, "
              f"SI 높이 {1e3 * rows[0].height_m:.2f} mm / {1e3 * b0:.3f} mm → thick {fixed['thick']:.4g}", flush=True)
    free = [i for i, n in enumerate(NAMES) if n not in fixed]
    for i, n in enumerate(NAMES):
        if n in fixed:
            x_all[i] = math.log(fixed[n])

    def full(xf):
        x = x_all.copy()
        x[free] = xf
        return x

    lo = np.log([BOUNDS[NAMES[i]][0] for i in free])
    hi = np.log([BOUNDS[NAMES[i]][1] for i in free])
    fit = least_squares(lambda xf: residuals(full(xf), rows, a.nstrip, weights), x_all[free], bounds=(lo, hi), x_scale=1.0)
    sc = {n: float(math.exp(v)) for n, v in zip(NAMES, full(fit.x))}
    at_bound = [NAMES[i] for i, v, l, h in zip(free, fit.x, lo, hi) if v - l < 1e-3 or h - v < 1e-3]
    print("모드 3 화자 배율 (3D 고체에 맞춤): " + ", ".join(f"{n} {v:.4g}" for n, v in sc.items())
          + (f"  — 범위 끝: {', '.join(at_bound)}" if at_bound else ""), flush=True)
    table = []
    for m in rows:
        e = matched_eps(m.freq_hz[0], m.ta, sc, a.nstrip)
        f, z, prof, comp, bulge = mode3(e, m.ta, sc, a.nstrip)
        zc = (np.arange(a.nstrip) + 0.5) / a.nstrip
        zf = np.linspace(0.0, 1.0, m.medial_profile.shape[1])
        row = dict(eps=m.eps, ta=m.ta, mode3_eps=e, fem_hz=m.freq_hz[:2].tolist(), mode3_hz=f.tolist(),
                   fem_zeta=float(m.damping_ratio[0]), mode3_zeta=float(z[0]),
                   fem_compliance_um_per_kpa=float(1e9 * np.interp(zc, zf, m.compliance_m_per_pa).mean()),
                   mode3_compliance_um_per_kpa=float(1e9 * comp.mean()),
                   fem_bulge_um=float(1e6 * np.interp(zc, zf, m.bulge_m).mean()), mode3_bulge_um=float(1e6 * bulge.mean()))
        table.append(row)
        print(f"  ε {m.eps:+.2f}(모드3 {e:+.2f}) 활성 {m.ta:.2f}: 모드 3D {row['fem_hz'][0]:6.1f}/{row['fem_hz'][1]:6.1f} · 모드3 {f[0]:6.1f}/{f[1]:6.1f} Hz, "
              f"감쇠비 {row['fem_zeta']:.3f}/{row['mode3_zeta']:.3f}, 순응도 {row['fem_compliance_um_per_kpa']:.0f}/{row['mode3_compliance_um_per_kpa']:.0f} µm/kPa, "
              f"불룩 {row['fem_bulge_um']:+.1f}/{row['mode3_bulge_um']:+.1f} µm", flush=True)
    doc = dict(schema_version=1, geometry=a.geometry, fixed=fixed, source="physics.vocal_fold_solid nominal three-layer FEM (serry2026 tissue), small-amplitude linearization",
               mesh=mesh, nstrip=a.nstrip, eps=eps_grid, ta=ta_grid, weights=weights, scales=sc,
               log_scales={f"td_log_ns_{n}": math.log(v) for n, v in sc.items()},
               at_bound=at_bound, cost=float(fit.cost), table=table,
               note="Matches modal frequencies, damping, medial SI shape, static compliance and TA bulge; not contact/flow/self-oscillation.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"→ {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
