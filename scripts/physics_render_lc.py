"""자려 진동(한계 순환) 음원을 **끊김 없이** 성도에 물린다 — 목표 녹음과 직접 대조하기 위한 렌더.

`physics_render.py` 의 충격+링다운 판은 F0 구간마다 성대를 정지 상태에서 새로 시작해 이음매마다 클릭이
났다 (out/PHY/README.md). 여기서는

1. **F0 → 늘어남** 을 자려 진동 모형 자체로 교정한 표에서 뽑는다 (시작할 때 몇 점 시뮬레이션해 만든다).
2. 구간이 바뀌어 모드 집합이 달라져도 **변위장 전체를 질량 가중으로 이어받는다**: q_new = Φ_newᵀ M (Φ_old q_old) (§52.18).
   유량 U 도 그대로 이어받는다.
3. 200 kHz 로 적분하고 `resample_poly` 로 48 kHz 에 내린다 (앨리어싱 방지).

잡음·기식 가지는 아직 없다. 그래서 이 렌더의 고역은 목표보다 비어 있어야 정상이다 — 판정은 **배음 영역**에서 한다.

    python scripts/physics_render_lc.py out/M/M14/s101 --out out/PHY/lc_s101
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.engine.control import PARAM_NAMES                     # noqa: E402
from formant_ml.engine.tract import VocalTract                        # noqa: E402
from formant_ml.physics import self_oscillation as S                  # noqa: E402

CM_H2O = 98.0665


def calibrate(p_sub_cm, h0, strains=None, n_modes=6, seconds=0.12):
    """늘어남 격자에서 한계 순환의 F0 를 잰다. 반환 (strains, f0s) — F0 가 단조인 점만.

    예전 격자는 0.02~0.40 의 5 점이라 F0 246~420 Hz 밖이 잘렸고 사이는 선형 보간이었다 — 렌더 F0 가 목표에서
    중앙 +44 센트, 프레임의 60 % 가 50 센트 넘게 어긋났다 (§52.10). **짧아짐(ε < 0)** 까지 넓히고 촘촘히 한다.
    ε −0.15 에서도 세로 응력이 양수다 (σ ≈ 0.6 kPa).
    """
    strains = np.linspace(-0.15, 0.50, 14) if strains is None else strains
    out = []
    for e in strains:
        r = S.simulate(p_sub_cm=p_sub_cm, strain=float(e), h0=h0, seconds=seconds, n_modes=n_modes)
        if r.get("ok") and r["f0"] > 0:
            out.append((float(e), r["f0"]))
    out.sort(key=lambda x: x[0])
    return monotone_table([o[0] for o in out], [o[1] for o in out])


def monotone_table(strains, f0s):
    """늘어남 순으로 놓인 (ε, F0) 에서 **F0 가 지금까지의 최대보다 큰 점만** 남긴다.

    예전에는 바로 앞 원소하고만 비교해, 비단조 점(예: ε −0.05 가 아직 자리 잡지 않아 259 Hz 로 튐) 뒤에 오는
    246 Hz 같은 점이 남았다. 그러면 `np.interp` 가 오름차순이 아닌 표를 받아 **조용히 틀린 보간**을 한다.
    """
    es, fs_ = [], []
    top = -np.inf
    for e, f in zip(strains, f0s):
        if f > top:
            es.append(float(e)); fs_.append(float(f)); top = f
    return np.array(es), np.array(fs_)


def _recent_f0(U, end, fs_sim, span_s=0.03, min_cross=3):
    """끝에서 `span_s` 안의 닫힘→열림 교차로 실제 F0 를 잰다. 못 재면 0."""
    a = max(0, end - int(span_s * fs_sim))
    x = U[a:end]
    if len(x) < 8 or x.max() <= 0:
        return 0.0
    thr = 0.05 * x.max()
    on = np.flatnonzero((x[1:] > thr) & (x[:-1] <= thr))
    if len(on) < max(2, int(min_cross)):
        return 0.0
    T = np.diff(on)[-3:] / fs_sim
    return float(1.0 / np.median(T))


def update_correction(corr, eps_ff, err, slope, lo, hi, gain):
    """늘어남 되먹임 보정의 한 걸음 — **적분 와인드업을 막는다**.

    목표 F0 의 21 % 가 모형이 낼 수 있는 하한(230 Hz) 아래였다. 그런 목표를 향해 보정을 계속 적분하면 늘어남이
    하한에 붙은 채 풀려나지 못해, 되먹임을 넣은 렌더가 오히려 |차| 중앙 237 센트·90 % 715 센트로 망가졌다
    (§52.11). 보정은 늘어남이 [lo, hi] 안에 머물게 자르고, 이미 한계에 붙어 있으면 그 방향으로는 더 쌓지 않는다.
    """
    step = gain * err * slope
    eps = eps_ff + corr
    if (eps <= lo and step < 0) or (eps >= hi and step > 0):
        step = 0.0
    corr = corr + step
    return float(np.clip(corr, lo - eps_ff, hi - eps_ff))


#: 교정표를 정하는 **물리 상수** — 캐시 키에 넣고, 병렬 작업자에 그대로 넘긴다 (§52.43). 잡음 상수는 교정할 때 0 으로 둔다
#: (표는 평균 F0 를 담는다). Windows 의 작업자는 모듈을 새로 불러오므로 부모의 모듈 상수가 저절로 넘어가지 않는다.
PHYSICS_CONST_NAMES = (("self_oscillation", "K_CONTACT"), ("self_oscillation", "C_CONTACT"), ("self_oscillation", "ZETA"),
                       ("self_oscillation", "MUCUS_GAMMA"), ("self_oscillation", "MUCUS_MU"), ("self_oscillation", "FILM_H"),
                       ("self_oscillation", "FILM_W"), ("fold_modes", "SIGMA0"), ("fold_modes", "L0"), ("fold_modes", "T0"),
                       ("fold_modes", "D0"), ("fold_modes", "E_Z0"), ("fold_modes", "E_T"), ("fold_modes", "G_Z"),
                       ("fold_modes", "TA_THICK"), ("fold_modes", "TA_DEPTH"))
NOISE_CONST_NAMES = ("P_NOISE", "F_NOISE", "JET_P_NOISE")


def physics_consts():
    """지금 프로세스의 물리 상수 값 (이름 순서 고정)."""
    from formant_ml.physics import fold_modes as FM
    mods = {"self_oscillation": S, "fold_modes": FM}
    return tuple(float(getattr(mods[m], n)) for m, n in PHYSICS_CONST_NAMES)


def apply_physics_consts(values):
    """작업자에서 부모의 물리 상수를 되살리고 잡음을 끈다."""
    from formant_ml.physics import fold_modes as FM
    mods = {"self_oscillation": S, "fold_modes": FM}
    for (m, n), v in zip(PHYSICS_CONST_NAMES, values):
        setattr(mods[m], n, float(v))
    for n in NOISE_CONST_NAMES:
        setattr(S, n, 0.0)


def steady_f0(U, fs):
    """뒤 절반의 열림 교차 주기로 잰 **정상 상태** F0. 주기 변동이 5 % 를 넘으면(서지 않은 진동) nan.

    예전 `calibrate` 는 0.12 s 자기상관이라 과도를 잡았다 — ε −0.05 가 "아직 자리 잡지 않았다" 고 본 점은 0.3 s 에서도
    주기 변동 0.0 % 인 다른 한계 순환이었다 (§52.22).
    """
    x = np.asarray(U)[len(U) // 2:]
    if len(x) < 8 or x.max() <= 0:
        return float("nan")
    thr = 0.05 * x.max()
    i = np.flatnonzero((x[1:] > thr) & (x[:-1] <= thr))
    if len(i) < 6:
        return float("nan")
    T = np.diff(i).astype(float)
    if np.std(T) > 0.05 * np.mean(T):
        return float("nan")
    return float(fs / np.median(T))


def _posture_point(args):
    p_sub_cm, h0, eps, ta, n_modes, seconds, consts = args
    apply_physics_consts(consts)
    r = S.simulate(p_sub_cm=p_sub_cm, strain=eps, h0=h0, seconds=seconds, n_modes=n_modes, ta=ta)
    return steady_f0(r["U"], 200000.0) if r.get("ok") else float("nan")


#: 자세 교정 격자. ε −0.25 는 ta 0~0.67 에서 진동이 서지 않았다 (§52.22).
POSTURE_STRAINS = tuple(float(v) for v in np.round(np.arange(-0.20, 0.501, 0.05), 3))
POSTURE_TAS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
#: **자세 규칙** (물리가 아니라 제어 규칙) — 목표 F0 가 무릎 아래로 내려가면 TA 를 쓰기 시작해 F_LOW 에서 최대가 된다.
#: 늘어남만으로는 약 222 Hz 아래로 못 가고, 그 가장자리(ε −0.20, 그 아래는 진동이 안 섬) 에 붙이지 않으려고 TA 를 미리 조금씩 쓴다.
TA_KNEE_HZ = 260.0
TA_LOW_HZ = 180.0


def calibrate_posture(p_sub_cm, h0, n_modes=6, seconds=0.3, cache=None, workers=6):
    """(ta, ε) 격자의 정상 상태 F0 표 F[ta, ε] (진동이 안 서면 nan). `cache` 가 있고 설정이 같으면 불러온다."""
    consts = physics_consts()
    key = np.array([p_sub_cm, h0, n_modes, seconds, *POSTURE_STRAINS, *POSTURE_TAS, *consts], float)
    strains, tas = np.array(POSTURE_STRAINS), np.array(POSTURE_TAS)
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["key"], key):
            return strains, tas, z["F"]
    jobs = [(p_sub_cm, h0, float(e), float(t), n_modes, seconds, consts) for t in POSTURE_TAS for e in POSTURE_STRAINS]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            vals = pool.map(_posture_point, jobs)
    else:
        saved = {n: getattr(S, n) for n in NOISE_CONST_NAMES}
        try:
            vals = [_posture_point(j) for j in jobs]
        finally:
            for n, v in saved.items():
                setattr(S, n, v)
    F = np.array(vals, float).reshape(len(POSTURE_TAS), len(POSTURE_STRAINS))
    if cache:
        np.savez_compressed(cache, F=F, key=key)
    return strains, tas, F


def posture_ta(f0):
    return float(np.clip((TA_KNEE_HZ - f0) / (TA_KNEE_HZ - TA_LOW_HZ), 0.0, 1.0))


def posture_for(f0, strains, tas, F):
    """목표 F0 -> (ε, ta). ta 는 자세 규칙으로, ε 은 그 ta 에서 로그 F0 를 두 격자 행 사이로 보간한 표를 뒤집어 얻는다."""
    ta = posture_ta(f0)
    i = int(np.clip(np.searchsorted(tas, ta, side="right") - 1, 0, len(tas) - 2))
    w = (ta - tas[i]) / (tas[i + 1] - tas[i])
    with np.errstate(invalid="ignore", divide="ignore"):
        row = np.exp((1.0 - w) * np.log(F[i]) + w * np.log(F[i + 1]))
    ok = np.isfinite(row)
    es, fs_ = monotone_table(strains[ok], row[ok])
    return float(np.interp(np.log(f0), np.log(fs_), es)), ta


#: **성구(가성) 축** r ∈ [0, 1] — 흉성 자세표(ε, ta) 로 닿지 않는 고음에서만 쓴다 (§52.33). r 이 오르면
#: 진동 깊이가 1 → VD_FALSETTO 로 얕아지고(체부가 굳어 덮개만 떤다), 휴지 반틈새가 H0_FALSETTO_SHIFT 만큼 벌어지고(덜 닫힌 가성),
#: 폐압이 (1 + P_FALSETTO_GAIN·r) 배로 오른다(고음의 발성 문턱압). 셋 다 **제어 규칙**이지 물리가 아니다 — 가성 영역 훑기에서
#: 가장 얕은 판이 내전을 풀거나 폐압을 올려야 섰기 때문에 이 방향을 골랐다 (§52.32). r 0 행은 흉성 ta 0 행과 같다.
REGISTER_ROWS = (0.0, 0.25, 0.5, 0.75, 1.0)
REGISTER_STRAINS = tuple(float(v) for v in np.round(np.arange(0.10, 0.551, 0.05), 3))
VD_FALSETTO = 0.25
H0_FALSETTO_SHIFT = 0.04e-3
P_FALSETTO_GAIN = 0.5
#: 자세표 최대의 이 몫을 넘는 F0 에서 성구를 올린다 (표 가장자리에 붙지 않게).
REGISTER_MARGIN = 0.97
#: 성구 궤적을 부드럽게 하는 창 [ms] — 이동 최대(조금 일찍 올린다) 뒤 이동 평균.
REGISTER_SMOOTH_MS = 30.0


def register_params(r):
    """r -> (진동 깊이, 휴지 반틈새 이동 [m], 폐압 배율)."""
    r = float(np.clip(r, 0.0, 1.0))
    return 1.0 - (1.0 - VD_FALSETTO) * r, H0_FALSETTO_SHIFT * r, 1.0 + P_FALSETTO_GAIN * r


def _register_point(args):
    p_sub_cm, h0, eps, r, n_modes, seconds, consts = args
    apply_physics_consts(consts)
    vd, dh0, ps = register_params(r)
    res = S.simulate(p_sub_cm=p_sub_cm * ps, strain=eps, h0=h0 + dh0, seconds=seconds, n_modes=n_modes,
                     vib_depth=vd)
    return steady_f0(res["U"], 200000.0) if res.get("ok") else float("nan")


def calibrate_register(p_sub_cm, h0, n_modes=6, seconds=0.3, cache=None, workers=3):
    """(r, ε) 격자의 정상 상태 F0 표. 반환 (strains, rows, F[r, ε])."""
    consts = physics_consts()
    key = np.array([p_sub_cm, h0, n_modes, seconds, VD_FALSETTO, H0_FALSETTO_SHIFT, P_FALSETTO_GAIN,
                    *REGISTER_STRAINS, *REGISTER_ROWS, *consts], float)
    strains, rows = np.array(REGISTER_STRAINS), np.array(REGISTER_ROWS)
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["key"], key):
            return strains, rows, z["F"]
    jobs = [(p_sub_cm, h0, float(e), float(r), n_modes, seconds, consts) for r in REGISTER_ROWS for e in REGISTER_STRAINS]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            vals = pool.map(_register_point, jobs)
    else:
        saved = {n: getattr(S, n) for n in NOISE_CONST_NAMES}
        try:
            vals = [_register_point(j) for j in jobs]
        finally:
            for n, v in saved.items():
                setattr(S, n, v)
    F = np.array(vals, float).reshape(len(REGISTER_ROWS), len(REGISTER_STRAINS))
    if cache:
        np.savez_compressed(cache, F=F, key=key)
    return strains, rows, F


def _interp_row(F, rows, x):
    i = int(np.clip(np.searchsorted(rows, x, side="right") - 1, 0, len(rows) - 2))
    w = float(np.clip((x - rows[i]) / (rows[i + 1] - rows[i]), 0.0, 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp((1.0 - w) * np.log(F[i]) + w * np.log(F[i + 1]))


def _row_table(strains, row):
    ok = np.isfinite(row)
    if ok.sum() < 2:
        return None
    return monotone_table(strains[ok], row[ok])


def register_track(f0_track, voiced, strains, rows, F, chest_max, frame_ms,
                   candidates=np.linspace(0.05, 1.0, 20)):
    """프레임마다 **필요한 최소 성구** r — 흉성으로 닿으면 0, 아니면 그 F0 를 표 안쪽(REGISTER_MARGIN) 에서 낼 수 있는 가장 작은 r.

    평활(이동 최대 → 이동 평균) 은 성구를 이웃 프레임으로 번지게 한다. 그런데 성구 행은 **바닥**이 있다 (r 0.75 행은 ε 0.10 에서도
    413 Hz). 낮은 F0 프레임에 높은 r 이 번지면 늘어남이 표 끝에 붙어 수백 센트 어긋났다 (s040 의 232 Hz 아래 382 센트, §52.34).
    그래서 평활한 r 을 [필요한 최소 r, 그 F0 까지 내려갈 수 있는 최대 r] 에 가둔다. 둘이 어긋나면 필요한 쪽을 따른다.
    """
    reach, floor = [], []
    for c in candidates:
        tb = _row_table(strains, _interp_row(F, rows, c))
        reach.append(-np.inf if tb is None else tb[1][-1])
        floor.append(np.inf if tb is None else tb[1][0])
    reach, floor = np.array(reach), np.array(floor)
    need = np.zeros(len(f0_track))
    allow = np.ones(len(f0_track))
    for i, f in enumerate(f0_track):
        if f <= 0:
            continue
        low_ok = np.flatnonzero(floor <= f)
        allow[i] = candidates[low_ok[-1]] if len(low_ok) else 0.0
        if not (voiced is None or voiced[i]) or f <= REGISTER_MARGIN * chest_max:
            continue
        ok = np.flatnonzero(REGISTER_MARGIN * reach >= f)
        need[i] = candidates[ok[0]] if len(ok) else 1.0
    r = need.copy()
    k = max(1, int(round(REGISTER_SMOOTH_MS / frame_ms)))
    if k > 1 and r.any():
        from scipy.ndimage import maximum_filter1d, uniform_filter1d
        r = uniform_filter1d(maximum_filter1d(r, k, mode="nearest"), k, mode="nearest")
    r = np.maximum(np.minimum(r, allow), need)
    return np.clip(r, 0.0, 1.0)


def register_posture(f0, r, strains, rows, F):
    """성구 r 에서 F0 -> (ε, 진동 깊이, 휴지 반틈새 이동, 폐압 배율). 닿지 않으면 표 끝에 붙인다."""
    tb = _row_table(strains, _interp_row(F, rows, r))
    vd, dh0, ps = register_params(r)
    if tb is None:
        return float(strains[-1]), vd, dh0, ps
    es, fs_ = tb
    return float(np.interp(np.log(f0), np.log(fs_), es)), vd, dh0, ps


def render_source(f0_track, frame_ms, p_sub_cm=8.0, h0=-0.02e-3, fs_sim=200000.0,
                  seg_ms=2.5, n_modes=6, seed=0, fb_gain=0.0, voiced=None, posture=None, abduct=None,
                  p_ramp_ms=0.0, outlet=None, p_frames=None, h0_frames=None):
    """F0 궤적을 따라 자려 진동을 **끊김 없이** 적분한다. `self_oscillation.Glottis` 를 그대로 쓴다.

    늘어남은 **교정표(앞먹임) 로 2.5 ms 마다** 정한다. 실제 주기를 재서 되먹이는 경로(`fb_gain`) 는 남겨 두었지만
    기본은 끈다 — U 에서 직접 잰 추종이 되먹임 0.6·10 ms 22.2 센트, 되먹임 0·10 ms 13.6, 되먹임 0·2.5 ms 6.3 이었고
    되먹임 0.6·2.5 ms 는 루프가 불안정했다 (§52.18). 분석기 기준으로도 범위 안 22.5 → 12.8 센트 (§52.19).
    """
    if posture is None:
        es, fs_cal = calibrate(p_sub_cm, h0, n_modes=n_modes)
        print(f"  교정: F0 {np.round(fs_cal, 1)} Hz <- 늘어남 {np.round(es, 2)}", flush=True)
        lf_cal = np.log(fs_cal)
    else:
        p_str, p_ta, p_F = posture["chest"] if isinstance(posture, dict) else posture
        for k, t_ in enumerate(p_ta):
            print(f"  자세 교정 ta {t_:.2f}: F0 {np.round(p_F[k], 1)}", flush=True)
    n_frames = len(f0_track)
    total = int(n_frames * frame_ms / 1000.0 * fs_sim)
    seg = int(seg_ms / 1000.0 * fs_sim)
    dt = 1.0 / fs_sim
    P = p_sub_cm * CM_H2O
    U_out = np.zeros(total)
    A_out = np.zeros(total)                              # 성문 면적 [m²] — 난류 구동에 쓴다
    L_out = np.zeros(total)                              # 성대 길이 [m]
    g = None
    corr = 0.0
    errs = []
    last = None
    h_cur = h0
    r_frames = np.zeros(n_frames)
    r_str = r_rows = r_F = None
    if isinstance(posture, dict) and posture.get("register") is not None:
        r_str, r_rows, r_F = posture["register"]
        for k_, r_ in enumerate(r_rows):
            print(f"  성구 교정 r {r_:.2f}: F0 {np.round(r_F[k_], 1)}", flush=True)
        tb0 = monotone_table(p_str[np.isfinite(p_F[0])], p_F[0][np.isfinite(p_F[0])])
        r_frames = register_track(f0_track, voiced, r_str, r_rows, r_F, tb0[1][-1], frame_ms)
        vv = voiced if voiced is not None else np.ones(n_frames, bool)
        print(f"  성구: 유성 프레임의 {100 * np.mean(r_frames[vv] > 0):.1f} % 에서 r > 0 (최대 {r_frames.max():.2f})", flush=True)
    for s0 in range(0, total, seg):
        s1 = min(total, s0 + seg)
        fr = min(int((s0 + s1) / 2 / fs_sim * 1000.0 / frame_ms), n_frames - 1)
        # **늘어남은 구간 끝의 목표 F0 로 정한다.** 모형은 늘어남이 바뀌면 한 주기 안에 자리 잡으므로(§52.10)
        # 앞당겨 맞춰도 된다. 구간 가운데 값을 쓰면 F0 가 움직일 때 반 구간 + 검출 창만큼 뒤처졌다 —
        # 범위 안에서 F0 가 빠르게 변하는 프레임의 오차가 |차| 중앙 26 센트, 90 % 222 센트였다 (§52.12).
        # 구간 가운데의 목표 F0 로 정한다. 구간 끝 값으로 앞당기고(lc4) 되먹임을 구간 안에서 재게(lc5) 바꿨더니
        # 분석기 기준 범위 안 추종이 22.5 -> 43.9 센트로 나빠져 되돌렸다 (§52.14).
        f0 = float(f0_track[fr])
        # **발성 여부는 유성 마스크로** 판정한다. 적합된 제어열의 f0_target 은 무성 구간에서도 100 % 양수라,
        # f0 > 0 으로 판정하면 성대가 한 번도 쉬지 않는다 (§52.11).
        is_voiced = (voiced[fr] if voiced is not None else True) and f0 > 0
        if abduct is None and not is_voiced:
            g = None; corr = 0.0                         # 무성 — 성대를 없앤다 (옛 방식: 경계에서 유량이 순간에 끊긴다)
            continue
        if is_voiced or last is None:
            f0_use = f0 if f0 > 0 else 240.0
            if posture is None:
                eps_ff = float(np.interp(np.log(f0_use), lf_cal, es))
                eps = float(np.clip(eps_ff + corr, es[0], es[-1]))
                ta = 0.0
                vd, dh0, ps = 1.0, 0.0, 1.0
            else:
                r_now = float(r_frames[fr])
                if r_now > 0.0:
                    eps, vd, dh0, ps = register_posture(f0_use, r_now, r_str, r_rows, r_F)
                    ta = 0.0
                else:
                    eps, ta = posture_for(f0_use, p_str, p_ta, p_F)
                    vd, dh0, ps = 1.0, 0.0, 1.0
            if h0_frames is not None:
                dh0 = dh0 + float(h0_frames[fr])     # 세기 보상 (§52.71)
            last, changed = (eps, ta, vd, dh0, ps), True
        else:
            (eps, ta, vd, dh0, ps), changed = last, False             # 무성 — 후두 자세는 그대로 두고 벌리기만 한다
        if abduct is not None:
            # **무성은 성대를 멈추는 것이 아니라 벌리는 것이다** (§52.25). 반틈새를 목표값으로 1 차 지연을 두고 옮기고,
            # 말하는 사람이 미리 움직이듯 `lead_ms` 앞의 유성 여부를 본다.
            fr_l = min(fr + int(abduct["lead_ms"] / frame_ms), n_frames - 1)
            ahead = (voiced[fr_l] if voiced is not None else True) and f0_track[fr_l] > 0
            h_tgt = (h0 + dh0) if ahead else abduct["h0_open"]
        if g is None:
            h_cur = (h0 + dh0) if abduct is None else abduct["h0_open"]
            g = S.Glottis(strain=eps, h0=h_cur, n_modes=n_modes, seed=seed + s0, ta=ta, vib_depth=vd)
        elif changed:
            g.reshape(eps, ta=ta, vib_depth=vd)
            if abduct is None and dh0 != 0.0:
                g.set_h0(h0 + dh0)
        if outlet is not None:
            g.a_out = max(float(outlet[fr]), 1e-8)       # 성문 위 출구 면적 (구강 협착 + 비강 포트)
        # 폐압: 프레임 궤적이 있으면 그것 (음절 세기·무음의 0, §52.36), 없으면 상수.
        P_seg = P if p_frames is None else float(p_frames[fr]) * CM_H2O
        for it in range(s0, s1):
            if abduct is not None and it % 20 == 0:
                h_cur += (h_tgt - h_cur) * (1.0 - np.exp(-20.0 * dt / abduct["tau_s"]))
                g.set_h0(h_cur)
            # 폐압은 렌더 시작에서 `p_ramp_ms` 에 걸쳐 오른다. 순간에 켜면 모드가 걷어차여 무른 자세에서 0.1 s 넘게
            # 울렸다 (첫 무음의 170 Hz 웅웅거림, §52.29).
            u_, hmin = g.step(P_seg * ps if p_ramp_ms <= 0.0 else P_seg * ps * min(1.0, it * dt * 1000.0 / p_ramp_ms), dt)
            U_out[it] = u_
            A_out[it] = 2.0 * max(hmin, 0.0) * g.ell
            L_out[it] = g.ell
        f_act = _recent_f0(U_out, s1, fs_sim, span_s=0.03, min_cross=3)
        if f_act > 0 and is_voiced:
            err = np.log(f0 / f_act)
            errs.append(1200.0 * err / np.log(2.0))
        if f_act > 0 and posture is None and is_voiced:
            slope = np.interp(np.log(f0), lf_cal[1:], np.diff(es) / np.maximum(np.diff(lf_cal), 1e-9))
            corr = update_correction(corr, eps_ff, err, slope, es[0], es[-1], fb_gain)
    if errs:
        e = np.abs(np.array(errs[len(errs) // 5:]))       # 초기 수렴 구간은 뺀다
        print(f"  F0 추종 (구간 끝의 실측 대 목표): |차| 중앙 {np.median(e):.1f} 센트, 90 % {np.percentile(e, 90):.1f}",
              flush=True)
    return U_out, A_out, L_out


#: 연구개 포트가 다 열렸을 때의 면적 [m²] — 가정값 (문헌의 비음 연구개 포트 0.1~0.3 cm² 의 가운데).
VELUM_PORT_M2 = 0.2e-4

#: 난류 소음의 스트롤할 수 — 정점 주파수 f ≈ St·v/d (v 제트 속도, d 수력 직경). Stevens 1998 의 0.2.
STROUHAL = 0.2
#: 평면 제트의 폭 성장률 db/dx (반폭 성장률 ≈ 0.1 의 두 배). 잡음이 나는 곳까지 x 만큼 가면 폭 b = b0 + 0.2·x 이고,
#: 운동량 유속 ρv²b 가 보존되므로 중심 속도는 v0·√(b0/b) 로 준다 (§52.21).
JET_SPREAD = 0.2


def turbulent_flow_noise(U, A, Lf, fs_sim, fs_out, gain, seed=0, x_obs=0.0, order=1):
    """성문 난류의 **유량 잡음** — 크기는 `glottal_noise.turbulence` 의 구동(Re 문턱 포함), 모양은 **스트롤할**.

    처음에는 백색 잡음을 썼다. 성도가 이것을 한 번 미분하므로 고역이 옥타브당 +6 dB 로 치솟아, 복사 효율 1e-4 에서
    이미 4-8 kHz 가 녹음보다 16 dB 밝았고 그보다 세면 분석기가 F0 를 못 잡았다 (§52.12). 실제 난류 소음은 정점
    f_s = St·v/d 근처에서 가장 세고 그 위로 떨어진다. 여기서는 1 차 저역통과를 f_s(t) 에 둔다 — 성도의 미분과
    합쳐 f_s 아래는 옥타브당 +6 dB, 위는 평평해진다 (쌍극자 음원의 전형적 모양).

    `gain` 은 **복사 효율 상수 하나**이고 목표의 2~12 kHz 유성 HNR·기울기에 맞춰 교정하는 유일한 값이다.

    `x_obs` [m] 는 잡음이 실제로 나는 곳 — 제트가 가성대·후두개에 부딪히는 곳 — 까지의 거리다. 0 이면 성문 틈 자체의
    척도(폭 ≈ 0.4 mm, 정점 10 kHz 급) 를 쓴다. 틈 척도로는 잡음이 너무 밝아 2-5 kHz HNR 을 내리기 전에 4-8 kHz 가 6.8 dB
    넘쳤다 (§52.20). `order` 는 정점 위의 저역통과 차수 (2 면 성도의 미분과 합쳐 정점 위가 옥타브당 −6 dB).
    """
    from formant_ml.physics.glottal_noise import turbulence
    n = len(U)
    drive = np.zeros(n); fc = np.full(n, 1000.0)
    for i in range(n):
        if U[i] > 0.0 and A[i] > 0.0:
            drive[i] = turbulence(U[i], A[i], Lf[i])
            v = U[i] / A[i]                                   # 제트 속도 [m/s]
            d_h = 2.0 * A[i] / max(Lf[i], 1e-9)               # 수력 직경 [m]
            if x_obs > 0.0:
                b = d_h + JET_SPREAD * x_obs                   # 퍼진 제트의 폭
                fc[i] = float(np.clip(STROUHAL * v * np.sqrt(d_h / b) / b, 50.0, 12000.0))
            else:
                fc[i] = float(np.clip(STROUHAL * v / max(d_h, 1e-6), 300.0, 12000.0))
    k = max(1, int(fs_sim * 1e-3))
    env = np.sqrt(np.convolve(drive, np.ones(k) / k, "same"))
    fcs = np.convolve(fc, np.ones(k) / k, "same")
    env = resample_poly(env, 6, 25); fcs = resample_poly(fcs, 6, 25)
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, len(env))
    y = np.zeros(len(env)); st = np.zeros(max(1, int(order)))
    for i in range(len(env)):                                 # 시간에 따라 변하는 1 차 저역통과 (order 번 직렬)
        a_c = 1.0 - np.exp(-2.0 * np.pi * max(fcs[i], 50.0) / fs_out)
        v_in = w[i]
        for j in range(len(st)):
            st[j] += a_c * (v_in - st[j])
            v_in = st[j]
        y[i] = v_in
    # 저역통과가 줄인 분산을 대략 되돌린다 (같은 1 차 둘을 직렬로 두면 잡음 등가 대역폭이 절반)
    y *= np.sqrt(fs_out / (2.0 * np.pi * max(np.median(fcs), 50.0)) * (2.0 if int(order) >= 2 else 1.0))
    return gain * env * y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p-sub", type=float, default=8.0)
    ap.add_argument("--h0-mm", type=float, default=-0.02)
    ap.add_argument("--seconds", type=float, default=None, help="앞부분만 렌더 (시간 절약)")
    ap.add_argument("--asp-gain", type=float, default=0.0,
                    help="난류 유량 잡음의 복사 효율 상수 (교정값). 0 이면 잡음 없음")
    ap.add_argument("--abduct-mm", type=float, default=0.0,
                    help="무성 구간의 외전 반틈새 [mm]. 0 이면 옛 방식(성대를 없앰, 경계 클릭). 0.4 에서 격자 전체가 진동하지 않는다")
    ap.add_argument("--abduct-tau-ms", type=float, default=20.0, help="외전·내전 1 차 지연 [ms] (제어 상수)")
    ap.add_argument("--abduct-lead-ms", type=float, default=10.0, help="외전·내전을 미리 시작하는 시간 [ms]")
    ap.add_argument("--posture", action="store_true",
                    help="(ε, ta) 자세 교정으로 늘어남과 갑상피열근 수축을 같이 정한다 (§52.23). 저음(230 Hz 아래) 이 닿는다")
    ap.add_argument("--posture-cache", default=None, help="자세 교정표 캐시 .npz")
    ap.add_argument("--register", action="store_true", help="성구(가성) 축을 더한다 — 흉성으로 못 닿는 고음 (§52.33)")
    ap.add_argument("--register-cache", default=None, help="성구 교정표 캐시 .npz")
    ap.add_argument("--source-cache", default=None, help="음원(U, A, L) 캐시 .npz — 있으면 불러오고 없으면 만든다")
    ap.add_argument("--noise-x-mm", type=float, default=0.0,
                    help="난류 잡음이 나는 곳까지의 거리 [mm] (제트 퍼짐). 0 이면 성문 틈 척도 (§52.21)")
    ap.add_argument("--noise-order", type=int, default=1, help="난류 잡음 정점 위 저역통과 차수")
    ap.add_argument("--f-noise", type=float, default=0.0,
                    help="성대 장력 요동 (고유진동수 대비) — 위상을 흔드는 주기 변동의 원천")
    ap.add_argument("--f-noise-tau-ms", type=float, default=None,
                    help="장력 요동의 시상수 [ms]. 목표 화자의 주기 편차 이웃 상관(+0.37)에는 10 ms 가 맞다 (§52.15)")
    ap.add_argument("--constriction", action="store_true",
                    help="제어열의 구강 협착(a_c·oral_open)·연구개로 성문 위 출구 면적을 준다 (§52.30)")
    ap.add_argument("--p-ramp-ms", type=float, default=0.0, help="렌더 시작에서 폐압이 오르는 시간 [ms]")
    ap.add_argument("--k-contact-scale", type=float, default=1.0, help="접촉 강성 배율 (§52.44)")
    ap.add_argument("--c-contact-scale", type=float, default=1.0, help="접촉 감쇠 배율 (§52.44)")
    ap.add_argument("--jet-x-mm", type=float, default=0.0, help="제트 난류 요동의 발달 거리 [mm] (§52.45)")
    ap.add_argument("--jet-order", type=int, default=1, help="제트 난류 요동 저역통과 차수 (§52.46)")
    ap.add_argument("--jet-noise", type=float, default=0.0,
                    help="성문 제트 난류 압력 요동 C (rms / 동압, 물리 범위 0.1~0.3). §52.28")
    ap.add_argument("--p-noise", type=float, default=0.0,
                    help="성문 난류의 압력 요동 (성문하압 대비). 주기 간 변동의 물리적 원천")
    ap.add_argument("--seg-ms", type=float, default=2.5, help="늘어남을 다시 정하는 간격 [ms] (§52.18)")
    ap.add_argument("--fb-gain", type=float, default=0.0, help="F0 되먹임 이득 (0 이면 교정표만, §52.18)")
    a = ap.parse_args()
    z = np.load(a.stem + "_track.npz", allow_pickle=True)
    names = [str(x) for x in z["names"]]
    vals = np.asarray(z["values"], float)
    frame_ms = float(z["frame_ms"]) if "frame_ms" in z else 1.0
    if a.seconds:
        vals = vals[:int(a.seconds * 1000.0 / frame_ms)]
    f0 = vals[:, names.index("f0_target")]
    # 유성 마스크는 목표 녹음을 분석기로 재서 얻는다 (제어열의 f0_target 은 무성에서도 양수다).
    from formant_ml.engine.analyze import analyze
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.segment import fricative_mask
    tw, tsr = sf.read(a.stem + "_target.wav")
    tw = np.asarray(tw, float)
    prof = SpeakerProfile.load("profiles/yang_female.json")
    atr = analyze(tw, tsr, prof, int(round(tsr * frame_ms / 1000.0)), t0=0.0, full=tw)
    v_an = np.asarray(atr.voiced, bool)
    f_an = np.asarray(fricative_mask(tw, tsr, prof, int(round(tsr * frame_ms / 1000.0))), bool)
    k_an = min(len(v_an), len(f_an))                   # 두 마스크의 프레임 수가 다르다 (1420 대 1395)
    vm = v_an[:k_an] & ~f_an[:k_an]
    voiced = np.zeros(len(f0), bool)
    voiced[:min(len(vm), len(f0))] = vm[:len(f0)]
    print(f"  유성 비마찰 프레임 {100*voiced.mean():.0f} %", flush=True)
    S.P_NOISE = float(a.p_noise)
    S.JET_P_NOISE = float(a.jet_noise)
    S.K_CONTACT = S.K_CONTACT * float(a.k_contact_scale)
    S.C_CONTACT = S.C_CONTACT * float(a.c_contact_scale)
    S.JET_X = float(a.jet_x_mm) * 1e-3
    S.JET_ORDER = int(a.jet_order)
    S.F_NOISE = float(a.f_noise)
    if a.f_noise_tau_ms is not None:
        S.F_NOISE_TAU = float(a.f_noise_tau_ms) * 1e-3
    # 음원 캐시 — 200 kHz 적분은 잡음·성도 변형을 훑을 때마다 다시 할 필요가 없다. **음원을 정하는 인자를 같이 저장하고
    # 불러올 때 대조한다** (다른 설정의 캐시를 조용히 쓰지 않게).
    src_key = np.array([a.p_sub, a.h0_mm, a.seg_ms, a.fb_gain, a.f_noise, S.F_NOISE_TAU, a.p_noise,
                        a.seconds or 0.0, len(f0), float(a.posture), TA_KNEE_HZ, TA_LOW_HZ,
                        a.abduct_mm, a.abduct_tau_ms, a.abduct_lead_ms, a.jet_noise, a.p_ramp_ms,
                        float(a.constriction), float(a.register), a.k_contact_scale, a.c_contact_scale,
                        a.jet_x_mm, float(a.jet_order)], float)
    if a.source_cache and os.path.exists(a.source_cache):
        zc = np.load(a.source_cache)
        if not np.array_equal(zc["key"], src_key):
            raise SystemExit(f"음원 캐시 설정이 다르다: {zc['key']} 대 {src_key}")
        U, A, Lf = zc["U"], zc["A"], zc["Lf"]
        print(f"  음원 캐시 사용: {a.source_cache}", flush=True)
    else:
        outlet = None
        if a.constriction:
            # 출구 면적 = 구강 개방도 × 최협착 단면 + 연구개 × 비강 포트 최대 면적 (VELUM_PORT_M2, 가정값).
            ac = np.clip(vals[:, names.index("a_c")], 0.0, None) * 1e-4
            oo = np.clip(vals[:, names.index("oral_open")], 0.0, 1.0)
            ve = np.clip(vals[:, names.index("velum")], 0.0, 1.0) if "velum" in names else np.zeros(len(ac))
            outlet = oo * ac + ve * VELUM_PORT_M2
        posture = (calibrate_posture(a.p_sub, a.h0_mm * 1e-3, cache=a.posture_cache) if a.posture else None)
        if a.posture and a.register:
            posture = {"chest": posture,
                       "register": calibrate_register(a.p_sub, a.h0_mm * 1e-3, cache=a.register_cache)}
        U, A, Lf = render_source(f0, frame_ms, p_sub_cm=a.p_sub, h0=a.h0_mm * 1e-3, voiced=voiced,
                                seg_ms=a.seg_ms, fb_gain=a.fb_gain, posture=posture,
                                abduct=(dict(h0_open=a.abduct_mm * 1e-3, tau_s=a.abduct_tau_ms * 1e-3,
                                             lead_ms=a.abduct_lead_ms) if a.abduct_mm > 0 else None),
                                p_ramp_ms=a.p_ramp_ms, outlet=outlet)
        if a.source_cache:
            np.savez_compressed(a.source_cache, U=U, A=A, Lf=Lf, key=src_key)
    du_sim = np.gradient(U, 1.0 / 200000.0)
    du = resample_poly(du_sim, 6, 25)                          # 200 kHz -> 48 kHz
    fs = 48000.0
    hop = int(round(fs * frame_ms / 1000.0))
    n = vals.shape[0] * hop
    du = np.pad(du, (0, max(0, n - len(du))))[:n]
    scale = max(np.abs(du).max(), 1e-12)
    du = du / scale
    # 난류 유량 잡음. 배음 음원과 **같은 눈금**으로 맞추려면 유량(U)을 미분한 du 와 같은 배율로 나눈다:
    # 성도가 asp 를 한 번 미분하므로 asp 는 유량 차원이고, du 는 이미 미분된 유량이다.
    asp = turbulent_flow_noise(U, A, Lf, 200000.0, fs, a.asp_gain, seed=11,
                               x_obs=a.noise_x_mm * 1e-3, order=a.noise_order)
    asp = np.pad(asp, (0, max(0, n - len(asp))))[:n] * (fs / scale)
    tr = VocalTract(fs, hop).double()
    ten = torch.tensor(vals, dtype=torch.float64).unsqueeze(0)
    c = {nm: ten[..., names.index(nm)] for nm in PARAM_NAMES if nm in names}
    x = torch.tensor(du, dtype=torch.float64).unsqueeze(0)
    z0 = torch.zeros_like(x)
    xa = torch.tensor(asp, dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        y = tr(x, z0, xa, z0, c)["audio"][0].numpy()
    y = y / max(np.abs(y).max(), 1e-12) * 0.5
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out + "_fit.wav", y, int(fs))
    t, sr = sf.read(a.stem + "_target.wav")
    sf.write(a.out + "_target.wav", t[:len(y)], sr)
    print(f"결과: {a.out}_fit.wav  ({len(y)/fs:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
