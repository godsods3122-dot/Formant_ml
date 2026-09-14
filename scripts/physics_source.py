"""물리 성문 음원을 **분석기 트랙에서 처음부터** 만든다 — 적합기의 외부 음원 (`copyfit --external-source`, MEASUREMENTS §52.31).

    python scripts/physics_source.py WAV --from T0 --to T1 --out out/PHY/src/NAME.npz

M 적합의 제어열은 쓰지 않는다 (메커니즘이 바뀌면 처음부터 돌린다). F0·유성 마스크·구강 협착은 분석기가 목표 녹음에서 잰 값이다.
구간 자르기와 잡음 제거는 `copyfit.py` 와 같게 한다 — 두 스크립트가 **같은 프레임 격자**를 봐야 음원이 제자리에 붙는다.

음원 수준은 같은 분석기 제어열로 LF 엔진이 낼 성문 배음의 유성 구간 rms 에 맞춘다 (적합기의 초기 이득이 LF 판과 같은 눈금에서 출발하게).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from formant_ml.engine.analyze import analyze                              # noqa: E402
from formant_ml.engine.control import INDEX, PARAM_NAMES                   # noqa: E402
from formant_ml.engine.denoise import denoise, noise_profile               # noqa: E402
from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile      # noqa: E402
from formant_ml.engine.segment import fricative_mask                       # noqa: E402
from formant_ml.engine.voice import EngineConfig, VoiceEngine              # noqa: E402
from formant_ml.physics import self_oscillation as S                       # noqa: E402

import physics_render_lc as R                                              # noqa: E402

FS_SIM = 200000.0
FS_OUT = 48000

#: **폐압 궤적** (§52.36). 음원 dU/dt 의 rms 는 폐압의 γ 제곱으로 자란다 — 흉성·두꺼운 저음·가성에서 잰 γ 2.24 / 2.44 / 2.44
#: (폐압 두 배에 약 14 dB). 목표의 유성 수준 윤곽을 이것으로 뒤집어 폐압을 정한다. 성도의 모음별 이득이 섞이는 몫은 적합기의
#: voice_gain·tract_gain 이 고친다.
LUNG_GAMMA = 2.3
#: 폐압 범위 [cmH2O] — 아래는 흉성이 확실히 서는 값(4~5 는 약하게 섬), 위는 외치는 세기.
LUNG_P_MIN = 5.5
LUNG_P_MAX = 16.0
#: 호흡근이 폐압을 바꾸는 시상수 [ms] (앞뒤로 걸어 대칭).
LUNG_TAU_MS = 30.0
#: 말 구간 = 목표의 **유성 중앙 수준**에서 이만큼 [dB] 안. 최대 −40 dB 로 잡았더니 s101 의 99 % 가 말로 잡혀 무음의 폐압이
#: 내려가지 않았다. 실측: s101 앞 40~150 ms(유성 중앙 −7.7 dB, 실제 소리) 는 말, 끝 1330~1419 ms(−46.7) 와 s040 앞(−54) 은 무음 (§52.37).
SPEECH_FLOOR_DB = 25.0


#: **보상: 세기 ↔ 내전** (§52.71·§52.72). 세게 말하면 성대를 더 붙여 기식이 준다 — H1−H2 가 수준을 따라 내려간다.
#:
#: 코퍼스 평균(−0.25 dB/dB) 을 모든 파일에 강제하면 안 된다: 파일별 분포가 [−0.71, +0.27] (10~90 %) 이고 21 % 는
#: **양수**다. 게다가 물리 음원은 폐압 궤적만으로 이미 −0.19~−0.65 를 내므로(닫힘이 날카로워진다) 평균을 더 얹으면
#: 두 번 세게 된다 — 실제로 그렇게 해서 목표에서 더 멀어졌다 (§52.72).
#:
#: 그래서 **닫힌 고리**로 둔다: 목표 녹음에서 기울기를 재고, 렌더한 음원에서 재고, 그 차이만큼 내전 이득을 고친다.
#: 이득의 물리 눈금은 내전 스윕(§52.41) 의 h0 0.04 mm -> H1−H2 7.4 dB, 즉 185 dB/mm 다.
ADDUCT_DB_PER_MM = 185.0
#: 보상이 움직일 수 있는 폭 [mm] — 내전은 좁은 범위 밖에서 진동이 불안정해진다 (§52.41).
ADDUCT_SPAN_MM = 0.02
#: 이득의 상한 [mm/dB] — 폭 안에서도 너무 가파르면 프레임마다 자세가 튄다.
ADDUCT_GAIN_LIMIT = 0.006


def h1h2_level_slope(x, sr, f0, good, nfft=2048, hop=480):
    """유성 창에서 (수준 dB, H1−H2 dB) 의 회귀 기울기 [dB/dB]. 창이 모자라면 nan."""
    from scipy.signal import stft
    f, t_, Z = stft(x, sr, nperseg=nfft, noverlap=nfft - hop)
    A = np.abs(Z)
    L, H = [], []
    for j in range(A.shape[1]):
        k = min(int(t_[j] * 1000.0), len(good) - 1)
        if not good[k] or f0[k] <= 0:
            continue
        F = f0[k]
        pk = lambda h: 20 * np.log10(A[(f > h * F - 0.3 * F) & (f < h * F + 0.3 * F), j].max() + 1e-12)  # noqa: E731
        L.append(20 * np.log10(np.sqrt((A[(f >= 80) & (f < 8000), j] ** 2).mean()) + 1e-12))
        H.append(pk(1) - pk(2))
    if len(L) < 20:
        return float("nan")
    L = np.asarray(L) - np.median(L)
    H = np.asarray(H) - np.median(H)
    if L.std() < 1e-6:
        return float("nan")
    return float(np.polyfit(L, H, 1)[0])


def adduction_track(level_db, voiced, frame_ms, gain_mm_per_db, span=ADDUCT_SPAN_MM, smooth_ms=31.0):
    """목표 수준 [dB] -> 휴지 반틈새 보상 [m]. 유성 구간 중앙 수준 대비 편차 × `gain_mm_per_db`."""
    from scipy.ndimage import uniform_filter1d
    v = np.asarray(voiced, bool) & np.isfinite(level_db)
    if not v.any() or not np.isfinite(gain_mm_per_db) or gain_mm_per_db == 0.0:
        return np.zeros(len(level_db))
    ref = float(np.median(level_db[v]))
    w = max(1, int(round(smooth_ms / frame_ms)))
    lv = uniform_filter1d(np.where(v, level_db, ref), w, mode="nearest")
    d = np.clip((lv - ref) * float(gain_mm_per_db), -span, span)
    return np.where(v, d, 0.0) * 1e-3


def frame_level_db(seg, sr, n_fr, frame_ms, lo=80.0, hi=8000.0, win_ms=20.0):
    """프레임 중심의 `win_ms` 창 에너지 [dB] (80 Hz~8 kHz 대역)."""
    from scipy.signal import butter, sosfiltfilt
    x = sosfiltfilt(butter(4, [lo, min(hi, 0.45 * sr)], "bandpass", fs=sr, output="sos"), seg)
    hop = frame_ms * sr / 1000.0
    w = max(1, int(win_ms * sr / 1000.0))
    out = np.full(n_fr, -200.0)
    for i in range(n_fr):
        c = int((i + 0.5) * hop)
        a, b = max(0, c - w // 2), min(len(x), c + w // 2)
        if b > a:
            out[i] = 10.0 * np.log10(np.mean(x[a:b] ** 2) + 1e-20)
    return out


def lung_pressure_track(level_db, voiced, p_ref, frame_ms, gamma=LUNG_GAMMA, floor_db=SPEECH_FLOOR_DB):
    """목표 수준 [dB] -> 프레임 폐압 [cmH2O]. 무음은 0, 말 안의 무성은 이웃 유성에서 잇는다, 앞뒤 1 차 지연."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from scipy.signal import filtfilt
    n = len(level_db)
    voiced = np.asarray(voiced, bool)
    k = max(1, int(round(30.0 / frame_ms)))
    ref = float(np.median(level_db[voiced])) if voiced.any() else float(level_db.max())
    speech = maximum_filter1d((level_db > ref - floor_db).astype(np.uint8), 2 * k + 1) > 0
    lv = np.where(voiced, level_db, np.nan)
    w = max(1, int(round(31.0 / frame_ms)))
    num = uniform_filter1d(np.nan_to_num(lv), w, mode="nearest")
    den = uniform_filter1d(np.isfinite(lv).astype(float), w, mode="nearest")
    sm = np.where(voiced & (den > 0.2), num / np.maximum(den, 1e-9), np.nan)
    idx = np.flatnonzero(np.isfinite(sm))
    if not len(idx):
        return np.zeros(n)
    filled = np.interp(np.arange(n), idx, sm[idx])
    l_ref = float(np.median(sm[idx]))
    p = np.clip(p_ref * 10.0 ** ((filled - l_ref) / (20.0 * gamma)), LUNG_P_MIN, LUNG_P_MAX)
    p = np.where(speech, p, 0.0)
    a = 1.0 - np.exp(-frame_ms / LUNG_TAU_MS)
    p = filtfilt([a], [1.0, -(1.0 - a)], p, padtype="constant")
    return np.clip(p, 0.0, LUNG_P_MAX)


def frame_f0_from_flow(U, n_fr, frame_ms, fs=FS_SIM):
    """유량의 닫힘→열림 교차(전역 99 % 분위의 5 % 문턱, 선형 보간) 로 잰 주기를 프레임에 배정한다. 못 재면 nan.

    닫혀서 U = 0 인 구간이 있어야 교차가 주기당 한 번이다 — 벌린 성문의 직류 유량(무성) 은 교차가 없어 nan 이 된다.
    """
    pos = U[U > 0]
    out = np.full(n_fr, np.nan)
    if not len(pos):
        return out
    thr = 0.05 * np.percentile(pos, 99)
    i = np.flatnonzero((U[1:] > thr) & (U[:-1] <= thr))
    if len(i) < 3:
        return out
    tc = (i + (thr - U[i]) / np.maximum(U[i + 1] - U[i], 1e-30)) / fs
    T = np.diff(tc)
    t_fr = (np.arange(n_fr) + 0.5) * frame_ms / 1000.0
    k = np.searchsorted(tc, t_fr, side="right") - 1
    ok = (k >= 0) & (k < len(T))
    kk = np.clip(k, 0, len(T) - 1)
    ok &= (T[kk] > 1.0 / 600.0) & (T[kk] < 1.0 / 80.0)
    out[ok] = 1.0 / T[kk[ok]]
    return out


def refine_command(f_cmd, f_target, f_meas, voiced, gain=0.8, width=9, limit_cents=300.0):
    """반복 학습 보정 한 걸음 — 로그 F0 비의 이동 중앙값(유효 프레임만) 을 명령에 `gain` 만큼 곱한다.

    지터(주기마다의 무작위) 는 중앙값 창이 거르고, 자세표 보간 오차·과도 지연 같은 느린 오차만 고친다.
    명령은 목표에서 `limit_cents` 안에 가둔다.
    """
    r = np.where(voiced & np.isfinite(f_meas) & (f_target > 0), np.log(np.maximum(f_target, 1e-9) / np.maximum(f_meas, 1e-9)), np.nan)
    h = width // 2
    rs = np.full(len(r), np.nan)
    for j in np.flatnonzero(np.isfinite(r)):
        rs[j] = np.nanmedian(r[max(0, j - h):j + h + 1])
    new = f_cmd.copy()
    upd = np.isfinite(rs)
    new[upd] = f_cmd[upd] * np.exp(gain * rs[upd])
    lim = limit_cents / 1200.0 * np.log(2.0)
    tgt = np.maximum(f_target, 1e-9)
    new = np.where(f_target > 0, np.exp(np.clip(np.log(np.maximum(new, 1e-9)), np.log(tgt) - lim, np.log(tgt) + lim)), new)
    return new


def tracking_report(f_meas, f_target, voiced):
    ok = voiced & np.isfinite(f_meas) & (f_target > 0)
    if not ok.any():
        return "측정 없음"
    c = np.abs(1200.0 * np.log2(f_meas[ok] / f_target[ok]))
    lo = ok & (f_target < 232.0)
    cl = np.abs(1200.0 * np.log2(f_meas[lo] / f_target[lo])) if lo.any() else np.array([np.nan])
    return (f"|차| 중앙 {np.median(c):.1f} 센트, 90 % {np.percentile(c, 90):.1f}, >50 센트 {100 * np.mean(c > 50):.1f} %, "
            f"232 Hz 아래 중앙 {np.nanmedian(cl):.1f} (측정 {int(ok.sum())}/{int(voiced.sum())} 프레임)")


#: 사람 F0 가 바뀔 수 있는 속도의 상한 [센트/ms]. 실측 문헌의 최대 변화율(대략 0.1 반음/ms) 에 여유를 둔 값.
#: 이보다 빨리 갔다 돌아오는 요동은 분석기의 유성 경계 오류로 본다 (§52.39).
F0_MAX_RATE_CENTS_PER_MS = 30.0


def sanitize_f0(f0, voiced, frame_ms, max_rate=F0_MAX_RATE_CENTS_PER_MS, width_ms=15.0, travel_ms=3.0):
    """분석기 F0 에서 **생리적으로 불가능한 짧은 요동**만 걷어낸다.

    유성 프레임마다 `width_ms` 창의 로그 F0 중앙값을 기준으로 삼고, 기준에서 `travel_ms` 동안 갈 수 있는 거리(max_rate × travel_ms,
    기본 90 센트) 보다 멀리 벗어난 프레임을 기준값으로 바꾼다. 허용 거리를 창 반폭에 비례시키면(15 ms 창에서 240 센트) V 자 요동의 얕은
    쪽(−200 센트) 이 남았다. 선형 글라이드는 창 중앙값 위에 정확히 놓이므로 빠르더라도 남는다.
    반환 (고친 F0, 고친 프레임 수).
    """
    f = np.asarray(f0, float).copy()
    v = np.asarray(voiced, bool) & (f > 0)
    h = max(1, int(round(width_ms / frame_ms / 2)))
    lim = max_rate * travel_ms
    lc = np.where(v, 1200.0 * np.log2(np.maximum(f, 1e-9)), np.nan)
    fixed = 0
    out = f.copy()
    for i in np.flatnonzero(v):
        seg = lc[max(0, i - h):i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 3:
            continue
        med = float(np.median(seg))
        if abs(lc[i] - med) > lim:
            out[i] = 2.0 ** (med / 1200.0)
            fixed += 1
    return out, fixed


def open_curve(area_sim, voiced, n_out, spf, fs_sim=FS_SIM, fs_out=FS_OUT, q=99.0):
    """물리 성문 면적 [m²] (시뮬레이션 표본률) -> 48 kHz 의 0~1 개방 곡선과 면적.

    엔진의 `OPEN_DAMP` 가 성문이 열린 동안 F1 을 감쇠·이동시키는 곡선이다. LF 위상에서 만들면 외부 물리 음원과 시각이 어긋나므로
    물리 면적에서 만든다 (§52.38). 기준은 유성 구간 면적의 `q` 분위 — 기식·외전 구간의 넓은 면적은 1 에서 잘린다.
    """
    idx = (np.arange(n_out) * (fs_sim / fs_out)).astype(int)
    idx = np.clip(idx, 0, len(area_sim) - 1)
    a48 = np.asarray(area_sim, float)[idx]
    vm = np.repeat(np.asarray(voiced, bool), spf)[:n_out]
    pos = a48[vm & (a48 > 0)]
    ref = float(np.percentile(pos, q)) if len(pos) else 1.0
    return np.clip(a48 / max(ref, 1e-12), 0.0, 1.0), a48


def closure_times(U, fs=FS_SIM):
    """물리 유량의 열림→닫힘 교차 시각 [s] (전역 99 % 분위의 5 % 문턱, 선형 보간)."""
    pos = U[U > 0]
    if not len(pos):
        return np.zeros(0)
    thr = 0.05 * np.percentile(pos, 99)
    i = np.flatnonzero((U[1:] <= thr) & (U[:-1] > thr))
    return (i + (U[i] - thr) / np.maximum(U[i] - U[i + 1], 1e-30)) / fs


def pulse_phase_errors(t_phys, t_tgt, f_lo=120.0, f_hi=600.0):
    """목표 펄스마다 가장 가까운 물리 닫힘까지의 시각 차를 목표 주기로 나눈 몫 (−0.5~0.5 로 감쌈).

    반환 (목표 펄스 시각, 몫, 주기, 유성 조각 번호). 조각은 이웃 목표 펄스 간격이 주기의 1.6 배를 넘는 곳에서 끊는다.
    """
    t_phys = np.asarray(t_phys, float)
    t_tgt = np.asarray(t_tgt, float)
    tj, ej, Tj, sj = [], [], [], []
    seg = 0
    for j in range(1, len(t_tgt) - 1):
        T = 0.5 * (t_tgt[j + 1] - t_tgt[j - 1])
        if (t_tgt[j] - t_tgt[j - 1]) > 1.6 * max(T, 1e-9):
            seg += 1
        if not (1.0 / f_hi < T < 1.0 / f_lo) or not len(t_phys):
            continue
        k = int(np.clip(np.searchsorted(t_phys, t_tgt[j]), 1, len(t_phys) - 1))
        k = k - 1 if abs(t_phys[k - 1] - t_tgt[j]) < abs(t_phys[k] - t_tgt[j]) else k
        e = (t_phys[k] - t_tgt[j]) / T
        tj.append(t_tgt[j]); ej.append((e + 0.5) % 1.0 - 0.5); Tj.append(T); sj.append(seg)
    return np.array(tj), np.array(ej), np.array(Tj), np.array(sj, int)


def remove_segment_offsets(e, seg):
    """유성 조각마다 원형 평균 어긋남을 빼 (−0.5~0.5) 로 감싼다. 반환 (남은 몫, 조각별 집중도 R 의 평균)."""
    d = np.zeros_like(e)
    rs = []
    for sid in np.unique(seg):
        m = seg == sid
        z = np.exp(2j * np.pi * e[m])
        mean = z.mean()
        rs.append(abs(mean))
        d[m] = np.angle(z * np.conj(mean) / max(abs(mean), 1e-12)) / (2.0 * np.pi)
    return d, (float(np.mean(rs)) if rs else float("nan"))


def phase_lock_command(f_cmd, frame_ms, tj, d, Tj, gain=0.5, cycles=4.0, limit=0.03):
    """위상 고정 한 걸음 — 물리 닫힘이 늦으면(d > 0) 그 앞 `cycles` 주기 동안 주파수를 올려 따라잡게 한다.

    펄스마다 Δf/f = gain·d/cycles (±limit) 를 목표 펄스보다 cycles·T/2 앞에 중심을 둔 **폭 cycles·T 의 삼각 창**으로 펴고, 창이 겹치는 곳은
    가중 평균한다. 처음에는 프레임으로 선형 보간했는데, 펄스가 하나면 그 순간에만 값이 서고 여러 개면 무성 틈을 건너 보정을 이어 버렸다.
    """
    n = len(f_cmd)
    t_fr = (np.arange(n) + 0.5) * frame_ms / 1000.0
    num = np.zeros(n)
    den = np.zeros(n)
    for t, dd, T in zip(tj, d, Tj):
        r = float(np.clip(gain * dd / cycles, -limit, limit))
        c = t - 0.5 * cycles * T
        half = 0.5 * cycles * T
        a = int(max(0, np.floor((c - half) * 1000.0 / frame_ms)))
        b = int(min(n, np.ceil((c + half) * 1000.0 / frame_ms) + 1))
        if b <= a:
            continue
        w = np.clip(1.0 - np.abs(t_fr[a:b] - c) / max(half, 1e-9), 0.0, 1.0)
        num[a:b] += w * r
        den[a:b] += w
    corr = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return np.where(f_cmd > 0, f_cmd * (1.0 + corr), f_cmd)


def alignment_report(t_phys, t_tgt):
    tj, e, Tj, seg = pulse_phase_errors(t_phys, t_tgt)
    if not len(e):
        return "짝 없음", None
    d, R = remove_segment_offsets(e, seg)
    return (f"짝 {len(e)}, 조각별 집중도 R {R:.2f}, 상수 뺀 |차| 중앙 {np.median(np.abs(d)):.3f} 주기, "
            f"0.25 주기 넘는 몫 {100 * np.mean(np.abs(d) > 0.25):.0f} %"), (tj, d, Tj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--frame-ms", type=float, default=1.0)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--p-sub", type=float, default=8.0)
    ap.add_argument("--h0-mm", type=float, default=-0.02)
    ap.add_argument("--abduct-mm", type=float, default=0.6)
    ap.add_argument("--abduct-tau-ms", type=float, default=8.0)
    ap.add_argument("--abduct-lead-ms", type=float, default=15.0)
    ap.add_argument("--p-ramp-ms", type=float, default=60.0)
    ap.add_argument("--jet-noise", type=float, default=0.2)
    ap.add_argument("--no-constriction", action="store_true")
    ap.add_argument("--posture-cache", default=None)
    ap.add_argument("--k-contact-scale", type=float, default=1.0,
                    help="접촉 강성 배율 (조직 압축). H1−H2·저역을 올린다 (§52.42). 교정표를 다시 만든다")
    ap.add_argument("--c-contact-scale", type=float, default=1.0,
                    help="접촉 감쇠 배율. 8 kHz 위를 줄인다 (§52.42). 교정표를 다시 만든다")
    ap.add_argument("--adduct-comp", action="store_true",
                    help="세기 ↔ 내전 보상을 켠다 (§52.74). **기본 꺼짐** — 기울기는 반쯤 당기지만 저음에서 자세가 무너진다")
    ap.add_argument("--tract-compliance", type=float, default=0.0,
                    help="성문 위 공간 순응도 [m³/Pa] (§52.49). 0 이면 준정상 두 오리피스. 녹음의 voice bar 에 맞는 후보 1e-8")
    ap.add_argument("--phase-lock", type=int, default=0,
                    help="위상 고정 반복 횟수 — 물리 닫힘 시각을 목표 펄스(Praat cc) 에 조각별 상수 어긋남까지 맞춘다 (§52.56)")
    ap.add_argument("--workers", type=int, default=3,
                    help="자세·성구 교정의 병렬 프로세스 수 (적합이 돌 때 동시 작업 6 개 한도를 넘지 않게)")
    ap.add_argument("--no-register", action="store_true", help="성구(가성) 축을 끈다")
    ap.add_argument("--no-lung-track", action="store_true", help="폐압을 상수(--p-sub) 로 둔다 (§52.36 이전)")
    ap.add_argument("--no-sanitize-f0", action="store_true", help="분석기 F0 의 불가능한 요동을 거르지 않는다 (§52.39 이전)")
    ap.add_argument("--register-cache", default=None)
    ap.add_argument("--refine", type=int, default=2,
                    help="F0 반복 학습 보정 횟수 — 외부 음원을 걸면 적합기가 F0 를 못 고치므로 음원 쪽에서 맞춘다")
    a = ap.parse_args()
    spf = a.frame_ms * FS_OUT / 1000.0
    if abs(spf - round(spf)) > 1e-9:
        raise SystemExit("--frame-ms 는 48 kHz 에서 정수 샘플이어야 한다")
    spf = int(round(spf))

    y, sr = sf.read(a.wav)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, dtype=np.float64)
    if not a.no_denoise:
        y = denoise(y, sr, noise_profile(y, sr))
    t1 = a.t1 if a.t1 is not None else len(y) / sr
    seg = y[int(a.t0 * sr):int(t1 * sr)]
    prof = SpeakerProfile.load(a.profile) if a.profile else DEFAULT_PROFILE
    hop = max(1, int(round(a.frame_ms * sr / 1000.0)))
    track = analyze(seg, sr, prof, hop, t0=a.t0, full=y)
    vals = np.asarray(track.values, dtype=np.float64)
    n_fr = track.n_frames
    f0 = vals[:, INDEX["f0_target"]]
    voi = np.asarray(track.voiced, bool)
    fr = np.asarray(getattr(track, "fricative", np.zeros(0, bool)), bool)
    if len(fr) == 0:
        fr = np.asarray(fricative_mask(seg, sr, prof, hop), bool)
    k = min(len(voi), len(fr), n_fr)
    voiced = np.zeros(n_fr, bool)
    voiced[:k] = voi[:k] & ~fr[:k]
    print(f"구간 {a.t0:.3f}~{t1:.3f} s, {n_fr} 프레임, 유성 비마찰 {100 * voiced.mean():.0f} %", flush=True)
    f0_raw = f0.copy()
    if not a.no_sanitize_f0:
        f0, n_fix = sanitize_f0(f0, voiced, a.frame_ms)
        print(f"  분석기 F0 거르기: {n_fix} 프레임 (변화율 {F0_MAX_RATE_CENTS_PER_MS:g} 센트/ms 초과)", flush=True)

    S.K_CONTACT = S.K_CONTACT * float(a.k_contact_scale)
    S.C_CONTACT = S.C_CONTACT * float(a.c_contact_scale)
    S.JET_P_NOISE = float(a.jet_noise)
    # 순응도는 출구 면적이 있을 때만 작동하고, 교정표(출구 없음) 에는 영향이 없어 캐시 키에 넣지 않는다.
    S.TRACT_COMPLIANCE = float(a.tract_compliance)
    outlet = None
    if not a.no_constriction:
        outlet = (np.clip(vals[:, INDEX["oral_open"]], 0.0, 1.0) * np.clip(vals[:, INDEX["a_c"]], 0.0, None) * 1e-4
                  + np.clip(vals[:, INDEX["velum"]], 0.0, 1.0) * R.VELUM_PORT_M2)
    posture = R.calibrate_posture(a.p_sub, a.h0_mm * 1e-3, cache=a.posture_cache, workers=a.workers)
    if not a.no_register:
        posture = {"chest": posture,
                   "register": R.calibrate_register(a.p_sub, a.h0_mm * 1e-3, cache=a.register_cache, workers=a.workers)}
    abduct = (dict(h0_open=a.abduct_mm * 1e-3, tau_s=a.abduct_tau_ms * 1e-3, lead_ms=a.abduct_lead_ms)
              if a.abduct_mm > 0 else None)
    p_frames = None
    h0_frames = None
    lvl = frame_level_db(seg, sr, n_fr, a.frame_ms)
    if not a.no_lung_track:
        p_frames = lung_pressure_track(lvl, voiced, a.p_sub, a.frame_ms)
    adduct_gain = 0.0
    slope_tgt = float("nan")
    if a.adduct_comp:
        slope_tgt = h1h2_level_slope(seg, sr, f0, voiced)
        print(f"  보상 목표: 녹음의 H1−H2 대 수준 {slope_tgt:+.3f} dB/dB", flush=True)
        sp = p_frames > 0.5
        print(f"  폐압 궤적: 말 {100 * sp.mean():.0f} % 프레임, 유성 분위 5/50/95 % "
              f"{np.round(np.percentile(p_frames[voiced], [5, 50, 95]), 1)} cmH2O", flush=True)
    f_cmd = f0.copy()
    for it in range(a.refine + 1):
        U, A_sim, _L = R.render_source(f_cmd, a.frame_ms, p_sub_cm=a.p_sub, h0=a.h0_mm * 1e-3, voiced=voiced,
                                    posture=posture, abduct=abduct, p_ramp_ms=a.p_ramp_ms, outlet=outlet,
                                    p_frames=p_frames, h0_frames=h0_frames)
        f_meas = frame_f0_from_flow(U, n_fr, a.frame_ms)
        print(f"  추종 (반복 {it}): {tracking_report(f_meas, f0, voiced)}", flush=True)
        # **보상은 첫 반복에서 한 번만 잡는다.** h0 가 바뀌면 F0 도 바뀌므로, 뒤 반복의 F0 보정이 그것을 흡수해야 한다.
        # 매 반복 갱신하면 마지막 갱신 뒤에 보정이 없어 추종이 9.6 -> 19.7 센트로 나빠졌다 (§52.73).
        if it == 0 and np.isfinite(slope_tgt):
            du_now = resample_poly(np.gradient(U, 1.0 / FS_SIM), 6, 25)
            slope_src = h1h2_level_slope(du_now, FS_OUT, f0, voiced)
            if np.isfinite(slope_src):
                # 음원이 목표보다 더 가파르면(더 음수) 세질 때 덜 붙게 이득을 올린다.
                raw = adduct_gain + (slope_tgt - slope_src) / ADDUCT_DB_PER_MM
                adduct_gain = float(np.clip(raw, -ADDUCT_GAIN_LIMIT, ADDUCT_GAIN_LIMIT))
                if abs(raw) > ADDUCT_GAIN_LIMIT:
                    print(f"    보상 이득이 상한에 걸렸다 ({raw:+.4f} -> {adduct_gain:+.4f} mm/dB) — 기울기를 다 못 맞춘다", flush=True)
                h0_frames = adduction_track(lvl, voiced, a.frame_ms, adduct_gain)
                print(f"    보상: 음원 기울기 {slope_src:+.3f} -> 이득 {adduct_gain:+.4f} mm/dB, "
                      f"h0 편차 5/95 % {np.round(np.percentile(h0_frames[voiced] * 1e3, [5, 95]), 4)} mm", flush=True)
        if it < a.refine:
            f_cmd = refine_command(f_cmd, f0, f_meas, voiced)
    t_tgt = np.asarray(getattr(track, "pulses", np.zeros(0)), float)
    if len(t_tgt):
        rep, _ = alignment_report(closure_times(U), t_tgt)
        print(f"  펄스 정렬 (위상 고정 전): {rep}", flush=True)
    for it in range(a.phase_lock if len(t_tgt) else 0):
        _rep, got = alignment_report(closure_times(U), t_tgt)
        if got is None:
            break
        tj, d, Tj = got
        f_cmd = phase_lock_command(f_cmd, a.frame_ms, tj, d, Tj)
        U, A_sim, _L = R.render_source(f_cmd, a.frame_ms, p_sub_cm=a.p_sub, h0=a.h0_mm * 1e-3, voiced=voiced,
                                       posture=posture, abduct=abduct, p_ramp_ms=a.p_ramp_ms, outlet=outlet,
                                       p_frames=p_frames, h0_frames=h0_frames)
        f_meas = frame_f0_from_flow(U, n_fr, a.frame_ms)
        rep, _ = alignment_report(closure_times(U), t_tgt)
        print(f"  위상 고정 {it + 1}: {rep} | 추종 {tracking_report(f_meas, f0, voiced)}", flush=True)
    du = resample_poly(np.gradient(U, 1.0 / FS_SIM), 6, 25)
    g_open, area48 = open_curve(A_sim, voiced, n_fr * spf, spf)
    n = n_fr * spf
    du = np.pad(du, (0, max(0, n - len(du))))[:n]

    # 수준: 같은 제어열로 LF 엔진이 내는 성문 배음의 유성 rms 에 맞춘다.
    eng = VoiceEngine(EngineConfig(sample_rate=FS_OUT, frame_ms=a.frame_ms, residual=False,
                                   speaker="female" if prof.f0_nominal > 165 else "male"), prof)
    ctrl = torch.as_tensor(vals, dtype=torch.float32).unsqueeze(0)
    c = {name: ctrl[..., INDEX[name]] for name in PARAM_NAMES}
    with torch.no_grad():
        du_lf = eng.glottis(c)["du"][0].double().numpy()[:n]
    vm = np.repeat(voiced, spf)[:n]
    if not vm.any():
        raise SystemExit("유성 프레임이 없다 — 음원 수준을 맞출 수 없다")
    r_lf = float(np.sqrt(np.mean(du_lf[vm] ** 2)))
    r_ph = float(np.sqrt(np.mean(du[vm] ** 2)))
    scale = r_lf / max(r_ph, 1e-30)
    du = du * scale
    print(f"  수준 맞춤: LF 유성 rms {r_lf:.4g}, 물리 {r_ph:.4g} -> 배율 {scale:.4g}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, du=du.astype(np.float32), fs=FS_OUT, t0=a.t0, t1=t1, frame_ms=a.frame_ms,
                        n_frames=n_fr, f0=f0, f0_raw=f0_raw, f0_command=f_cmd, f0_measured=f_meas, voiced=voiced, scale=scale,
                        p_frames=(p_frames if p_frames is not None else np.zeros(0)),
                        adduct_gain=adduct_gain, slope_target=slope_tgt,
                        g_open=g_open.astype(np.float32), area=area48.astype(np.float32),
                        settings=json.dumps(vars(a)))
    peak = max(float(np.abs(du).max()), 1e-12)
    sf.write(os.path.splitext(a.out)[0] + "_du.wav", (du / peak * 0.5).astype(np.float32), FS_OUT)
    print(f"결과: {a.out}  ({n / FS_OUT:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
