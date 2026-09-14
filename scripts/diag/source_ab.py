"""음원 A/B 통합 판정 — 목표 녹음 대비. `docs/MEASUREMENTS.md` §52 의 표를 한 번에 만든다.

    python scripts/diag/source_ab.py out/M/M14/s101 out/PHY/lc6_s101_fit.wav out/PHY/lcj_s101_fit.wav --range 232 420
    python scripts/diag/source_ab.py --selfcheck          # 척도 자체를 합성 신호로 점검

1. **F0 추종** — 분석기로 잰 렌더 F0 대 목표 F0 (센트). 원시 값과 **51 ms 중앙값 윤곽**(억양) 을 따로 낸다.
   무작위 지터는 목표와 표본 단위로 맞을 수 없으므로, 억양 오차와 **자기 떠돎**(자기 F0 대 자기 윤곽) 을 나눠 본다.
2. **녹음 대비 배음·기울기** — 프레임 이득 정렬 뒤 배음 오차, 광대역 기울기차 (0.5-1k 대 2-4k / 4-8k).
3. **대역별 배음성** (로그 스펙트럼의 F0 간격 자기상관) 과 **HNR** (배음 자리 대 배음 사이 파워).
   **각 신호는 자기 F0 로 잰다** (§52.17). 목표 F0 로 렌더를 재면 22 센트 어긋남만으로 0-2 kHz HNR 이 19 → 5 dB 로
   떨어져, 음원의 잡음이 아니라 추종 오차를 쟀다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, stft

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BANDS = ((300, 2000), (2000, 5000), (5000, 8000), (8000, 12000))
NPERSEG = 4096
HOP = 512
CONTOUR_MS = 51


def band_metrics(x, fs, f0_frames, good_frames, frame_ms=1.0, nperseg=NPERSEG, hop=HOP):
    """대역별 (배음성, HNR) 중앙값. `f0_frames`·`good_frames` 는 `frame_ms` 격자.

    배음성의 추세 제거 창은 **F0 간격의 4 배**다. 예전에는 21 칸 고정이었는데 4096 창(11.7 Hz 칸) 에서 F0 240 Hz 는
    20 칸이라, 추세 제거가 배음 무늬 자체를 지웠다 — 목표 2-5 kHz 가 0.041 로 나왔다.
    """
    f, _t, Z = stft(x, fs, nperseg=nperseg, noverlap=nperseg - hop)
    P = np.abs(Z) ** 2
    S = 20 * np.log10(np.abs(Z) + 1e-10)
    df = f[1] - f[0]
    m = len(f0_frames)
    harm = {b: [] for b in BANDS}
    hnr = {b: [] for b in BANDS}
    for j in range(P.shape[1]):
        tm = min(int(j * hop / fs * 1000.0 / frame_ms), m - 1)
        if not good_frames[tm]:
            continue
        F0 = float(f0_frames[tm])
        lag = int(round(F0 / df))
        dist = np.abs(f / F0 - np.round(f / F0)) * F0
        for lo, hi in BANDS:
            s = (f >= lo) & (f < hi)
            y = S[s, j]
            if len(y) > 3 * lag + 3 and lag >= 3:
                nsm = 4 * lag + 1
                if nsm > len(y):                 # F0 가 높으면 창이 대역보다 길어져 convolve "same" 이 길이를 바꿨다
                    nsm = len(y) if len(y) % 2 else len(y) - 1
                y = y - np.convolve(y, np.ones(nsm) / nsm, "same")
                y = y - y.mean()
                ac = np.correlate(y, y, "full")[len(y) - 1:]
                harm[(lo, hi)].append(ac[lag - 1:lag + 2].max() / max(ac[0], 1e-12))
            on = s & (dist <= 1.5 * df)
            off = s & (dist >= 0.3 * F0)
            if on.sum() >= 2 and off.sum() >= 2:
                hnr[(lo, hi)].append(10 * np.log10(P[on, j].mean() / max(P[off, j].mean(), 1e-30)))
    return ([float(np.median(harm[b])) if harm[b] else np.nan for b in BANDS],
            [float(np.median(hnr[b])) if hnr[b] else np.nan for b in BANDS])


def contour_log2(f, ok, width=CONTOUR_MS):
    """유효 프레임만으로 잰 `width` 프레임 이동 중앙값 (log2 Hz). 무효 프레임은 nan."""
    lc = np.where(ok, np.log2(np.maximum(f, 1.0)), np.nan)
    out = np.full(len(f), np.nan)
    h = width // 2
    for i in np.flatnonzero(ok):
        out[i] = np.nanmedian(lc[max(0, i - h):i + h + 1])
    return out


def _synth(fs, jitter, hnr_db, seconds=1.5, f0=240.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x = np.zeros(n)
    t = 0.0
    while t < seconds - 0.02:
        x[int(t * fs)] += 1.0
        t += (1.0 / f0) * (1.0 + jitter * rng.normal())
    x = lfilter([1.0], [1.0, -0.97], x)
    if hnr_db is not None:
        nz = lfilter([1.0], [1.0, -0.97], rng.normal(size=n))
        x = x + nz * np.std(x) / np.std(nz) * 10 ** (-hnr_db / 20)
    return x


def selfcheck(fs=48000):
    """합성 펄스열(알려진 지터·HNR)로 척도가 **단조**인지 본다. 실패하면 1 을 돌려준다."""
    n_fr = 1500
    f0 = np.full(n_fr, 240.0)
    ok = np.zeros(n_fr, bool)
    ok[100:1400] = True
    rows = [("완전 주기", 0.0, None), ("지터 0.5 %", 0.005, None), ("지터 1 %", 0.01, None),
            ("지터 2 %", 0.02, None), ("HNR 20 dB", 0.0, 20.0), ("HNR 0 dB", 0.0, 0.0)]
    res = {}
    for name, j, h in rows:
        hm, hn = band_metrics(_synth(fs, j, h), fs, f0, ok)
        res[name] = (hm, hn)
        print(f"  {name:10s} 배음성 {' / '.join(f'{v:.3f}' for v in hm)}   HNR {' / '.join(f'{v:5.1f}' for v in hn)} dB")
    bad = []
    for b in range(len(BANDS)):
        seq = [res[k][1][b] for k in ("완전 주기", "지터 0.5 %", "지터 1 %", "지터 2 %")]
        if not all(seq[i] >= seq[i + 1] - 0.5 for i in range(3)):
            bad.append(f"HNR 대역 {BANDS[b]} 이 지터에 단조가 아니다: {np.round(seq, 1)}")
        if not res["HNR 20 dB"][1][b] > res["HNR 0 dB"][1][b] + 10:
            bad.append(f"HNR 대역 {BANDS[b]} 이 가산 잡음을 구별하지 못한다")
        if not res["완전 주기"][0][b] > res["HNR 0 dB"][0][b] + 0.2:
            bad.append(f"배음성 대역 {BANDS[b]} 이 가산 잡음을 구별하지 못한다")
    for s in bad:
        print("  실패:", s)
    print("  통과" if not bad else "  실패")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="목표 F0 가 이 범위 [Hz] 안인 프레임만 따로 한 줄 더 낸다 (모형의 음역)")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    from formant_ml.engine.analyze import analyze
    from formant_ml.engine.profile import SpeakerProfile
    from formant_ml.engine.segment import fricative_mask

    stem, paths = a.stem, a.paths
    prof = SpeakerProfile.load("profiles/yang_female.json")
    t, fs = sf.read(stem + "_target.wav")
    t = np.asarray(t, float)
    tr = analyze(t, fs, prof, 48, t0=0.0, full=t)
    voi = np.asarray(tr.voiced, bool)
    f0 = np.asarray(tr["f0_target"], float)
    fm = np.asarray(fricative_mask(t, fs, prof, 48), bool)
    m = min(len(voi), len(f0), len(fm))
    f0 = f0[:m]
    good = voi[:m] & ~fm[:m] & (f0 > 0)
    t_ct = contour_log2(f0, good)
    t_drift = 1200 * (np.log2(f0[good]) - t_ct[good])

    f2, _t2, A2 = stft(t, fs, nperseg=2048, noverlap=2048 - 256)
    A2 = np.abs(A2)
    th, tn = band_metrics(t, fs, f0, good)
    print(f"목표 {stem}")
    print(f"  자기 떠돎(F0 대 {CONTOUR_MS} ms 윤곽) |차| 중앙 {np.median(np.abs(t_drift)):.1f} 센트")
    print(f"  배음성 {' / '.join(f'{v:.3f}' for v in th)}   HNR {' / '.join(f'{v:.1f}' for v in tn)} dB")
    print()
    for pth in paths:
        y, _ = sf.read(pth)
        y = np.asarray(y, float)
        y = y[:min(len(t), len(y))]
        ty = analyze(y, fs, prof, 48, t0=0.0, full=y)
        fy = np.asarray(ty["f0_target"], float)
        vy = np.asarray(ty.voiced, bool)
        mm = min(m, len(fy), len(vy))
        fy = fy[:mm]
        gg = good[:mm] & vy[:mm] & (fy > 0)
        y_ct = contour_log2(fy, gg)
        subsets = [("전체", gg)]
        if a.range:
            lo, hi = a.range
            subsets.append((f"{lo:.0f}-{hi:.0f} Hz", gg & (f0[:mm] >= lo) & (f0[:mm] <= hi)))
        print(f"  {pth}")
        for label, sel in subsets:
            if not sel.any():
                continue
            c = 1200 * np.log2(fy[sel] / f0[:mm][sel])
            cc = 1200 * (y_ct[sel] - t_ct[:mm][sel])
            dr = 1200 * (np.log2(fy[sel]) - y_ct[sel])
            print(f"    F0 [{label}, {int(sel.sum())} 프레임] 원시 |차| 중앙 {np.median(np.abs(c)):5.1f} 센트 "
                  f"(>50 센트 {100 * np.mean(np.abs(c) > 50):4.1f} %), 윤곽 {np.median(np.abs(cc)):5.1f}, "
                  f"자기 떠돎 {np.median(np.abs(dr)):4.1f}")
        _f, _t, B2 = stft(y, fs, nperseg=2048, noverlap=2048 - 256)
        B2 = np.abs(B2)
        rel, tilts = [], []
        for j in range(min(A2.shape[1], B2.shape[1])):
            tm = min(int(j * 256 / 48), mm - 1)
            if not gg[tm]:
                continue
            Fa, Fb = f0[tm], fy[tm]
            ha, hb = [], []
            for k in range(1, 40):
                if k * Fa > 5000:
                    break
                wa = (f2 > k * Fa - 0.4 * Fa) & (f2 < k * Fa + 0.4 * Fa)
                wb = (f2 > k * Fb - 0.4 * Fb) & (f2 < k * Fb + 0.4 * Fb)
                if wa.sum() < 3 or wb.sum() < 3:
                    continue
                ha.append(20 * np.log10(A2[wa, j].max() + 1e-12))
                hb.append(20 * np.log10(B2[wb, j].max() + 1e-12))
            if len(ha) >= 4:
                ha, hb = np.array(ha), np.array(hb)
                d = hb - ha
                d -= np.median(d)
                rel.extend(np.abs(d[ha > ha.max() - 20]).tolist())
            bnd = lambda S, lo, hi: 20 * np.log10(S[(f2 >= lo) & (f2 < hi), j].mean() + 1e-12)   # noqa: E731
            tilts.append(((bnd(B2, 500, 1000) - bnd(B2, 2000, 4000)) - (bnd(A2, 500, 1000) - bnd(A2, 2000, 4000)),
                          (bnd(B2, 500, 1000) - bnd(B2, 4000, 8000)) - (bnd(A2, 500, 1000) - bnd(A2, 4000, 8000))))
        tl = np.median(np.array(tilts), 0)
        hh, hn = band_metrics(y, fs, fy, gg)
        print(f"    기울기차 2-4k / 4-8k {tl[0]:+6.2f} / {tl[1]:+6.2f} dB   센 배음 오차 중앙 {np.median(rel):5.2f} dB")
        print(f"    배음성 {' / '.join(f'{v:.3f}' for v in hh)}   HNR {' / '.join(f'{v:.1f}' for v in hn)} dB  (자기 F0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
