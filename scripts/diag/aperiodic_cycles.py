"""주기 동기 비주기 몫 — 목표 펄스로 주기마다 자르고, 이웃 주기(j−1, j+1) 평균을 주기 성분으로 본 나머지의 세기 비 [dB] (§52.474).

    python scripts/diag/aperiodic_cycles.py out/C/W17a_target.wav 판1.wav 판2.wav … [--hop 0.1]

한 주기를 M 표본으로 다시 떠 배음 k → 빈 k. 비주기 몫 = Σ|X_j − (X_{j−1}+X_{j+1})/2|² / Σ|X_j|² 을 대역마다(배음 주파수로) 모아
**구간 중앙값**으로 낸다. 기식·마찰 같은 잡음, 주기마다 위상이 도는 갈라짐, 주기 겹침이 모두 여기 들어온다 — 목표보다 크면 합성이
목표보다 덜 주기적인 것이다. 느린 변화(포먼트 이동·세기 변화)는 이웃 평균이 따라가므로 거의 안 잡힌다.
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
M = 256
BANDS = ((100, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, 10000))


def cycles(x, pul, fs):
    H = []
    for a, b in zip(pul[:-1], pul[1:]):
        i0, i1 = int(round(a * fs)), int(round(b * fs))
        if i1 - i0 < 40 or i1 - i0 > fs // 60 or i1 > len(x):
            H.append(None)
            continue
        s = x[i0:i1]
        r = np.interp(np.linspace(0, len(s), M, endpoint=False), np.arange(len(s)), s)
        H.append((1.0 / (b - a), np.fft.rfft(r)[1:M // 2], 0.5 * (a + b)))
    return H


def table(x, pul, fs, hop, lo_t, hi_t):
    H = cycles(x, pul, fs)
    rows = []
    for j in range(1, len(H) - 1):
        if None in (H[j - 1], H[j], H[j + 1]):
            continue
        f0, X, tc = H[j]
        dev = X - 0.5 * (H[j - 1][1] + H[j + 1][1])
        fk = np.arange(1, len(X) + 1) * f0
        r = []
        for lo, hi in BANDS:
            m = (fk >= lo) & (fk < hi)
            r.append(10 * np.log10((np.abs(dev[m]) ** 2).sum() / max((np.abs(X[m]) ** 2).sum(), 1e-30) + 1e-12) if m.any() else np.nan)
        rows.append((tc, r))
    out = []
    for t0 in np.arange(lo_t, hi_t, hop):
        v = [r for tc, r in rows if t0 <= tc < t0 + hop]
        out.append((t0, np.nanmedian(np.array(v), 0) if v else [np.nan] * len(BANDS)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--hop", type=float, default=0.1)
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    a = ap.parse_args()
    from formant_ml.engine.analyze import glottal_pulses
    from formant_ml.engine.profile import SpeakerProfile
    prof = SpeakerProfile.load(os.path.join(HERE, "..", "..", "profiles", "yang_female.json"))
    t, fs = sf.read(a.target)
    pul = np.asarray(glottal_pulses(t, fs, prof), float)
    t1 = a.t1 or len(t) / fs
    tt = table(t, pul, fs, a.hop, a.t0, t1)
    others = []
    for w in a.wavs:
        x, _ = sf.read(w)
        x = x if x.ndim == 1 else x.mean(1)
        others.append((os.path.basename(os.path.dirname(w)) or w, table(x[:len(t)], pul, fs, a.hop, a.t0, t1)))
    bl = " ".join(f"{lo/1000:g}-{hi/1000:g}k" for lo, hi in BANDS)
    print(f"비주기 몫 [dB] — 목표 값, 그리고 판마다 (판 − 목표). 대역: {bl}")
    for i, (t0, v) in enumerate(tt):
        if np.all(np.isnan(v)):
            continue
        line = f"{t0:5.2f} 목표 " + " ".join(f"{q:6.1f}" for q in v)
        for nm, tab in others:
            d = np.array(tab[i][1]) - np.array(v)
            line += "  | " + " ".join(f"{q:+5.1f}" for q in d)
        print(line)
    print("판 순서: " + ", ".join(nm for nm, _ in others))


if __name__ == "__main__":
    main()
