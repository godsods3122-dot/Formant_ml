"""모자란 대역 진단 — 구간·대역마다 합성−목표 dB 와, 목표의 그 대역이 **배음**(유성 음원 몫)인지 **잡음**(기식·마찰 몫)인지.

    .venv/Scripts/python.exe scripts/diag/deficit.py out/C/W17a [--win 0.40:0.48,0.56:0.72] [--hop 40]

배음 몫: 0.04 s 창의 스펙트럼에서 f0 의 배음 ±f0/6 안 에너지 / 대역 전체 에너지 (f0 는 `_track.npz` 의 `f0_target`).
잡음뿐이면 약 1/3, 순수 배음이면 1 에 가깝다. 목표가 배음인데 합성이 모자라면 음원·성도(유성 경로)의 한계, 잡음인데 모자라면 잡음원의 한계다.
"""
import argparse
import sys

import numpy as np
import soundfile as sf

BANDS = ((0.1, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 9.0), (9.0, 13.0), (13.0, 18.0))


def band_stats(x, sr, t0, t1, f0):
    a, b = int(t0 * sr), int(t1 * sr)
    seg = x[a:b] * np.hanning(b - a)
    n = 1 << int(np.ceil(np.log2(len(seg) * 4)))
    P = np.abs(np.fft.rfft(seg, n)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    lev, harm = [], []
    for lo, hi in BANDS:
        m = (f >= lo * 1e3) & (f < hi * 1e3)
        e = P[m].sum()
        lev.append(10 * np.log10(e + 1e-30))
        if f0 > 50:
            k = np.round(f[m] / f0)
            near = np.abs(f[m] - k * f0) < f0 / 6
            harm.append(P[m][near].sum() / max(e, 1e-30))
        else:
            harm.append(np.nan)
    return np.array(lev), np.array(harm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--win", default=None, help="t0:t1,… (s). 없으면 --hop ms 로 전 구간")
    ap.add_argument("--hop", type=float, default=40.0)
    ap.add_argument("--min-db", type=float, default=-50.0, help="목표 전체 대역이 파일 최대보다 이만큼 아래면 건너뜀")
    a = ap.parse_args()
    y, sr = sf.read(a.stem + "_fit.wav")
    x, _ = sf.read(a.stem + "_target.wav")
    z = np.load(a.stem + "_track.npz")
    names = [str(n) for n in z["names"]]
    f0tr = z["values"][:, names.index("f0_target")]
    fm = float(z["frame_ms"]) * 1e-3
    if a.win:
        wins = [tuple(float(v) for v in w.split(":")) for w in a.win.split(",")]
    else:
        h = a.hop * 1e-3
        wins = [(t, t + h) for t in np.arange(0, len(x) / sr - h, h)]
    ref = 10 * np.log10(np.max(np.convolve(x ** 2, np.ones(int(0.04 * sr)), "same")) + 1e-30)
    hd = "".join(f"{f'{lo:g}-{hi:g}k':>13s}" for lo, hi in BANDS)
    print(f"{'구간 s':>12s} {'f0':>5s}{hd}")
    print(f"{'':18s}" + "".join(f"{'합−목 배음':>13s}" for _ in BANDS))
    for t0, t1 in wins:
        e = 10 * np.log10(np.sum(x[int(t0 * sr):int(t1 * sr)] ** 2) + 1e-30)
        if e < ref + a.min_db:
            continue
        i0, i1 = int(t0 / fm), int(t1 / fm)
        f0 = float(np.median(f0tr[i0:i1]))
        lx, hx = band_stats(x, sr, t0, t1, f0)
        ly, _ = band_stats(y, sr, t0, t1, f0)
        cells = []
        for d, h in zip(ly - lx, hx):
            mark = "▼" if d < -6 else ("▲" if d > 6 else " ")
            cells.append(f"{d:+6.1f}{mark}{('' if np.isnan(h) else f'{h:4.2f}'):>5s}")
        print(f"{t0:5.2f}-{t1:5.2f} {f0:5.0f}" + "".join(f"{c:>13s}" for c in cells))


if __name__ == "__main__":
    sys.exit(main())
