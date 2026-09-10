"""적합 결과의 스펙트럼을 목표와 나란히 놓는다 — 대역 오차·골 깊이·구멍.

    python scripts/bandcmp.py out/fix/s040 out/pole/s040 out/free/s040

무엇을 재는가
    **대역 오차** 합성 − 목표 [dB]. 전체 레벨을 맞춘 뒤 재므로 이득이 아니라
    **모양**의 차이다.
    **골 깊이** 이웃한 봉우리 사이가 얼마나 파이는가. 공진이 "닫히는" 정도로,
    얕으면 소리가 퍼지고 무디게 들린다.
    **구멍** 국소 3 kHz 포락 대비 가장 깊이 파인 곳. 구조적 인공물을 잡는다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

EDGES = [(0.2, 1), (1, 2), (2, 3), (3, 4), (4, 5.6), (5.6, 8), (8, 12), (12, 16)]


def psd(x: np.ndarray, fs: float, n: int = 1 << 13):
    w = np.hanning(n)
    acc = None
    cnt = 0
    for i in range(0, max(len(x) - n, 1), n // 2):
        seg = x[i:i + n]
        if len(seg) < n:
            break
        S = np.abs(np.fft.rfft(seg * w)) ** 2
        acc = S if acc is None else acc + S
        cnt += 1
    return np.fft.rfftfreq(n, 1.0 / fs), 10.0 * np.log10(acc / max(cnt, 1) + 1e-20)


def band(fr, d, lo, hi):
    m = (fr >= lo * 1000) & (fr < hi * 1000)
    return float(d[m].mean())


def dehar(fr, d, width=250.0):
    """하모닉 빗을 지운다. 21 ms 창은 F0 330 Hz 를 분해하므로, 평활하지 않으면
    빗살 사이의 골이 구멍·골 깊이로 잘못 잡힌다 (목표 자체가 −10.5 dB 를 낸다)."""
    k = int(width / (fr[1] - fr[0])) | 1
    return np.convolve(d, np.ones(k) / k, mode="same")


def worst_hole(fr, d, lo, hi, width=3000.0):
    k = int(width / (fr[1] - fr[0])) | 1
    env = np.convolve(d, np.ones(k) / k, mode="same")
    m = (fr > lo) & (fr < hi)
    r = (d - env)[m]
    return float(r.min()), float(fr[m][int(r.argmin())])


def valleys(fr, d, lo=2600.0, hi=8000.0, width=1500.0):
    """봉우리−골 깊이의 중앙값. 국소 포락 위의 잔차에서 극대·극소를 짝짓는다."""
    k = int(width / (fr[1] - fr[0])) | 1
    r = d - np.convolve(d, np.ones(k) / k, mode="same")
    m = (fr > lo) & (fr < hi)
    r = r[m]
    pk = [i for i in range(1, len(r) - 1) if r[i] > r[i - 1] and r[i] >= r[i + 1]]
    vl = [i for i in range(1, len(r) - 1) if r[i] < r[i - 1] and r[i] <= r[i + 1]]
    if len(pk) < 2 or len(vl) < 1:
        return float("nan")
    dep = []
    for v in vl:
        left = [p for p in pk if p < v]
        right = [p for p in pk if p > v]
        if left and right:
            dep.append(0.5 * (r[left[-1]] + r[right[0]]) - r[v])
    return float(np.median(dep)) if dep else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="out/<run>/<stem> (…_target.wav / …_fit.wav)")
    ap.add_argument("--valleys", action="store_true", help="대역별 골 깊이만 낸다")
    a = ap.parse_args()
    rows = []
    tgt_row = None
    for stem in a.stems:
        t, fs = sf.read(f"{stem}_target.wav")
        y, _ = sf.read(f"{stem}_fit.wav")
        fr, dt = psd(np.asarray(t, dtype=np.float64), fs)
        _, dy = psd(np.asarray(y, dtype=np.float64), fs)
        off = float(np.mean([band(fr, dt, lo, hi) for lo, hi in EDGES])
                    - np.mean([band(fr, dy, lo, hi) for lo, hi in EDGES]))
        err = [band(fr, dy, lo, hi) + off - band(fr, dt, lo, hi) for lo, hi in EDGES]
        sy, st = dehar(fr, dy), dehar(fr, dt)
        h, hf = worst_hole(fr, sy, 5500.0, 12000.0)
        rows.append((os.path.basename(os.path.dirname(stem)), err, h, hf,
                     valleys(fr, sy)))
        if tgt_row is None:
            th, thf = worst_hole(fr, st, 5500.0, 12000.0)
            tgt_row = (th, thf, valleys(fr, st))
    if a.valleys:
        seg = [(0.5, 1.5), (1.5, 2.6), (2.6, 4.0), (4.0, 5.6), (5.6, 8.0), (8.0, 12.0)]
        print("골 깊이 [dB] — 봉우리와 골의 차. 얕으면 공진이 닫히지 않고 소리가 퍼진다")
        print(f"{'':>10s}" + "".join(f"{lo}-{hi}k".rjust(9) for lo, hi in seg))
        for stem in a.stems:
            for tag, path in (("목표", f"{stem}_target.wav"), (os.path.basename(
                    os.path.dirname(stem)), f"{stem}_fit.wav")):
                w, fs = sf.read(path)
                fr, d = psd(np.asarray(w, dtype=np.float64), fs)
                d = dehar(fr, d)
                print(f"{tag:>10s}" + "".join(
                    f"{valleys(fr, d, lo * 1000, hi * 1000):9.2f}" for lo, hi in seg))
            print()
        return 0
    head = "".join(f"{lo}-{hi}k".rjust(9) for lo, hi in EDGES)
    print(f"{'':>10s}{head}{'|오차|':>9s}{'구멍':>9s}{'@Hz':>7s}{'골깊이':>8s}")
    for name, err, h, hf, v in rows:
        print(f"{name:>10s}" + "".join(f"{e:+9.1f}" for e in err)
              + f"{np.mean(np.abs(err)):9.2f}{h:9.2f}{hf:7.0f}{v:8.2f}")
    print(f"{'목표':>10s}" + " " * (9 * len(EDGES) + 9)
          + f"{tgt_row[0]:9.2f}{tgt_row[1]:7.0f}{tgt_row[2]:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
