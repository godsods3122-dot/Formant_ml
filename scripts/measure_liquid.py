#!/usr/bin/env python3
"""사용자 녹음에서 유음을 잰다 — `docs/LIQUID.md` §2 의 수치를 그대로 낸다.

    PYTHONPATH=src python scripts/measure_liquid.py --formants --timing --zeros

왜 스크립트로 남기는가: §2 의 표들이 다시 재서 확인 가능해야 한다. 특히 §2.4
(영점 증거 없음)는 예전 구현의 전제를 뒤집는 결론이라, 다음 사람이 의심하면
직접 돌려 볼 수 있어야 한다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.analysis.acoustic import load
from formant_ml.analysis.track import _roots

SR = 24000
REC = os.path.join(os.path.dirname(__file__), "..", "reference", "recordings",
                   "ko_liquid_ra-eulla-ara_male_44k.wav")

#: 구간 [초]. reference/README.md 의 토큰 표와 같은 녹음이다.
SEGMENTS = {
    "라 설측음": (0.53, 0.60),
    "을라 설측음 지속": (1.72, 2.10),
    "아라#1 탄음": (3.565, 3.600),
    "을라 뒤 모음 /아/": (2.22, 2.32),
    "라 뒤 모음 /아/": (0.72, 0.82),
}


def _signal():
    y, _ = load(REC, SR)
    return y / max(np.abs(y).max(), 1e-9)


def _cepstral_envelope(seg, n_fft=1 << 14, quefrency=48):
    """하모닉을 지운 로그 포락선 [dB]. 극·영점 **구조**만 남는다."""
    s = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
    c = np.fft.irfft(np.log(s + 1e-12), n_fft)
    c[quefrency:-quefrency] = 0
    return np.fft.rfftfreq(n_fft, 1.0 / SR), np.fft.rfft(c, n_fft).real * 8.686


def _all_pole_db(freq, bw, fr):
    """DC 정규화 극쌍 캐스케이드의 크기 [dB] — `dsp.filters._pole_pair` 와 같은 식."""
    r = np.exp(-np.pi * np.asarray(bw) / SR)
    th = 2.0 * np.pi * np.asarray(freq) / SR
    z = np.exp(-1j * 2.0 * np.pi * fr / SR)[:, None]
    b1, b2 = -2.0 * r * np.cos(th), r * r
    return 20.0 * np.log10(np.abs((1 + b1 + b2) / (1 + b1 * z + b2 * z * z))).sum(1)


def formants(y, hop=120, win=512):
    """5 ms 간격 / 21 ms 창. **탄음이 40 ms 라 이보다 굵으면 안 보인다.**"""
    from formant_ml.analysis.track import track_formants
    F, _ = track_formants(y, SR, hop, win, n=10)
    rms = np.array([np.sqrt((y[i * hop:i * hop + win] ** 2).mean())
                    for i in range(len(F))])
    print("\n=== 포먼트 (docs/LIQUID.md §2.1) ===")
    for name, (a, b) in SEGMENTS.items():
        i0, i1 = int(a * SR / hop), int(b * SR / hop)
        m = rms[i0:i1].max()
        print(f"\n  {name}  ({a:.3f}~{b:.3f}s)")
        print(f"    {'t[s]':>6} {'rms':>5} {'F1':>5} {'F2':>5} {'F3':>5}")
        for i in range(i0, i1):
            print(f"    {i * hop / SR:6.3f} {rms[i] / m:5.2f} "
                  + " ".join(f"{F[i, k]:5.0f}" for k in range(3)))


def timing(y, hop=120, win=480):
    """세기 골의 폭과 깊이 (docs/LIQUID.md §2.2)."""
    n = (len(y) - win) // hop
    rms = np.array([np.sqrt((y[i * hop:i * hop + win] ** 2).mean()) for i in range(n)])
    t = np.arange(n) * hop / SR
    print("\n=== 시간 구조 (docs/LIQUID.md §2.2) ===")
    print(f"  {'구간':22s} {'폭(50%)':>9} {'깊이':>8}")
    for name, a, b in (("아라#1 탄음", 3.50, 3.66), ("아라#2 탄음", 5.62, 5.78),
                       ("을라 설측음", 1.95, 2.30)):
        m = (t >= a) & (t < b)
        r, tt = rms[m], t[m]
        thr = r.min() + (r.max() - r.min()) * 0.5
        i = int(np.argmin(r))
        lo, hi = i, i
        while lo > 0 and r[lo] < thr:
            lo -= 1
        while hi < len(r) - 1 and r[hi] < thr:
            hi += 1
        print(f"  {name:22s} {(tt[hi] - tt[lo]) * 1000:7.1f} ms "
              f"{20 * np.log10(r[i] / r.max()):+7.1f} dB")


def zeros(y):
    """전극 예측과의 차 — 영점이 필요한가 (docs/LIQUID.md §2.4).

    **모음을 대조군으로 같이 재는 것이 핵심이다.** 모음은 전극이 맞아야 하므로
    거기서 나오는 산포가 이 방법의 잡음 바닥이다(실측 ±3 dB). 그보다 깊고
    **차수에 안 흔들리는** 골만 영점의 증거로 친다.
    """
    print("\n=== 영점 검정 (docs/LIQUID.md §2.4) ===")
    print("  1.2~5.5 kHz 에서 전극이 못 만드는 가장 깊은 골 [dB]")
    print(f"  {'구간':22s} " + " ".join(f"order{o:3d}" for o in (20, 24, 26, 30)))
    for name, (a, b) in SEGMENTS.items():
        seg = y[int(a * SR):int(b * SR)]
        fr, e = _cepstral_envelope(seg)
        x = lfilter([1.0, -0.97], [1.0], seg)
        n = min(len(x), 4096)
        row = []
        for order in (20, 24, 26, 30):
            f, _ = _roots(x[:n] * np.hanning(n), SR, order, 120.0, 9000.0, 900.0)
            f = f[f < 6000][:6]
            if len(f) < 3:
                row.append("     n/a")
                continue
            bw = 50.0 + 20.0 * (f / 1000.0) ** 2 + 10.0 * (f / 1000.0)
            p = _all_pole_db(f, bw, fr)
            band = (fr > 200) & (fr < 6000)
            d = (e - e[band].max()) - (p - p[band].max())
            probe = (fr > 1200) & (fr < 5500)
            row.append(f"{d[probe].min():+8.1f}")
        print(f"  {name:22s} " + " ".join(row))
    print("\n  해석: 모음 두 줄이 대조군이다. 유음이 모음보다 뚜렷하게 더 깊고"
          "\n        차수에 안 흔들려야 영점의 증거다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formants", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--zeros", action="store_true")
    args = ap.parse_args()
    if not (args.formants or args.timing or args.zeros):
        args.formants = args.timing = args.zeros = True
    if not os.path.exists(REC):
        raise SystemExit(f"기준 녹음이 없다: {REC}")
    y = _signal()
    if args.formants:
        formants(y)
    if args.timing:
        timing(y)
    if args.zeros:
        zeros(y)


if __name__ == "__main__":
    main()
