"""유음 측정 CLI — 실제 녹음과 합성음을 같은 코드로 재서 비교한다.

    PYTHONPATH=src python3 scripts/analyze_liquid.py FILE.wav [--from S --to S]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from formant_ml.analysis.acoustic import (band_db, energy_db, formants, load,
                                          voicing)


def report(path, t0=None, t1=None, label=""):
    y, sr = load(path)
    if t0 is not None:
        y = y[int(t0 * sr): int((t1 if t1 else len(y) / sr) * sr)]
    F = formants(y, sr)
    e = energy_db(y, sr)
    per, f0 = voicing(y, sr)
    hi = band_db(y, sr, 3000, 7000) - band_db(y, sr, 200, 1000)
    name = label or os.path.basename(path)
    print(f"\n=== {name}  ({len(y)/sr:.2f}s) ===")
    print(f"  유성 프레임 비율 {float((per>0.4).mean()):.2f}   "
          f"F0 중앙값 {np.median(f0[per>0.4]) if (per>0.4).any() else 0:.0f} Hz   "
          f"주기성 중앙값 {np.median(per):.2f}")
    print("   t(s)   dB   per   F0    F1    F2    F3    F4   hi-lo(dB)")
    for i in range(0, len(F), max(1, len(F) // 28)):
        f = F[i]
        s = "  ".join("  -- " if np.isnan(v) else f"{v:5.0f}" for v in f)
        print(f"  {i*0.01:5.2f} {e[i]:5.1f} {per[i]:5.2f} {f0[i]:4.0f}  {s}  {hi[i]:+6.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--from", dest="t0", type=float, default=None)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    report(a.wav, a.t0, a.t1, a.label)
