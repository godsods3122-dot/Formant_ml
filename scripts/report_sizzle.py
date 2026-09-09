"""저장된 적합 결과에서 **지글거림**만 다시 찍는다 — copyfit 과 같은 자.

    python scripts/report_sizzle.py out/sib/w0 out/sib/r0.003 ...

`copyfit` 이 적합 직후에 찍는 것과 같은 계산이다. 예전에 돌린 적합이나 다른
설정으로 돌린 적합을 나란히 놓고 볼 때 쓴다. 목표·합성 wav 와 트랙 npz 만 있으면
되므로 다시 적합할 필요가 없다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf

from formant_ml.engine import turbulence as tb
from formant_ml.engine.control import INDEX

FS = 48000.0


def _read(p):
    y, sr = sf.read(p)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, float)
    if sr != FS:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), int(FS))
        y = resample_poly(y, int(FS) // g, sr // g)
    return y


def sizzle(stem: str) -> None:
    tgt, syn = _read(stem + "_target.wav"), _read(stem + "_fit.wav")
    d = np.load(stem + "_track.npz")
    v, fm = d["values"], float(d["frame_ms"])
    hop = int(round(fm * FS / 1000.0))
    ac, ps, f0 = v[:, INDEX["a_c"]], v[:, INDEX["p_sub"]], v[:, INDEX["f0_target"]]
    print(f"{os.path.basename(stem)}")
    for label, sel in (("마찰", (ps > 2.0) & (ac < 0.5)),
                       ("유성", (ps > 2.0) & (ac > 1.0))):
        if sel.sum() < 20:
            continue
        f0m = float(np.median(f0[sel]))
        bands = [(0.8 * f0m, 2.5 * f0m), (60.0, 150.0), (150.0, 400.0)]
        msk = np.repeat(sel.astype(float), hop)
        vt, lt = tb.env_modulation_index(tgt, FS, msk, bands)
        vs, ls = tb.env_modulation_index(syn, FS, msk, bands)
        r = vs / np.maximum(vt, 1e-30)
        print(f"  {label}({sel.mean()*100:2.0f}%, F0 {f0m:.0f} Hz)  "
              f"F0대역 {r[0]:6.2f}x  60-150 {r[1]:6.2f}x  150-400 {r[2]:6.2f}x   "
              f"대역레벨 {lt:6.1f} -> {ls:6.1f} dB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    for s in ap.parse_args().stems:
        sizzle(s)


if __name__ == "__main__":
    main()
