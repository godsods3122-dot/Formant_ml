"""유성 구간 고역의 **F0 동기 맥동**을 잰다 — "지지직" 의 자.

    python scripts/probe_source_buzz.py out/sib/s101

가설
----
성문 펄스가 실제 성대보다 **날카로우면**(사각파에 가까우면) 고차 하모닉이 필요
이상으로 많아진다. 4~12 kHz 에는 F0=250 Hz 기준 하모닉이 30 개 넘게 들어 있고,
그것들의 합은 **한 주기에 한 번 뾰족해진다** — 즉 고역 포락선이 F0 로 맥동한다.
사람은 그 진폭 변조를 거칠기(roughness)로 듣는다.

LTAS 로는 안 잡힌다. 대역 **총 에너지**는 맞춰져 있고(대역 MAE 0.4 dB), 틀린 것은
그 에너지가 **맥동하는 하모닉**과 **고른 잡음** 사이에 어떻게 나뉘는가다.

재는 법
-------
1. 4~12 kHz 를 대역통과 -> 힐베르트 포락 -> 그 포락의 스펙트럼(변조 스펙트럼).
2. F0 부근(0.8~2.5 F0) 과 60~150 Hz 의 **절대** 에너지를 목표와 나란히 놓는다.
   상대값(변조 지수)으로 재면 전체 레벨 차이에 오염된다.
3. 같은 구간의 하모닉/잔차 분해로 4~12 kHz 안에서의 주기성 비율도 같이 낸다.

**유성 · 비마찰 프레임만 쓴다.** 마찰이 든 프레임을 섞으면 난류의 무작위 변조가
F0 대역까지 채워서 두 소리가 다 커 보인다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
from scipy.signal import butter, hilbert, sosfiltfilt

from formant_ml.engine.waveform import decompose

FS = 48000.0


def _read(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != FS:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), int(FS))
        y = resample_poly(y, int(FS) // g, sr // g)
    return np.asarray(y, float)


def hi_env(y, lo=4000.0, hi=12000.0):
    sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, y)))


def mod_energy(env, mask, bands):
    """포락선의 **절대** 변조 에너지. mask 밖은 0 으로 두어 창을 맞춘다."""
    e = env * mask
    e = e - e.mean()
    n = len(e)
    S = np.abs(np.fft.rfft(e * np.hanning(n))) ** 2 / n
    f = np.fft.rfftfreq(n, 1 / FS)
    return [float(S[(f >= a) & (f < b)].sum()) for a, b in bands]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem")
    ap.add_argument("--hop", type=int, default=48)
    ap.add_argument("--frames", choices=("voiced", "fric", "all"), default="voiced",
                    help="voiced=유성·비마찰, fric=마찰이 든 프레임, all=전체")
    a = ap.parse_args()
    tgt, syn = _read(a.stem + "_target.wav"), _read(a.stem + "_fit.wav")
    n = min(len(tgt), len(syn)); tgt, syn = tgt[:n], syn[:n]

    tr = np.load(a.stem + "_track.npz")
    names = [str(x) for x in tr["names"]]
    vals = tr["values"]
    hop = a.hop
    f0 = vals[:, names.index("f0_target")]
    amp = vals[:, names.index("p_sub")]
    a_c = vals[:, names.index("a_c")]
    m = min(len(f0), n // hop)
    # 유성 · 비마찰: 폐압이 서 있고 협착이 열려 있다
    live = amp[:m] > 2.0
    if a.frames == "voiced":
        keep = live & (a_c[:m] > 1.0)
    elif a.frames == "fric":
        keep = live & (a_c[:m] < 0.5)
    else:
        keep = live
    mask = np.repeat(keep.astype(float), hop)[:n]
    if mask.sum() < FS * 0.05:
        print("유성·비마찰 프레임이 너무 적다"); return
    f0m = float(np.median(f0[:m][keep]))
    bands = [(0.8 * f0m, 2.5 * f0m), (60.0, 150.0), (150.0, 400.0)]

    et, es = hi_env(tgt), hi_env(syn)
    mt, ms = mod_energy(et, mask, bands), mod_energy(es, mask, bands)
    print(f"[{a.frames}]  F0 중앙 {f0m:.0f} Hz,  대상 프레임 {mask.mean()*100:.0f} % "
          f"({mask.sum()/FS:.2f} s)")
    print("\n4~12 kHz 포락선의 절대 변조 에너지")
    print(f"{'대역':>16}  {'목표':>11}  {'합성':>11}   배율")
    for (lo, hi), vt, vs in zip(bands, mt, ms):
        print(f"{f'{lo:.0f}-{hi:.0f} Hz':>16}  {vt:11.4e}  {vs:11.4e}   {vs/max(vt,1e-30):5.2f}x")

    # 4~12 kHz 안에서의 하모닉/잔차
    print("\n4~12 kHz 안의 주기성 (하모닉 에너지 / 전체)")
    f0s = np.repeat(f0[:m], hop)[:n]
    for label, y in (("목표", tgt), ("합성", syn)):
        sos = butter(4, [4000 / (FS / 2), 12000 / (FS / 2)], btype="band", output="sos")
        yb = sosfiltfilt(sos, y)
        _, har, res = decompose(yb, FS, f0[:m], hop)
        k = min(len(har), n)
        w = mask[:k] > 0
        eh = float((har[:k][w] ** 2).sum()); er = float((res[:k][w] ** 2).sum())
        nw = max(int(w.sum()), 1)
        print(f"  {label}  하모닉 {100*eh/max(eh+er,1e-30):5.1f} %   "
              f"하모닉 rms {10*np.log10(eh/nw+1e-30):6.1f} dB   "
              f"잔차 rms {10*np.log10(er/nw+1e-30):6.1f} dB")


if __name__ == "__main__":
    main()
