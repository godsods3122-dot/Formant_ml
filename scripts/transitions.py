"""전이 검출기 — 전사 정렬 없이 **전이 자체**를 찾아 프레임별 궤적을 낸다.

이 화자는 F0 가 400 Hz 를 넘어 포먼트 추적이 불안정하다. 그래서 포먼트가 아니라
**대역 에너지·유성도·저역/중역 비**로 사건을 찾는다.

  nasal  : 저역(0.1-0.5k) 대 중역(0.8-2.5k) 비가 크고 유성 → 머머
  oral   : 중역이 살아 있는 유성 구간
  fric   : 고역(3-16k)이 지배하고 저역이 죽은 구간

세 가지 전이를 뽑는다.
  N>V / V>N : 머머 ↔ 모음 (연구개 결합의 시간 상수 = 우리가 계단으로 만든 그것)
  F>V       : 마찰 ↔ 발성 (마찰 종료와 발성 개시의 **중첩**)
각 전이마다 5 ms 프레임으로 대역·F0·HNR 궤적을 찍는다.
"""
from __future__ import annotations

import argparse

import numpy as np
import parselmouth
import soundfile as sf


def features(y, sr, step=0.005, win=0.02):
    snd = parselmouth.Sound(y, sr)
    pt = snd.to_pitch(time_step=step, pitch_floor=70, pitch_ceiling=600)
    hn = snd.to_harmonicity_cc(time_step=step, minimum_pitch=70)
    it = snd.to_intensity(minimum_pitch=100, time_step=step)
    n = int(win * sr)
    ts, B, f0, H, dB = [], [], [], [], []
    bands = [(80, 500), (500, 800), (800, 2500), (2500, 6000), (6000, 16000)]
    for t in np.arange(win / 2, len(y) / sr - win / 2, step):
        i = int(t * sr) - n // 2
        s = y[i:i + n] * np.hanning(n)
        Y = np.abs(np.fft.rfft(s)) ** 2
        f = np.fft.rfftfreq(n, 1 / sr)
        ts.append(t)
        B.append([10 * np.log10(Y[(f >= a) & (f < b)].sum() + 1e-20) for a, b in bands])
        v = pt.get_value_at_time(t); f0.append(0.0 if v != v else v)
        v = hn.get_value(t); H.append(-99.0 if (v is None or v != v or v < -99) else v)
        v = it.get_value(t); dB.append(0.0 if v != v else v)
    return (np.array(ts), np.array(B), np.array(f0), np.array(H), np.array(dB))


def classify(B, f0, H, dB):
    lo, lomid, mid, hi, vhi = B.T
    ref = dB.max()
    loud = dB > ref - 30
    voiced = (f0 > 0) & (H > 3)
    nasal = voiced & loud & ((lo - mid) > 20) & ((lo - hi) > 30)
    fric = loud & ((hi + vhi) / 2 - lo > -6) & (~voiced | (H < 2))
    oral = voiced & loud & ~nasal & ~fric
    return nasal, oral, fric, loud, voiced


def runs(mask, ts):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return [(a, b) for a, b in out if ts[b] - ts[a] >= 0.02]


def show(ts, B, f0, H, dB, i0, i1, label):
    print(f"\n--- {label}: {ts[i0]*1000:.0f}~{ts[i1]*1000:.0f} ms")
    print("      t  dB   F0 HNR   80-500 .5-.8k .8-2.5k 2.5-6k 6-16k   lo-mid  hi-lo")
    for i in range(i0, i1 + 1):
        lo, lm, mid, hi, vhi = B[i]
        print(f"  {ts[i]*1000:6.0f} {dB[i]:4.0f} {f0[i]:4.0f} {H[i]:3.0f}  "
              f"{lo:7.1f}{lm:7.1f}{mid:8.1f}{hi:7.1f}{vhi:7.1f}   {lo-mid:6.1f}{(hi+vhi)/2-lo:7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--pad", type=float, default=0.05, help="전이 앞뒤로 볼 시간")
    ap.add_argument("--kinds", default="NV,VN,FV,VF")
    args = ap.parse_args()
    y, sr = sf.read(args.wav, always_2d=True); y = y.mean(1)
    ts, B, f0, H, dB = features(y, sr)
    nasal, oral, fric, loud, voiced = classify(B, f0, H, dB)
    pad = int(args.pad / 0.005)
    kinds = args.kinds.split(",")
    nr, orr, fr = runs(nasal, ts), runs(oral, ts), runs(fric, ts)
    print(f"# {args.wav}: nasal {len(nr)} / oral {len(orr)} / fric {len(fr)} 구간")
    for a, b in nr:
        print(f"  NASAL  {ts[a]*1000:6.0f}~{ts[b]*1000:6.0f} ({(ts[b]-ts[a])*1000:4.0f} ms)")
    for a, b in fr:
        print(f"  FRIC   {ts[a]*1000:6.0f}~{ts[b]*1000:6.0f} ({(ts[b]-ts[a])*1000:4.0f} ms)")
    for a, b in nr:
        if "VN" in kinds:
            show(ts, B, f0, H, dB, max(0, a - pad), min(len(ts) - 1, a + pad), "V>N 진입")
        if "NV" in kinds:
            show(ts, B, f0, H, dB, max(0, b - pad), min(len(ts) - 1, b + pad), "N>V 해제")
    for a, b in fr:
        if "VF" in kinds:
            show(ts, B, f0, H, dB, max(0, a - pad), min(len(ts) - 1, a + pad), "V>F 진입")
        if "FV" in kinds:
            show(ts, B, f0, H, dB, max(0, b - pad), min(len(ts) - 1, b + pad), "F>V 해제")


if __name__ == "__main__":
    main()
