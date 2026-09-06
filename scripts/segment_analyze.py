"""녹음을 5 ms 프레임으로 전수 계측한다 (음소 경계를 손으로 잡기 위한 표).

    python scripts/segment_analyze.py data/ref/female_yang_bibitan.wav [--from 0 --to 3.2] [--step 0.005]

프레임마다:
  dB      : 세기 (Praat Intensity, 100 Hz 최소 피치)
  F0      : Praat pitch (70~600)
  HNR     : 조화음 대 잡음비 (유성/무성·기식 판정)
  F1..F4  : Praat Burg (여성 5500 Hz)
  H1-H2   : 성문 개방지수 대용 (기식성)
  A1-P0   : 비음성 지표 (F1 진폭 − 250 Hz 부근 저역 극 진폭). 음수일수록 비음성 크다
  lo/mid/hi/vhi : 0.1-0.8 / 0.8-2.5 / 2.5-6 / 6-16 kHz 대역 (총 대비 dB)
  zc      : 영교차율 (kHz) — 마찰 유무의 빠른 지표
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import parselmouth
import soundfile as sf


def band_db(y, sr, lo, hi, tot):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return 10 * np.log10(Y[(f >= lo) & (f < hi)].sum() / tot + 1e-20)


def harmonic_amp(y, sr, f0, k):
    """k 번째 하모닉의 진폭 [dB] — 창 신호에 대해 f = k·f0 에서 DFT 직접 계산."""
    n = len(y); w = np.hanning(n); t = np.arange(n) / sr
    z = (y * w * np.exp(-2j * np.pi * k * f0 * t)).sum() / (w.sum() / 2)
    return 20 * np.log10(abs(z) + 1e-12)


def peak_near(y, sr, f_hz, bw=150.0):
    """f_hz 부근 최대 스펙트럼 진폭 [dB]."""
    n = len(y); Y = np.abs(np.fft.rfft(y * np.hanning(n))) / (n / 4)
    f = np.fft.rfftfreq(n, 1 / sr)
    m = (f > f_hz - bw) & (f < f_hz + bw)
    return 20 * np.log10(Y[m].max() + 1e-12) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--win", type=float, default=0.025)
    ap.add_argument("--maxformant", type=float, default=5500.0)
    args = ap.parse_args()

    y, sr = sf.read(args.wav, always_2d=True); y = y.mean(1)
    t1 = args.t1 if args.t1 is not None else len(y) / sr
    snd = parselmouth.Sound(y, sr)
    it = snd.to_intensity(minimum_pitch=100, time_step=args.step)
    pt = snd.to_pitch(time_step=args.step, pitch_floor=70, pitch_ceiling=600)
    fm = snd.to_formant_burg(time_step=args.step, max_number_of_formants=5,
                             maximum_formant=args.maxformant, window_length=0.02)
    hnr = snd.to_harmonicity_cc(time_step=args.step, minimum_pitch=70)
    n = int(args.win * sr)
    print("   t(ms)    dB   F0  HNR    F1    F2    F3    F4  H1H2  A1P0   lo  mid   hi  vhi   zc")
    for t in np.arange(args.t0, t1, args.step):
        i = int(t * sr) - n // 2
        if i < 0 or i + n > len(y):
            continue
        s = y[i:i + n]
        Y = np.abs(np.fft.rfft(s * np.hanning(n))) ** 2
        tot = Y.sum() + 1e-20
        f0 = pt.get_value_at_time(t); db = it.get_value(t); hn = hnr.get_value(t)
        F = [fm.get_value_at_time(k, t) for k in (1, 2, 3, 4)]
        h1h2 = a1p0 = float("nan")
        if f0 == f0 and f0 > 0:
            h1h2 = harmonic_amp(s, sr, f0, 1) - harmonic_amp(s, sr, f0, 2)
            if F[0] == F[0]:
                a1 = peak_near(s, sr, F[0], max(60.0, F[0] * 0.15))
                p0 = peak_near(s, sr, 280.0, 160.0)
                a1p0 = a1 - p0
        zc = ((s[:-1] * s[1:]) < 0).sum() / (2 * args.win) / 1000.0
        print(f"  {t*1000:7.1f} {db if db==db else 0:5.1f} {(f0 if f0==f0 else 0):4.0f} "
              f"{(hn if hn==hn and hn>-100 else -99):4.0f} " +
              " ".join(f"{(v if v==v else 0):5.0f}" for v in F) +
              f" {h1h2 if h1h2==h1h2 else 0:5.1f} {a1p0 if a1p0==a1p0 else 0:5.1f} " +
              " ".join(f"{band_db(s, sr, lo, hi, tot):4.0f}" for lo, hi in
                       ((100, 800), (800, 2500), (2500, 6000), (6000, 16000))) +
              f" {zc:4.1f}")


if __name__ == "__main__":
    main()
