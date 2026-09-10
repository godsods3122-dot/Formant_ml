"""조화 대 비조화 — **이 프로젝트에서 가장 중요한 자**.

    python scripts/probe_hnr.py out/fix/s040 out/lad/s040

왜 가장 중요한가
    손실의 크기 항은 난류부를 *기대* 스펙트럼으로 비교한다(§9). 그래서 **하모닉과
    잡음이 서로 교환 가능하다** — 적합기가 성문 구동을 반으로 줄이고 마찰 이득을
    11 배로 올려 같은 스펙트럼을 만들 수 있고, 실제로 그렇게 했다 (§44). 총 레벨이
    같으니 포락 일치율은 안 떨어지고 소리만 망가진다.

    스펙트럼 **크기**만 재는 자는 그 사건에 침묵한다. 프레임별 대역 편향도, 공진
    대비도, 장기 평균 대역 오차도 전부 그랬다. 이 자만 바로 말했다.

두 가지를 낸다
    **조화 ÷ 비조화 [dB]** — `waveform.decompose` 로 갈라서 대역별로. 목표에
    가까울수록 좋다. 목표보다 **낮으면 시끄럽고, 높으면 기계적**이다.

    **주기성** — F0 지연에서의 정규화 자기상관 (Praat HNR 과 같은 양). 손실 항
    `--hnr` 이 최적화하는 바로 그 값이라, 그 항이 듣는 것을 그대로 볼 수 있다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.engine.waveform import decompose

BANDS = ((80, 1000, "0.1-1k"), (1000, 4000, "1-4k"),
         (4000, 8000, "4-8k"), (8000, 16000, "8-16k"))


def _band_db(sig, fs, lo, hi, n=1 << 12):
    w = np.hanning(n)
    acc, c = None, 0
    for i in range(0, max(len(sig) - n, 1), n // 2):
        seg = sig[i:i + n]
        if len(seg) < n:
            break
        S = np.abs(np.fft.rfft(seg * w)) ** 2
        acc = S if acc is None else acc + S
        c += 1
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    P = acc / max(c, 1)
    m = (fr >= lo) & (fr < hi)
    return 10 * np.log10(P[m].sum() + 1e-30)


def _bandpass(x, lo, hi, fs):
    n = len(x)
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(X * ((fr >= lo) & (fr < hi)), n)


def _periodicity(x, f0, hop, fs, win_periods=3.0):
    out = np.full(len(f0), np.nan)
    for i, v in enumerate(f0):
        if not np.isfinite(v) or v < 50.0:
            continue
        lag = int(round(fs / v))
        w = int(win_periods * lag)
        a = i * hop - w // 2
        if a < 0 or a + w + lag > len(x):
            continue
        u, z = x[a:a + w], x[a + lag:a + w + lag]
        d = np.sqrt((u * u).sum() * (z * z).sum())
        if d > 1e-20:
            out[i] = (u * z).sum() / d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--src", default=None, help="F0 를 잴 원본 (기본: 첫 stem 의 target)")
    a = ap.parse_args()

    import parselmouth
    src = a.src or (a.stems[0] + "_target.wav")
    y, sr = sf.read(src)
    y = np.asarray(y, np.float64)
    hop_p = 480                                   # 10 ms — 주기성용
    pt = parselmouth.Sound(y, sr).to_pitch(time_step=hop_p / sr, pitch_floor=90,
                                           pitch_ceiling=700)
    n = len(y) // hop_p
    f0p = np.array([pt.get_value_at_time((i + 0.5) * hop_p / sr) for i in range(n)])
    hop_d = 48                                    # 1 ms — decompose 용
    ptd = parselmouth.Sound(y, sr).to_pitch(time_step=hop_d / sr, pitch_floor=90,
                                            pitch_ceiling=700)
    nd = len(y) // hop_d
    f0d = np.nan_to_num(np.array([ptd.get_value_at_time((i + 0.5) * hop_d / sr)
                                  for i in range(nd)]), nan=0.0)

    items = [("목표", a.stems[0] + "_target.wav")]
    for s in a.stems:
        items.append((os.path.basename(os.path.dirname(s)), s + "_fit.wav"))

    print("조화 ÷ 비조화 [dB] — 목표에 가까울수록 좋다. 낮으면 시끄럽고 높으면 기계적이다")
    print(f"{'':>10s} {'전체':>8s} " + " ".join(f"{nm:>8s}" for _, _, nm in BANDS))
    ref = None
    for tag, path in items:
        x, fs = sf.read(path)
        x = np.asarray(x, np.float64)
        _, har, res = decompose(x, fs, f0d, hop_d)
        row = [_band_db(har, fs, 80, 16000) - _band_db(res, fs, 80, 16000)]
        row += [_band_db(har, fs, lo, hi) - _band_db(res, fs, lo, hi)
                for lo, hi, _ in BANDS]
        if ref is None:
            ref = row
            print(f"{tag:>10s} " + " ".join(f"{v:8.1f}" for v in row))
        else:
            print(f"{tag:>10s} " + " ".join(f"{v:8.1f}" for v in row)
                  + "   (목표 대비 " + " ".join(f"{v - r:+.1f}" for v, r in zip(row, ref)) + ")")

    print()
    print("주기성 (F0 지연 자기상관) — `--hnr` 항이 최적화하는 값")
    PB = ((80, 20000, "전대역"), (80, 1000, "0.1-1k"), (1000, 4000, "1-4k"), (4000, 8000, "4-8k"))
    print(f"{'':>10s} " + " ".join(f"{nm:>9s}" for _, _, nm in PB))
    for tag, path in items:
        x, fs = sf.read(path)
        x = np.asarray(x, np.float64)
        row = [np.nanmean(_periodicity(_bandpass(x, lo, hi, fs), f0p, hop_p, fs))
               for lo, hi, _ in PB]
        print(f"{tag:>10s} " + " ".join(f"{v:9.4f}" for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
