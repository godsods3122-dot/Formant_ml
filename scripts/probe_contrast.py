"""공진 대비 — 봉우리와 골이 얼마나 뚜렷한가.

    python scripts/probe_contrast.py out/fix/s040 out/pole/s040

왜 이 자를 만들었나
    사용자의 청취 소견은 "울림이 부족하고 살짝 플랫하게 들려. 공진 조건이 잘 닫히지
    않은 문제처럼 보여" 였다. 포락 일치율은 그것을 **볼 수 없다** — 멜 48 밴드는
    6 kHz 에서 폭이 700~900 Hz 라 1199 Hz 간격의 포먼트 골을 통째로 뭉갠다.

무엇을 재는가
    짧은 창(256 = 5.3 ms, 분해능 187 Hz)의 선형 dB 스펙트럼에서, 폭 `win_hz` 의
    국소 이동평균을 뺀 잔차의 평균 |·|. 창은 포먼트 간격 c/(2L) ≈ 1200 Hz 보다
    넓게 잡아야 봉우리와 골이 모두 잔차에 남는다.

    짧은 창을 쓰는 이유는 포락 손실과 같다 — 187 Hz 분해능이면 F0 하모닉이 뭉개져
    포락만 남는다. 긴 창을 쓰면 하모닉 빗살이 잔차를 지배해 조음이 아니라 F0 를
    재게 된다.

    **비(적합/목표)로 읽는다.** 1 이면 목표와 같은 만큼 공진이 닫힌 것이고, 1 보다
    작으면 퍼진 소리다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

BANDS = ((300, 1500), (1500, 3000), (3000, 5600), (5600, 8000), (8000, 12000))
N_FFT = 256


def _db(x: np.ndarray, fs: float) -> torch.Tensor:
    w = torch.hann_window(N_FFT)
    S = torch.stft(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0), n_fft=N_FFT,
                   hop_length=N_FFT // 4, win_length=N_FFT, window=w, center=True,
                   return_complex=True, pad_mode="reflect")
    return 20.0 * torch.log10(S.abs() + 1e-8)[0]


def contrast(db: torch.Tensor, fs: float, win_hz: float = 2000.0,
             lo: float = 300.0, hi: float = 12000.0) -> torch.Tensor:
    bw = fs / N_FFT
    k = int(round(win_hz / bw)) | 1
    env = torch.nn.functional.conv1d(db.t().unsqueeze(1),
                                     torch.ones(1, 1, k) / k, padding=k // 2)[:, 0].t()
    f = torch.arange(db.shape[0]) * bw
    m = (f >= lo) & (f < hi)
    return (db - env)[m].abs().mean(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--win", type=float, default=2000.0)
    a = ap.parse_args()
    print(f"공진 대비 (적합/목표). 1.0 = 목표만큼 닫힘, 창 {a.win:.0f} Hz")
    print(f"{'':>12s}{'전체':>8s}" + "".join(f"{lo//1000}-{hi//1000}k".rjust(8)
                                             for lo, hi in BANDS))
    for stem in a.stems:
        t, fs = sf.read(f"{stem}_target.wav")
        y, _ = sf.read(f"{stem}_fit.wav")
        dt, dy = _db(np.asarray(t, np.float32), fs), _db(np.asarray(y, np.float32), fs)
        n = min(dt.shape[1], dy.shape[1])
        dt, dy = dt[:, :n], dy[:, :n]
        lvl = dt.mean(0)
        live = lvl > float(lvl.max()) - 45.0
        cells = []
        for lo, hi in ((300, 12000),) + BANDS:
            ct = contrast(dt, fs, a.win, lo, hi)[live].mean()
            cy = contrast(dy, fs, a.win, lo, hi)[live].mean()
            cells.append(float(cy / ct.clamp_min(1e-6)))
        tag = f"{os.path.basename(os.path.dirname(stem))}/{os.path.basename(stem)}"
        print(f"{tag:>12s}" + "".join(f"{c:8.3f}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
