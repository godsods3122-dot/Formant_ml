"""소리가 어디서 끊어지는가 — 시간축으로 본다.

    python scripts/plot_breakup.py out/fix/s040 out/lad/s040 --out out/fig/breakup.png

무엇을 그리는가
    ① 짧은 창(20 ms)의 파형 상관. 1 에 가까우면 그 순간의 파형이 목표와 같다.
       **0 근처로 떨어지는 곳이 귀에 끊겨 들리는 곳**이다 (펄스열이 반 주기 어긋나면
       상관이 음수까지 간다).
    ② 프레임별 멜 dB 오차. 평균만 보면 안 되는 이유 — 한 창의 붕괴는 평균에서
       1/1000 무게다.
    ③ 그 오차의 **창 사이 변화량**. 사용자가 말한 "연속 조건" 이 겨냥하는 양이다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False


def corr_track(t, y, fs, win_ms=20.0, hop_ms=5.0):
    n, h = int(win_ms * fs / 1000), int(hop_ms * fs / 1000)
    m = min(len(t), len(y))
    out, tt = [], []
    for i in range(0, m - n, h):
        a, b = t[i:i + n], y[i:i + n]
        d = np.sqrt((a * a).sum() * (b * b).sum())
        out.append((a * b).sum() / d if d > 1e-14 else np.nan)
        tt.append((i + n / 2) / fs)
    return np.array(tt), np.array(out)


def mel_err(t, y, fs, n=256, n_mels=48):
    import torch
    from formant_ml.engine.fit import mel_bank
    mb = mel_bank(n, fs, n_mels, fmax=0.9 * fs / 2)
    def M(x):
        S = torch.stft(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0), n_fft=n,
                       hop_length=n // 4, win_length=n, window=torch.hann_window(n),
                       center=True, return_complex=True, pad_mode="reflect").abs()[0]
        return 20 * torch.log10(mb[:, :S.shape[0]] @ S + 1e-6)
    a, b = M(t), M(y)
    k = min(a.shape[1], b.shape[1])
    e = (b[:, :k] - a[:, :k]).abs().mean(0).numpy()
    lvl = a[:, :k].mean(0).numpy()
    tt = np.arange(k) * (n // 4) / fs
    return tt, e, lvl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--out", default="out/fig/breakup.png")
    a = ap.parse_args()
    t, fs = sf.read(a.stems[0] + "_target.wav")
    t = np.asarray(t, np.float64)
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    cols = ["tab:blue", "tab:red", "tab:green"]
    for stem, c in zip(a.stems, cols):
        y, _ = sf.read(stem + "_fit.wav")
        y = np.asarray(y, np.float64)
        tag = os.path.basename(os.path.dirname(stem))
        tt, r = corr_track(t, y, fs)
        ax[0].plot(tt, r, c, lw=1.2, label=tag)
        tm, e, lvl = mel_err(t, y, fs)
        ax[1].plot(tm, e, c, lw=0.9, label=tag)
        L = 4
        ax[2].plot(tm[L:], np.maximum(e[L:] - e[:-L], 0.0), c, lw=0.9, label=tag)
    live = None
    tm, _, lvl = mel_err(t, t, fs)
    live = lvl > lvl.max() - 45.0
    for x_ in ax:
        x_.fill_between(tm, *x_.get_ylim(), where=~live, color="0.85", alpha=0.6, lw=0)
        x_.grid(alpha=0.25)
        x_.legend(loc="upper right", fontsize=9)
    ax[0].axhline(0, color="0.4", lw=0.8, ls="--")
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].set_ylabel("상관")
    ax[0].set_title("① 20 ms 창의 파형 상관 — 0 근처로 떨어지는 곳이 끊겨 들린다 (회색 = 무음)")
    ax[1].set_ylabel("dB")
    ax[1].set_title("② 프레임별 멜 dB 오차")
    ax[2].set_ylabel("dB")
    ax[2].set_title("③ 그 오차가 직전 창보다 나빠진 양 — 연속 조건이 무는 것")
    ax[2].set_xlabel("시간 [s]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
