"""원본과 합성음을 **눈으로** 비교한다 — 스펙트로그램·진폭·F0·유성도.

    PYTHONPATH=src python3 scripts/plot_compare.py 원본.wav 합성.wav -o out/cmp.png
    (원본 구간을 잘라 쓰려면 --from/--to)

숫자 하나로 요약하면 놓치는 것이 있다. "어느 순간에" 무엇이 튀는지는 그림이
가장 빠르다 — 특히 전이 구간의 짧은 사건은 평균 오차에 거의 안 잡힌다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from formant_ml.analysis.acoustic import energy_db, load
from formant_ml.analysis.track import track_formants
from formant_ml.data.features import yin_f0

SR = 24000


def spectrogram(y, sr=SR, n_fft=1024, hop=120):
    t = max(1, 1 + (len(y) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(t)[:, None]
    fr = y[idx] * np.hanning(n_fft)
    S = np.abs(np.fft.rfft(fr, n_fft, axis=1)).T
    return 20 * np.log10(S + 1e-8)


def panel(ax, y, title, fmax=8000.0, vtop=None):
    S = spectrogram(y)
    top = vtop if vtop is not None else S.max()
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
              vmin=top - 70, vmax=top,
              extent=[0, len(y) / SR, 0, SR / 2])
    ax.set_ylim(0, fmax)
    ax.set_ylabel("Hz")
    ax.set_title(title, fontsize=9, loc="left")
    return top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("synth")
    ap.add_argument("-o", "--out", default="out/compare.png")
    ap.add_argument("--from", dest="t0", type=float, default=None)
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    o, _ = load(a.original, SR)
    if a.t0 is not None:
        o = o[int(a.t0 * SR): int((a.t1 or len(o) / SR) * SR)]
    s, _ = load(a.synth, SR)
    n = min(len(o), len(s))
    o, s = o[:n] / max(abs(o[:n]).max(), 1e-9), s[:n] / max(abs(s[:n]).max(), 1e-9)

    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 3, 2, 2]})
    top = panel(ax[0], o, f"original  {a.label}")
    panel(ax[1], s, "resynthesis (copy-synthesis)", vtop=top)

    eo, es = energy_db(o, SR), energy_db(s, SR)
    t = np.arange(len(eo)) * 0.01
    ax[2].plot(t, eo - eo.max(), label="original", lw=1.4)
    ax[2].plot(np.arange(len(es)) * 0.01, es - es.max(), label="resynth", lw=1.4)
    ax[2].set_ylabel("dB"); ax[2].set_ylim(-60, 3); ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)

    for sig, lab in ((o, "orig"), (s, "syn")):
        f0, voi = yin_f0(torch.from_numpy(sig).float()[None], SR, 240)
        f0, voi = f0[0].numpy(), voi[0].numpy()
        tt = np.arange(len(f0)) * 0.01
        ax[3].plot(tt, np.where(voi > 0.5, f0, np.nan), lw=1.4, label=f"{lab} F0")
        ax[3].plot(tt, voi * 60, lw=0.9, ls=":", alpha=.7, label=f"{lab} voicing")
    ax[3].set_ylabel("Hz"); ax[3].set_xlabel("time (s)"); ax[3].legend(fontsize=7, ncol=2)
    ax[3].grid(alpha=.3)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(a.out)


if __name__ == "__main__":
    main()
