#!/usr/bin/env python3
"""벌점 조건별 스펙트럼 대조 — 장기평균 스펙트럼 + 저역 확대 + 스펙트로그램.

    python scripts/plot_penalty_sweep.py out/bench3/s34 out/bench3/s34_d0.03 ...

저장된 `_fit.wav` 만 읽는다 (재렌더 금지, docs/MEASUREMENTS.md §17).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import soundfile as sf                                             # noqa: E402

FS = 48000.0


def use_korean_font() -> None:
    from matplotlib import font_manager as fm
    for path in glob.glob("/usr/share/fonts/**/Nanum*.ttf", recursive=True):
        try:
            fm.fontManager.addfont(path)
        except Exception:
            pass
    if "NanumGothic" in {f.name for f in fm.fontManager.ttflist}:
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False


def lts(x, n=4096):
    w, hop = np.hanning(n), n // 4
    fr = [np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2
          for i in range(0, max(len(x) - n, 1), hop)]
    p = np.mean(fr, axis=0) if fr else np.abs(np.fft.rfft(x, n)) ** 2
    return np.fft.rfftfreq(n, 1 / FS), 10 * np.log10(p / p.max() + 1e-12)


def load(stem):
    t, sr = sf.read(stem + "_target.wav")
    s, _ = sf.read(stem + "_fit.wav")
    t, s = np.asarray(t, float), np.asarray(s, float)
    if sr != FS:
        t = np.interp(np.arange(0, len(t) / sr, 1 / FS), np.arange(len(t)) / sr, t)
    n = min(len(t), len(s))
    t, s = t[:n], s[:n]
    return t, s * (np.sqrt((t ** 2).mean()) / (np.sqrt((s ** 2).mean()) + 1e-12))


def spec(ax, x, title):
    n, hop = 256, 64
    w = np.hanning(n)
    S = 20 * np.log10(np.abs(np.array(
        [np.fft.rfft(x[i:i + n] * w) for i in range(0, len(x) - n, hop)])).T + 1e-9)
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
              vmin=S.max() - 65, vmax=S.max(),
              extent=[0, len(x) / FS * 1000, 0, FS / 2000])
    ax.set_title(title, fontsize=8.5)
    ax.set_ylabel("kHz", fontsize=7.5)
    ax.tick_params(labelsize=6.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--labels", default=None, help="쉼표로 구분한 라벨")
    ap.add_argument("--out", default="out/bench3/penalty_sweep.png")
    a = ap.parse_args()
    labels = (a.labels.split(",") if a.labels
              else [os.path.basename(s) for s in a.stems])
    use_korean_font()

    k = len(a.stems)
    fig = plt.figure(figsize=(6.0 + 2.6 * k, 9.2))
    gs = fig.add_gridspec(3, k, height_ratios=[1.25, 1.0, 1.35], hspace=0.42, wspace=0.22)
    cmap = plt.get_cmap("viridis")

    ax = fig.add_subplot(gs[0, :])
    tgt, _ = load(a.stems[0])
    f, pt = lts(tgt)
    ax.semilogx(f[1:], pt[1:], color="#1b1b1b", lw=1.6, label="목표 (녹음)", zorder=9)
    for i, stem in enumerate(a.stems):
        _, s = load(stem)
        _, ps = lts(s)
        ax.semilogx(f[1:], ps[1:], lw=1.0, alpha=0.85,
                    color=cmap(i / max(k - 1, 1)), label=labels[i])
    ax.set_xlim(80, FS / 2); ax.set_ylim(-72, 3); ax.grid(alpha=0.25, which="both")
    ax.set_title("장기평균 스펙트럼 (최대 대비 dB)", fontsize=9.5)
    ax.set_xlabel("Hz", fontsize=8); ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, :])
    ax.semilogx(f[1:], pt[1:], color="#1b1b1b", lw=1.8, zorder=9)
    for i, stem in enumerate(a.stems):
        _, s = load(stem)
        _, ps = lts(s)
        ax.semilogx(f[1:], ps[1:], lw=1.1, alpha=0.9, color=cmap(i / max(k - 1, 1)))
    ax.set_xlim(80, 600); ax.set_ylim(-55, -5); ax.grid(alpha=0.25, which="both")
    ax.set_title("저역 확대 80~600 Hz — 벌점이 저역 초과를 건드리는가", fontsize=9.5)
    ax.set_xlabel("Hz", fontsize=8); ax.tick_params(labelsize=7)

    spec(fig.add_subplot(gs[2, 0]), tgt, "스펙트로그램 — 목표 (녹음)")
    for i, stem in enumerate(a.stems[1:], start=1):
        _, s = load(stem)
        spec(fig.add_subplot(gs[2, i]), s, f"스펙트로그램 — {labels[i]}")
    for x in fig.axes[-k:]:
        x.set_xlabel("ms", fontsize=7.5)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
