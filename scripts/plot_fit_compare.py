#!/usr/bin/env python3
"""복사합성 결과의 대조 그림 — 파형 / 장기평균 스펙트럼 / 스펙트로그램.

    python scripts/plot_fit_compare.py out/bench3/s34 --title "치찰 ㅅ (34)"

`copyfit.py` 가 남긴 `_target.wav` / `_fit.wav` 를 **그대로 읽는다**. 재렌더하지
않는다 — 난류가 든 대역에서는 시드 하나가 트랙 조작보다 크게 움직이므로
(docs/MEASUREMENTS.md §17), 그림도 저장된 산출물에서 떠야 한다.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import soundfile as sf                                             # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TGT_C, FIT_C = "#1b1b1b", "#c8452e"


def use_korean_font() -> None:
    """한글 라벨이 두부(□)로 안 나오게. **그림을 만들기 전에** 불러야 한다."""
    import glob
    from matplotlib import font_manager as fm
    for path in glob.glob("/usr/share/fonts/**/Nanum*.ttf", recursive=True):
        try:
            fm.fontManager.addfont(path)
        except Exception:
            pass
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("NanumGothic", "NanumBarunGothic", "NanumSquare", "Noto Sans CJK KR"):
        if fam in have:
            plt.rcParams["font.family"] = fam
            plt.rcParams["axes.unicode_minus"] = False
            return


def lts(x: np.ndarray, sr: float, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """장기평균 스펙트럼 [dB]. 프레임 평균이라 한 실현의 요동이 눌린다."""
    hop, w = n // 4, np.hanning(n)
    frames = [np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2
              for i in range(0, max(len(x) - n, 1), hop)]
    p = np.mean(frames, axis=0) if frames else np.abs(np.fft.rfft(x, n)) ** 2
    return np.fft.rfftfreq(n, 1 / sr), 10 * np.log10(p / p.max() + 1e-12)


def spec(ax, x, sr, title, vmax):
    n, hop = 512, 128
    w = np.hanning(n)
    cols = [np.abs(np.fft.rfft(x[i:i + n] * w)) for i in range(0, len(x) - n, hop)]
    s = 20 * np.log10(np.array(cols).T + 1e-9)
    ax.imshow(s, origin="lower", aspect="auto", cmap="magma",
              vmin=vmax - 70, vmax=vmax,
              extent=[0, len(x) / sr * 1000, 0, sr / 2000])
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("kHz", fontsize=8)
    ax.tick_params(labelsize=7)
    return s.max()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="copyfit 의 --out 값 (예: out/bench3/s34)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--zoom-ms", type=float, default=20.0, help="파형 확대 창 길이")
    a = ap.parse_args()

    tgt, sr_t = sf.read(a.stem + "_target.wav")
    fit, sr_f = sf.read(a.stem + "_fit.wav")
    tgt, fit = np.asarray(tgt, float), np.asarray(fit, float)
    if sr_t != sr_f:                       # copyfit 은 목표를 원 sr 로, 합성을 48 k 로 쓴다
        m = min(len(tgt) / sr_t, len(fit) / sr_f)
        t_new = np.arange(0, m, 1 / sr_f)
        tgt = np.interp(t_new, np.arange(len(tgt)) / sr_t, tgt)
        sr_t = sr_f
    n = min(len(tgt), len(fit))
    tgt, fit = tgt[:n], fit[:n]
    # **rms 정규화만** 한다. 레벨은 copyfit 의 gain 이 이미 맞췄고, 여기서 다시
    # 맞추면 그림이 실제보다 좋아 보인다.
    fit = fit * (np.sqrt((tgt ** 2).mean()) / (np.sqrt((fit ** 2).mean()) + 1e-12))

    use_korean_font()
    title = a.title or os.path.basename(a.stem)
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.35, 1.35], hspace=0.55,
                          wspace=0.22)
    t_ms = np.arange(n) / sr_t * 1000

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t_ms, tgt, color=TGT_C, lw=0.6, label="목표 (녹음)")
    ax.plot(t_ms, fit, color=FIT_C, lw=0.6, alpha=0.75, label="합성")
    ax.set_title(f"{title} — 파형 전체", fontsize=9)
    ax.set_xlabel("ms", fontsize=8); ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=7); ax.margins(x=0)

    # 확대: 에너지가 가장 큰 곳
    win = int(a.zoom_ms * sr_t / 1000)
    e = np.convolve(tgt ** 2, np.ones(win) / win, "same")
    c = int(np.argmax(e)); lo = max(0, c - win // 2); hi = min(n, lo + win)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t_ms[lo:hi], tgt[lo:hi], color=TGT_C, lw=0.9)
    ax.plot(t_ms[lo:hi], fit[lo:hi], color=FIT_C, lw=0.9, alpha=0.8)
    ax.set_title(f"파형 확대 ({a.zoom_ms:.0f} ms, 에너지 최대 지점)", fontsize=9)
    ax.set_xlabel("ms", fontsize=8); ax.tick_params(labelsize=7); ax.margins(x=0)

    ax = fig.add_subplot(gs[2, :])
    f, pt = lts(tgt, sr_t); _, pf = lts(fit, sr_t)
    ax.semilogx(f[1:], pt[1:], color=TGT_C, lw=1.1, label="목표")
    ax.semilogx(f[1:], pf[1:], color=FIT_C, lw=1.1, alpha=0.85, label="합성")
    ax.set_xlim(80, sr_t / 2); ax.set_ylim(-75, 3); ax.grid(alpha=0.25, which="both")
    ax.set_title("장기평균 스펙트럼 (최대 대비 dB)", fontsize=9)
    ax.set_xlabel("Hz", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc="lower left")

    vmax = 20 * np.log10(np.abs(np.fft.rfft(tgt[:512] * np.hanning(512))).max() + 1e-9)
    vmax = max(vmax, -10)
    a1 = fig.add_subplot(gs[3, 0]); spec(a1, tgt, sr_t, "스펙트로그램 — 목표", vmax)
    a2 = fig.add_subplot(gs[3, 1]); spec(a2, fit, sr_t, "스펙트로그램 — 합성", vmax)
    for x in (a1, a2):
        x.set_xlabel("ms", fontsize=8)

    out = a.stem + "_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
