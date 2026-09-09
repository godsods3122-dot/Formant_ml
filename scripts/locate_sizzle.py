#!/usr/bin/env python3
"""지글거림이 **언제** 나는지 시간축에서 찾는다.

    python scripts/locate_sizzle.py out/bench3/s34 --title "치찰 ㅅ (34)"

왜 필요한가
-----------
`copyfit` 의 지글거림 표는 프레임 **종류**(마찰/유성)로만 갈라서 구간 전체의 한 값을
낸다. 청취는 "특정 순간이 튄다" 인데 그 자로는 순간을 못 짚는다. 포락 92 % 가 귀에
지글거리는 소리를 못 잡았으면 자를 시간축으로 내려야 한다.

무엇을 재는가
-------------
고역(기본 4~12 kHz) 힐베르트 포락의 **변조 지수** std/mean 을 짧은 창으로 미끄러뜨려
목표와 합성에서 각각 낸다. 비율 = 합성/목표. 레벨 차이와 안 섞이도록 창 안에서 제
평균으로 정규화한다 (`turbulence.env_modulation_index` 와 같은 원리, 시간 분해만 추가).

포락 변조가 F0 에 잠겨 있는지도 창마다 본다 — 잠겨 있으면 사람이 거칠기로 듣는다.

`_track.npz` 가 있으면 최악 창에서 어떤 제어 파라미터가 흔들리는지도 같이 낸다.
저장된 `_fit.wav` 를 그대로 읽는다 (재렌더 금지, docs/MEASUREMENTS.md §17).
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
from formant_ml.engine.control import INDEX                        # noqa: E402

TGT_C, FIT_C = "#1b1b1b", "#c8452e"


def use_korean_font() -> None:
    import glob
    from matplotlib import font_manager as fm
    for path in glob.glob("/usr/share/fonts/**/Nanum*.ttf", recursive=True):
        try:
            fm.fontManager.addfont(path)
        except Exception:
            pass
    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("NanumGothic", "NanumBarunGothic", "NanumSquare"):
        if fam in have:
            plt.rcParams["font.family"] = fam
            plt.rcParams["axes.unicode_minus"] = False
            return


def band_envelope(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """대역통과 뒤 해석신호 크기. FFT 로 한 번에 — 창 경계가 안 생긴다."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    X[(f < lo) | (f >= hi)] = 0.0
    # 해석신호: 양의 주파수만 두 배 (rfft 를 그대로 ifft 하면 실수부만 나온다)
    z = np.fft.irfft(X, n) + 1j * np.fft.irfft(-1j * X, n)
    return np.abs(z)


def sliding_mod(e: np.ndarray, fs: float, win: int, hop: int) -> np.ndarray:
    """창마다 변조 지수 std/mean. 창 안에서 정규화하므로 레벨과 무관하다."""
    out = []
    for i in range(0, max(len(e) - win, 1), hop):
        w = e[i:i + win]
        mu = w.mean() + 1e-20
        out.append(w.std() / mu)
    return np.array(out)


def dominant_mod_hz(e: np.ndarray, fs: float) -> float:
    """이 창에서 포락 변조의 우세 주파수. 5 Hz 아래(조음)는 뺀다."""
    z = e - e.mean()
    p = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
    f = np.fft.rfftfreq(len(z), 1.0 / 48000.0)
    m = f >= 5.0
    return float(f[m][np.argmax(p[m])]) if m.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--title", default=None)
    ap.add_argument("--lo", type=float, default=4000.0)
    ap.add_argument("--hi", type=float, default=12000.0)
    ap.add_argument("--win-ms", type=float, default=25.0)
    ap.add_argument("--hop-ms", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=8, help="최악 창 몇 개를 표로 낼까")
    a = ap.parse_args()

    tgt, sr_t = sf.read(a.stem + "_target.wav")
    fit, fs = sf.read(a.stem + "_fit.wav")
    tgt, fit = np.asarray(tgt, float), np.asarray(fit, float)
    if sr_t != fs:
        t_new = np.arange(0, min(len(tgt) / sr_t, len(fit) / fs), 1 / fs)
        tgt = np.interp(t_new, np.arange(len(tgt)) / sr_t, tgt)
    n = min(len(tgt), len(fit))
    tgt, fit = tgt[:n], fit[:n]
    fit = fit * (np.sqrt((tgt ** 2).mean()) / (np.sqrt((fit ** 2).mean()) + 1e-12))

    et = band_envelope(tgt, fs, a.lo, a.hi)
    ef = band_envelope(fit, fs, a.lo, a.hi)
    win, hop = int(a.win_ms * fs / 1000), int(a.hop_ms * fs / 1000)
    mt, mf = sliding_mod(et, fs, win, hop), sliding_mod(ef, fs, win, hop)
    k = min(len(mt), len(mf))
    mt, mf = mt[:k], mf[:k]
    ratio = mf / np.maximum(mt, 1e-12)
    t_win = (np.arange(k) * hop + win / 2) / fs * 1000

    # 창의 에너지가 너무 낮으면(무음) 비율이 의미 없다 — 대역 rms 로 거른다.
    lvl = np.array([20 * np.log10(ef[i:i + win].mean() + 1e-20)
                    for i in range(0, max(len(ef) - win, 1), hop)])[:k]
    live = lvl > (lvl.max() - 25.0)

    print(f"고역 {a.lo/1000:.0f}~{a.hi/1000:.0f} kHz 포락, 창 {a.win_ms:.0f} ms / 홉 {a.hop_ms:.0f} ms")
    print(f"변조 지수 중앙값  목표 {np.median(mt[live]):.3f}  합성 {np.median(mf[live]):.3f}"
          f"  ->  비율 중앙값 {np.median(ratio[live]):.2f}x   (최대 {ratio[live].max():.2f}x)")

    npz = a.stem + "_track.npz"
    trk = fr_ms = None
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        trk, fr_ms = d["values"], float(d["frame_ms"])

    order = np.argsort(-np.where(live, ratio, -1))[:a.top]
    order = np.sort(order)
    print(f"\n가장 튀는 창 {len(order)} 개")
    hdr = f"{'시각 ms':>9}{'비율':>8}{'목표':>8}{'합성':>8}{'우세변조 Hz':>13}{'F0 Hz':>8}"
    if trk is not None:
        hdr += "   그 구간에서 가장 흔들린 파라미터"
    print(hdr)
    for i in order:
        s0 = i * hop
        dom = dominant_mod_hz(ef[s0:s0 + win], fs)
        line = (f"{t_win[i]:9.0f}{ratio[i]:8.2f}{mt[i]:8.3f}{mf[i]:8.3f}"
                f"{dom:13.0f}")
        f0 = float("nan")
        if trk is not None:
            f0a = trk[:, INDEX["f0_target"]]
            j0, j1 = int(s0 / fs * 1000 / fr_ms), int((s0 + win) / fs * 1000 / fr_ms)
            j1 = min(max(j1, j0 + 3), len(trk))
            seg = trk[j0:j1]
            f0 = float(np.median(f0a[j0:j1]))
            # 창 안에서 각 파라미터의 2 차 차분 크기 (제 중앙값으로 정규화 = 상대 잔물결)
            acc = np.abs(np.diff(seg, 2, axis=0)).mean(0)
            rel = acc / (np.abs(np.median(seg, axis=0)) + 1e-9)
            names = list(INDEX)
            top3 = np.argsort(-rel)[:3]
            worst = " ".join(f"{names[t]}({rel[t]:.3f})" for t in top3)
            line += f"{f0:8.0f}   {worst}"
        print(line)

    # ---------------------------------------------------------------- 그림
    use_korean_font()
    title = a.title or os.path.basename(a.stem)
    iw = int(order[np.argmax(ratio[order])]) if len(order) else 0
    z0, z1 = iw * hop, iw * hop + win
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.1, 1.1], hspace=0.62)
    tms = np.arange(n) / fs * 1000

    ax = fig.add_subplot(gs[0])
    ax.plot(tms, et, color=TGT_C, lw=0.7, label="목표")
    ax.plot(tms, ef, color=FIT_C, lw=0.7, alpha=0.8, label="합성")
    ax.axvspan(z0 / fs * 1000, z1 / fs * 1000, color="#f0c040", alpha=0.35, zorder=0)
    ax.set_title(f"{title} — 고역 {a.lo/1000:.0f}~{a.hi/1000:.0f} kHz 포락 (노란 띠 = 최악 창)",
                 fontsize=9)
    ax.set_xlabel("ms", fontsize=8); ax.legend(fontsize=8); ax.tick_params(labelsize=7)
    ax.margins(x=0)

    ax = fig.add_subplot(gs[1])
    ax.plot(t_win, np.where(live, ratio, np.nan), color="#2f5d8a", lw=1.0)
    ax.axhline(1.0, color="#888", ls="--", lw=0.8)
    ax.axvspan(z0 / fs * 1000, z1 / fs * 1000, color="#f0c040", alpha=0.35, zorder=0)
    ax.set_title(f"변조 지수 비율 (합성/목표), 창 {a.win_ms:.0f} ms — 1.0 이 같음", fontsize=9)
    ax.set_xlabel("ms", fontsize=8); ax.tick_params(labelsize=7); ax.margins(x=0)

    ax = fig.add_subplot(gs[2])
    zt = np.arange(z0, z1) / fs * 1000
    ax.plot(zt, et[z0:z1] / (et[z0:z1].mean() + 1e-20), color=TGT_C, lw=1.2, label="목표")
    ax.plot(zt, ef[z0:z1] / (ef[z0:z1].mean() + 1e-20), color=FIT_C, lw=1.2, label="합성")
    if trk is not None and np.isfinite(f0) and f0 > 0:
        for m in np.arange(zt[0], zt[-1], 1000.0 / f0):
            ax.axvline(m, color="#2f8a5d", lw=0.6, alpha=0.55)
        ax.plot([], [], color="#2f8a5d", lw=0.6, label=f"성문 주기 ({f0:.0f} Hz)")
    ax.set_title("최악 창 확대 — 포락을 제 평균으로 나눔", fontsize=9)
    ax.set_xlabel("ms", fontsize=8); ax.legend(fontsize=8); ax.tick_params(labelsize=7)
    ax.margins(x=0)

    ax = fig.add_subplot(gs[3])
    if trk is not None:
        show = ["fric_gain", "tract_gain", "a_c", "p_sub", "aspiration", "tilt"]
        for nm in show:
            v = trk[:, INDEX[nm]]
            tt = np.arange(len(v)) * fr_ms
            ax.plot(tt, v / (np.abs(np.median(v)) + 1e-9), lw=0.9, label=nm)
        ax.axvspan(z0 / fs * 1000, z1 / fs * 1000, color="#f0c040", alpha=0.35, zorder=0)
        ax.set_title("적합된 제어열 (각자 제 중앙값으로 정규화)", fontsize=9)
        ax.set_xlabel("ms", fontsize=8); ax.legend(fontsize=7, ncol=6)
        ax.tick_params(labelsize=7); ax.margins(x=0)

    out = a.stem + "_sizzle.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
