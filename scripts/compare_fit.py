"""적합 결과를 **눈으로** 비교한다 — 파형·스펙트로그램·평균 스펙트럼.

    python scripts/compare_fit.py out/long/s101 --out out/long/s101_compare.png

무엇을 그리는가
---------------
1. **파형** (전체 + 확대 한 주기 구간). 조화 성분은 파형으로 비교할 수 있다.
2. **스펙트로그램** 목표/합성/차이. 어느 시각·어느 대역이 틀렸는지 한눈에 보인다.
3. **시간 평균 스펙트럼** 겹쳐 그리기 + 차이. 난류는 이걸로 비교한다.

왜 파형 확대를 같이 그리는가
----------------------------
크기 스펙트럼이 맞아도 위상이 틀리면 파형은 다르다. 이 프로젝트의 전제가 "기존 TTS 는
스펙트럼만 맞추고 위상을 망가뜨린다" 이므로, 파형이 실제로 겹치는지를 봐야 한다.
다만 **마찰 구간은 겹칠 수 없다** — 난류의 실현은 재현 불가다 (engine/turbulence.py).
그래서 확대는 유성 구간에서 잡는다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from formant_ml.engine import turbulence as tb


def spectrogram(x, fs, n=1024):
    hop = n // 4
    w = np.hanning(n + 1)[:n]
    t = max(1, 1 + (len(x) - n) // hop)
    fr = np.stack([x[i * hop:i * hop + n] * w for i in range(t)])
    S = np.abs(np.fft.rfft(fr, axis=-1)).T
    return 20 * np.log10(S + 1e-8), np.fft.rfftfreq(n, 1 / fs), np.arange(t) * hop / fs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stem", help="copyfit 의 --out 값 (예: out/long/s101)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--zoom-ms", type=float, default=30.0, help="파형 확대 폭")
    ap.add_argument("--zoom-at", type=float, default=None, help="확대 중심 시각 s")
    a = ap.parse_args()

    tgt, sr_t = sf.read(a.stem + "_target.wav")
    syn, sr_s = sf.read(a.stem + "_fit.wav")
    if tgt.ndim > 1:
        tgt = tgt.mean(1)
    if syn.ndim > 1:
        syn = syn.mean(1)
    if sr_t != sr_s:                       # 목표가 48 kHz 가 아니면 맞춘다
        from scipy.signal import resample_poly
        g = np.gcd(int(sr_t), int(sr_s))
        tgt = resample_poly(tgt, sr_s // g, sr_t // g)
    fs = float(sr_s)
    n = min(len(tgt), len(syn))
    tgt, syn = np.asarray(tgt[:n], float), np.asarray(syn[:n], float)

    # 확대 위치: 지정이 없으면 **에너지가 가장 큰 유성부**를 고른다 (마찰은 겹칠 수 없다)
    if a.zoom_at is None:
        w = int(0.02 * fs)
        m = n // w
        r = np.sqrt((tgt[:m * w].reshape(-1, w) ** 2).mean(1))
        a.zoom_at = float((int(np.argmax(r)) + 0.5) * w / fs)
    z0 = max(0, int((a.zoom_at - a.zoom_ms / 2000.0) * fs))
    z1 = min(n, z0 + int(a.zoom_ms / 1000.0 * fs))

    fig, ax = plt.subplots(4, 1, figsize=(13, 13),
                           gridspec_kw={"height_ratios": [1, 1, 2, 1.4]})
    t = np.arange(n) / fs

    ax[0].plot(t, tgt, lw=0.4, color="#1f77b4", label="목표(녹음)")
    ax[0].plot(t, syn, lw=0.4, color="#d62728", alpha=0.75, label="합성")
    ax[0].axvspan(z0 / fs, z1 / fs, color="0.85", zorder=0)
    ax[0].set_xlim(0, n / fs); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"파형 전체 ({n/fs:.2f} s). 회색 = 아래 확대 구간")

    tz = np.arange(z0, z1) / fs
    ax[1].plot(tz, tgt[z0:z1], lw=1.1, color="#1f77b4", label="목표")
    ax[1].plot(tz, syn[z0:z1], lw=1.1, color="#d62728", alpha=0.8, label="합성")
    ax[1].set_xlim(tz[0], tz[-1]); ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title(f"파형 확대 {a.zoom_ms:.0f} ms @ {a.zoom_at:.3f} s "
                    f"— 유성부는 위상까지 겹쳐야 한다")

    St, f, tt = spectrogram(tgt, fs)
    Sp, _, _ = spectrogram(syn, fs)
    m = min(St.shape[1], Sp.shape[1])
    vmax = float(St.max())
    kw = dict(aspect="auto", origin="lower", cmap="magma",
              extent=[0, tt[m - 1], 0, f[-1] / 1000], vmin=vmax - 80, vmax=vmax)
    ax[2].imshow(St[:, :m], **kw)
    ax[2].set_ylabel("목표  kHz")
    ax[2].set_title("스펙트로그램 (위 목표 / 아래 합성은 같은 눈금)")
    div = ax[2].inset_axes([0, -1.05, 1, 1.0])
    div.imshow(Sp[:, :m], **kw)
    div.set_ylabel("합성  kHz")
    div.set_xlabel("s")
    ax[2].set_xticks([])

    ft, pt = tb.average_psd(tgt, fs)
    fp, pp = tb.average_psd(syn, fs)
    k = min(len(pt), len(pp))
    dt = 10 * np.log10(pt[:k] + 1e-20)
    dp = 10 * np.log10(pp[:k] + 1e-20)
    ax[3].plot(ft[:k] / 1000, dt, lw=1.0, color="#1f77b4", label="목표")
    ax[3].plot(ft[:k] / 1000, dp, lw=1.0, color="#d62728", alpha=0.8, label="합성")
    ax[3].plot(ft[:k] / 1000, dp - dt, lw=0.8, color="#2ca02c", alpha=0.7, label="차이")
    ax[3].axhline(0, color="0.6", lw=0.5)
    ax[3].set_xlim(0, min(24, ft[k - 1] / 1000))
    ax[3].set_ylim(dt.max() - 90, dt.max() + 6)
    ax[3].set_xlabel("kHz"); ax[3].set_ylabel("dB")
    ax[3].legend(loc="upper right", fontsize=8)
    ax[3].set_title("시간 평균 스펙트럼 — 난류는 이걸로 비교한다")

    for x in ax:
        x.grid(alpha=0.25)
    plt.tight_layout()
    out = a.out or (a.stem + "_compare.png")
    plt.savefig(out, dpi=110)
    print(f"그림: {out}")

    # 수치 요약
    c = tb.compare(tgt, syn, fs)
    print(f"\n시간평균 스펙트럼 일치 {c['match_spectrum']:.1f} %   대역 MAE "
          f"{c['band_mae_db']:.2f} dB   변조 MAE {c['mod_mae_pct']:.1f} %p")
    print(f"무게중심 {c['target']['centroid']:.0f} -> {c['synth']['centroid']:.0f} Hz "
          f"({c['centroid_err']:+.0f})   치찰도 {c['sibilance_err']:+.1f} dB")
    print(f"rms 목표 {c['target']['rms_db']:.1f} dB / 합성 {c['synth']['rms_db']:.1f} dB")
    print("\n대역 레벨 (총합 대비 dB)")
    edges = tb.LTAS_EDGES
    print("        " + " ".join(f"{lo/1000:g}-{hi/1000:g}k".rjust(9)
                                for lo, hi in zip(edges[:-1], edges[1:])))
    print("목표    " + " ".join(f"{v:9.1f}" for v in c["target"]["bands"]))
    print("합성    " + " ".join(f"{v:9.1f}" for v in c["synth"]["bands"]))


if __name__ == "__main__":
    main()
