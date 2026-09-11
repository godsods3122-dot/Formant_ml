"""잡음이 어디서 늘었는지 그린다 — 조화부와 비조화부를 갈라서.

    python scripts/plot_noise.py out/fix/s040 out/lad/s040 --out out/fig/noise.png

왜 이렇게 보는가
    "시끄럽다" 와 "먹먹하다" 는 같은 원인일 수 있다. 잡음 바닥이 올라가면 하모닉
    구조를 덮어서, 소리가 거칠어지는 동시에 또렷함을 잃는다. 그래서 전체
    스펙트럼만 보면 안 되고 **조화부와 비조화부를 갈라서** 봐야 한다.
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

# **설치된 한글 폰트를 골라 쓴다.** 한 이름으로 못박으면 그 폰트가 없는 기계에서
# 라벨이 전부 두부(□)가 되고, 그림이 유일한 산출물인 이 스크립트는 쓸모가 없어진다
# (윈도우에는 NanumGothic 이 없고 Malgun Gothic 이 있다).
def _pick_korean_font() -> str:
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR",
                 "Noto Sans KR", "Gulim", "Batang"):
        if name in have:
            return name
    return rcParams["font.family"][0]


rcParams["font.family"] = _pick_korean_font()
rcParams["axes.unicode_minus"] = False

from formant_ml.engine.waveform import decompose


def psd(x, fs, n=1 << 12):
    w = np.hanning(n)
    acc, c = None, 0
    for i in range(0, max(len(x) - n, 1), n // 2):
        seg = x[i:i + n]
        if len(seg) < n:
            break
        S = np.abs(np.fft.rfft(seg * w)) ** 2
        acc = S if acc is None else acc + S
        c += 1
    return np.fft.rfftfreq(n, 1.0 / fs), 10 * np.log10(acc / max(c, 1) + 1e-30)


def smooth(d, k):
    k = int(k) | 1
    return np.convolve(d, np.ones(k) / k, mode="same")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--src", default="data/voices/yang_00000040.wav")
    ap.add_argument("--out", default="out/fig/noise.png")
    a = ap.parse_args()

    import parselmouth
    y, sr = sf.read(a.src)
    y = np.asarray(y, np.float64)
    hop = 48
    pt = parselmouth.Sound(y, sr).to_pitch(time_step=hop / sr, pitch_floor=90,
                                           pitch_ceiling=700)
    nf = len(y) // hop
    f0 = np.array([pt.get_value_at_time((i + 0.5) * hop / sr) or 0.0 for i in range(nf)])

    items = [("목표", a.stems[0] + "_target.wav")]
    for s in a.stems:
        items.append((os.path.basename(os.path.dirname(s)), s + "_fit.wav"))

    fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    cols = ["k", "tab:blue", "tab:red", "tab:green", "tab:orange"]
    store = {}
    for (tag, path), c in zip(items, cols):
        x, fs = sf.read(path)
        x = np.asarray(x, np.float64)
        _, har, res = decompose(x, fs, f0, hop)
        fr, dh = psd(har, fs)
        _, dr = psd(res, fs)
        _, dt = psd(x, fs)
        k = int(250.0 / (fr[1] - fr[0]))
        dh, dr, dt = smooth(dh, k), smooth(dr, k), smooth(dt, k)
        store[tag] = (fr, dh, dr)
        lw = 2.4 if tag == "목표" else 1.6
        ax[0].semilogx(fr[1:], dt[1:], c, lw=lw, label=tag)
        ax[1].semilogx(fr[1:], dr[1:], c, lw=lw, label=tag)
        ax[2].semilogx(fr[1:], (dh - dr)[1:], c, lw=lw, label=tag)

    ax[0].set_title("① 전체 스펙트럼 — 여기만 보면 차이가 작아 보인다")
    ax[1].set_title("② 비조화(잡음) 성분만 — 위로 올라간 만큼이 늘어난 잡음이다")
    ax[2].set_title("③ 조화 ÷ 비조화 [dB] — 내려간 만큼 하모닉이 잡음에 덮인다")
    for k_, x_ in enumerate(ax):
        x_.grid(True, which="both", alpha=0.25)
        x_.legend(loc="upper right", fontsize=9)
        x_.set_xlim(80, 20000)
        x_.set_ylabel("dB")
    ax[2].set_xlabel("주파수 [Hz]")
    ax[2].axhline(0, color="0.5", lw=0.8, ls="--")
    fr = store["목표"][0]
    for lo, hi in ((1000, 4000),):
        for x_ in ax:
            x_.axvspan(lo, hi, color="tab:red", alpha=0.06)
    ax[1].text(1300, ax[1].get_ylim()[1] - 4, "1~4 kHz", color="tab:red", fontsize=10)
    fig.suptitle(f"{os.path.basename(a.src)} — 잡음이 어디서 늘었나", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
