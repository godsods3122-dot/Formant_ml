"""같은 화자 A/B — 사용자 본인 녹음(나/마/사/자/차)과 그 프로파일로 합성한 같은 음절을 나란히 잰다.

    OMP_NUM_THREADS=2 python scripts/ab_male.py --out out/ab_male

* 1/3 옥타브 스펙트럼(정상부), HNR, 3 kHz 고역통과 포락선(5 ms) 을 실측/합성 나란히 출력한다.
* out/ 에 `<syl>_A_rec.wav`(녹음 조각) / `<syl>_B_syn.wav`(합성) 을 쓴다 — 귀로 직접 비교.
목표 화자(여성)와 무관하게 **장치 자체**를 검증하는 도구다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

torch.set_num_threads(2)
from formant_ml.engine import EngineConfig, VoiceEngine, phones  # noqa: E402
from formant_ml.engine.profile import SpeakerProfile  # noqa: E402

REC = "reference/recordings/ko_nasal-sibilant_na-ma-sa-ja-cha_male.wav"
# (음절, 녹음 구간, 자음 종류). 비교 창은 **신호에서 찾는다** — 합성과 녹음의 시간축이
# 다르므로 고정 창을 쓰면 한쪽만 전이를 물어 20 dB 짜리 가짜 차이가 난다(실제로 겪었다).
CLIPS = {
    "na": (0.20, 0.90, "nasal"), "ma": (7.10, 7.80, "nasal"),
    "sa": (14.15, 14.95, "fric"), "ja": (19.30, 20.10, "fric"),
    "cha": (24.40, 25.20, "fric"),
}


def find_windows(y, kind, fs=48000, step=0.005):
    """자음 정상부와 모음 정상부의 구간을 신호에서 찾는다.

    fric : 3~16 kHz 포락선이 최대의 −6 dB 이상인 최장 구간의 가운데 60 %
    nasal: 저역(0.1~0.5k)이 살아 있으면서 중역(0.8~2.5k)이 최대보다 12 dB 낮은 최장 구간
    모음 : 자음이 끝난 뒤 중역이 최대의 −6 dB 이상인 최장 구간의 가운데 60 %
    """
    n = int(0.02 * fs); hop = int(step * fs)
    idx = np.arange(0, len(y) - n, hop)
    F = np.fft.rfftfreq(n, 1 / fs)
    S = np.stack([np.abs(np.fft.rfft(y[i:i + n] * np.hanning(n))) ** 2 for i in idx])
    band = lambda a, b: 10 * np.log10(S[:, (F >= a) & (F < b)].sum(1) + 1e-30)
    lo, mid, hi = band(100, 500), band(800, 2500), band(3000, 16000)

    def longest(mask):
        best = (0, 0, 0); s = None
        for i, m in enumerate(list(mask) + [False]):
            if m and s is None:
                s = i
            elif not m and s is not None:
                if i - s > best[0]:
                    best = (i - s, s, i - 1)
                s = None
        return best[1], best[2]

    if kind == "fric":
        a, b = longest(hi > hi.max() - 6)
    else:
        a, b = longest((mid < mid.max() - 12) & (lo > lo.max() - 12))
    c0, c1 = idx[a] / fs, idx[b] / fs
    d = (c1 - c0) * 0.2
    con = (c0 + d, c1 - d) if c1 - c0 > 0.04 else (c0, c1)
    va, vb = longest((mid > mid.max() - 6) & (idx / fs > c1))
    v0, v1 = idx[va] / fs, idx[vb] / fs
    d = (v1 - v0) * 0.2
    vow = (v0 + d, v1 - d) if v1 - v0 > 0.05 else (v0, v1)
    return vow, con
FS = 48000


def third_oct(y, sr, lo=100, hi=16000):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr); tot = Y[(f >= lo) & (f < hi)].sum()
    edges = lo * 2 ** (np.arange(0, np.log2(hi / lo) + 0.34, 1 / 3))
    return [(a, 10 * np.log10(Y[(f >= a) & (f < b)].sum() / tot + 1e-20)) for a, b in zip(edges[:-1], edges[1:])]


def hf_env(y, sr, hp=3000.0, win=0.005):
    sos = butter(4, hp / (sr / 2), "high", output="sos"); e = sosfiltfilt(sos, y)
    n = int(win * sr)
    return np.array([20 * np.log10(np.sqrt((e[i:i + n] ** 2).mean()) + 1e-9) for i in range(0, len(e) - n, n)])


def hnr(y, sr):
    import parselmouth
    h = parselmouth.Sound(y.astype(np.float64), sr).to_harmonicity_cc(time_step=0.01, minimum_pitch=70)
    v = h.values[h.values > -100]
    return float(np.median(v)) if len(v) else float("nan")


def row(rows):
    return " ".join(f"{a / 1000:.2g}k:{v:.0f}" for a, v in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/ab_male")
    ap.add_argument("--profile", default="profiles/user_male.json")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prof = SpeakerProfile.load(args.profile)
    eng = VoiceEngine(EngineConfig(), profile=prof)
    rec, sr = sf.read(REC, always_2d=True); rec = rec.mean(1)
    g = np.gcd(sr, FS); rec = resample_poly(rec, FS // g, sr // g)
    builders = {"na": lambda: phones.na(prof), "ma": lambda: phones.ma(prof), "sa": lambda: phones.sa(prof),
                "ja": lambda: phones.ja(prof), "cha": lambda: phones.cha(prof)}
    for syl, (t0, t1, kind) in CLIPS.items():
        if args.only and syl not in args.only:
            continue
        a = rec[int(t0 * FS):int(t1 * FS)]
        b = eng.render(builders[syl]())
        # 합성의 첫 무음 50 ms 를 녹음 조각의 개시에 맞춘다: 둘 다 자음 개시가 t=0.05 s 부근이 되게
        a = a / (np.abs(a).max() + 1e-9) * 0.7; b = b / (np.abs(b).max() + 1e-9) * 0.7
        sf.write(os.path.join(args.out, f"{syl}_A_rec.wav"), a, FS)
        sf.write(os.path.join(args.out, f"{syl}_B_syn.wav"), b, FS)
        vowA, conA = find_windows(a, kind); vowB, conB = find_windows(b, kind)
        print(f"\n=== {syl} ===  A 자음 {conA[0]*1000:.0f}~{conA[1]*1000:.0f} 모음 {vowA[0]*1000:.0f}~{vowA[1]*1000:.0f}"
              f" / B 자음 {conB[0]*1000:.0f}~{conB[1]*1000:.0f} 모음 {vowB[0]*1000:.0f}~{vowB[1]*1000:.0f} ms")
        va = a[int(vowA[0] * FS):int(vowA[1] * FS)]; vb = b[int(vowB[0] * FS):int(vowB[1] * FS)]
        print(f" 모음 정상부 HNR  A {hnr(va, FS):5.1f}  B {hnr(vb, FS):5.1f}")
        print(" 모음 1/3oct A", row(third_oct(va, FS)))
        print("            B", row(third_oct(vb, FS)))
        ca = a[int(conA[0] * FS):int(conA[1] * FS)]; cb = b[int(conB[0] * FS):int(conB[1] * FS)]
        la = 20 * np.log10(np.sqrt((ca ** 2).mean()) / np.sqrt((va ** 2).mean()) + 1e-12)
        lb = 20 * np.log10(np.sqrt((cb ** 2).mean()) / np.sqrt((vb ** 2).mean()) + 1e-12)
        print(f" 자음 레벨(모음 대비) A {la:5.1f} dB  B {lb:5.1f} dB")
        print(" 자음 1/3oct A", row(third_oct(ca, FS)))
        print("            B", row(third_oct(cb, FS)))
        ea, eb = hf_env(a, FS), hf_env(b, FS)
        n = min(len(ea), len(eb), 60)
        print(" HF env A", " ".join(f"{v:.0f}" for v in ea[:n]))
        print(" HF env B", " ".join(f"{v:.0f}" for v in eb[:n]))


if __name__ == "__main__":
    main()
