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
# (음절, 녹음 구간, 정상부 구간(상대), 자음 구간(상대))
CLIPS = {
    "na": (0.20, 0.90, (0.25, 0.55), (0.08, 0.14)),
    "ma": (7.10, 7.80, (0.25, 0.55), (0.06, 0.12)),
    "sa": (14.15, 14.95, (0.40, 0.70), (0.11, 0.23)),
    "ja": (19.30, 20.10, (0.35, 0.65), (0.10, 0.20)),
    "cha": (24.40, 25.20, (0.40, 0.70), (0.10, 0.22)),
}
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
    for syl, (t0, t1, vow, con) in CLIPS.items():
        if args.only and syl not in args.only:
            continue
        a = rec[int(t0 * FS):int(t1 * FS)]
        b = eng.render(builders[syl]())
        # 합성의 첫 무음 50 ms 를 녹음 조각의 개시에 맞춘다: 둘 다 자음 개시가 t=0.05 s 부근이 되게
        a = a / (np.abs(a).max() + 1e-9) * 0.7; b = b / (np.abs(b).max() + 1e-9) * 0.7
        sf.write(os.path.join(args.out, f"{syl}_A_rec.wav"), a, FS)
        sf.write(os.path.join(args.out, f"{syl}_B_syn.wav"), b, FS)
        print(f"\n=== {syl} ===  (A 녹음 {t0}-{t1}s / B 합성 {len(b) / FS:.2f}s)")
        va, vb = a[int(vow[0] * FS):int(vow[1] * FS)], b[int(vow[0] * FS):int(vow[1] * FS)]
        print(f" 모음 정상부 HNR  A {hnr(va, FS):5.1f}  B {hnr(vb, FS):5.1f}")
        print(" 모음 1/3oct A", row(third_oct(va, FS)))
        print("            B", row(third_oct(vb, FS)))
        ca, cb = a[int(con[0] * FS):int(con[1] * FS)], b[int(con[0] * FS):int(con[1] * FS)]
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
