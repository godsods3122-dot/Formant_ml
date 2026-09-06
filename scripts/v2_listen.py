"""v2 엔진 청취 세트 + 측정표.

    OMP_NUM_THREADS=2 python scripts/v2_listen.py --out out/v2 --profile profiles/yang_female.json [--praat]

out/<dir>/*.wav 를 만들고, 기준 녹음 계측(docs/MEASUREMENTS.md)과 같은 지표를 합성음에서 잰다.
피드백은 **파일 이름 + 커밋 해시 + 이 표**로 남긴다 (docs/VERSIONING.md).
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

torch.set_num_threads(2)

from formant_ml.engine import EngineConfig, VoiceEngine, phones  # noqa: E402
from formant_ml.engine.profile import SpeakerProfile  # noqa: E402

FS = 48000


def env_db(y, win=480):
    return np.array([20 * np.log10(np.sqrt((y[i:i + win] ** 2).mean()) + 1e-9)
                     for i in range(0, len(y) - win, win)])


def bands(y):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / FS); tot = Y.sum()
    s = " ".join(f"{lo // 1000}-{hi // 1000}k:{10 * np.log10(Y[(f >= lo) & (f < hi)].sum() / tot + 1e-20):5.1f}"
                 for lo, hi in [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000),
                                (8000, 10000), (10000, 12000), (12000, 16000), (16000, 22000)])
    return s + f"  centroid {(f * Y).sum() / tot:.0f} Hz"


def formant_table(y, label, step=0.01, t0=0.0, t1=None):
    try:
        import parselmouth
    except ImportError:
        print("  (praat-parselmouth 없음: 포먼트 표 생략)"); return
    snd = parselmouth.Sound(y.astype(np.float64), FS)
    fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5,
                             maximum_formant=5500, window_length=0.02)
    it = snd.to_intensity(minimum_pitch=100, time_step=0.005)
    pt = snd.to_pitch(time_step=0.005, pitch_floor=70, pitch_ceiling=600)
    print(f"  [{label}]  t(ms)   dB   F0    F1    F2    F3")
    t1 = len(y) / FS - 0.02 if t1 is None else t1
    for t in np.arange(max(t0, 0.02), t1, step):
        F = [fm.get_value_at_time(k, t) for k in (1, 2, 3)]; f0 = pt.get_value_at_time(t)
        print(f"    {t * 1000:5.0f} {it.get_value(t):5.1f} {(f0 if f0 == f0 else 0):4.0f} " +
              " ".join(f"{(v if v == v else 0):5.0f}" for v in F))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/v2")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--praat", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prof = SpeakerProfile.load(args.profile) if args.profile else None
    eng = VoiceEngine(EngineConfig(), profile=prof)
    items = [("01_ara_tap", phones.ara), ("02_ra_lateral", phones.ra),
             ("03_sa_lax", phones.sa), ("04_ssa_tense", lambda p: phones.sa(p, tense=True)),
             ("05_na_nasal", phones.na), ("06_ma_nasal", phones.ma),
             ("07_ja_affricate", phones.ja), ("08_cha_aspirated", phones.cha),
             ("09_irinilssirirae", phones.irinilssirirae)]
    if args.only:
        items = [it for it in items if args.only in it[0]]
    ys = {}
    for name, fn in items:
        tr = fn(prof)
        y = eng.render(tr)
        ys[name] = y
        sf.write(os.path.join(args.out, name + ".wav"), y / (np.abs(y).max() + 1e-9) * 0.7, FS)
        e = env_db(y)
        print(f"{name}: {len(y) / FS:.2f} s, peak {np.abs(y).max():.2f}")
        print("   10 ms RMS dB:", " ".join(f"{v:.0f}" for v in e))

    def seg(y, a, b):
        return y[int(a * FS):int(b * FS)]

    if "01_ara_tap" in ys:
        e = env_db(ys["01_ara_tap"])
        print(f"tap dip: {e[8:15].mean() - e[17:25].min():.1f} dB  (프로파일 dip_db {prof.tap['dip_db'] if prof else '-'})")
    if "02_ra_lateral" in ys:
        e = env_db(ys["02_ra_lateral"])
        h1 = int((0.05 + (prof.lateral['hold_ms'] if prof else 120) / 1000) * 100)
        print(f"lateral hold(end 30 ms) − vowel: {e[h1 - 3:h1].mean() - e[h1 + 12:h1 + 20].mean():.1f} dB")
    for key, a, b in (("03_sa_lax", 0.06, 0.15), ("04_ssa_tense", 0.06, 0.11), ("07_ja_affricate", 0.115, 0.18),
                      ("08_cha_aspirated", 0.115, 0.20)):
        if key in ys:
            y = ys[key]; s = seg(y, a, b)
            e = env_db(y); v = e[int((b + 0.08) * 100):int((b + 0.20) * 100)].mean()
            print(f"{key} fricative: {bands(s)}\n    level vs vowel {20 * np.log10(np.sqrt((s ** 2).mean()) + 1e-9) - v:.1f} dB")
            eh = env_db(y, 240)[int(a * 200) - 6:int(b * 200) + 8]
            print("    5 ms env:", " ".join(f"{x:.0f}" for x in eh))
    for key in ("05_na_nasal", "06_ma_nasal"):
        if key in ys:
            y = ys[key]; s = seg(y, 0.06, 0.10); v = seg(y, 0.20, 0.30)
            print(f"{key} murmur: {bands(s)}\n    level vs vowel {20 * np.log10(np.sqrt((s ** 2).mean()) / np.sqrt((v ** 2).mean())):.1f} dB")
    if "02_ra_lateral" in ys:
        tr = phones.ra(prof)
        off = eng.render(tr); st = np.concatenate(list(eng.stream(tr, chunk_ms=20)))
        print(f"streaming vs offline max diff: {np.abs(st - off).max():.2e} (peak {np.abs(off).max():.2f})")
    if args.praat:
        for key in ("01_ara_tap", "02_ra_lateral", "09_irinilssirirae"):
            if key in ys:
                formant_table(ys[key], key)


if __name__ == "__main__":
    main()
