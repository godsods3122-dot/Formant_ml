"""v2 엔진 청취 세트 + 측정표.

    OMP_NUM_THREADS=2 python scripts/v2_listen.py --out out/v2 [--praat]

out/v2/*.wav 를 만들고, 기준 녹음(남성) 계측표와 같은 지표를 합성음에서 잰다.
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

FS = 48000


def env_db(y, win=441):
    return np.array([20 * np.log10(np.sqrt((y[i:i + win] ** 2).mean()) + 1e-9)
                     for i in range(0, len(y) - win, win)])


def formant_table(y, label, step=0.01):
    try:
        import parselmouth
    except ImportError:
        print("  (praat-parselmouth 없음: 포먼트 표 생략)"); return
    snd = parselmouth.Sound(y.astype(np.float64), FS)
    fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5,
                             maximum_formant=5500, window_length=0.025)
    it = snd.to_intensity(minimum_pitch=100, time_step=0.005)
    print(f"  [{label}]  t(ms)   dB    F1    F2    F3")
    for t in np.arange(0.02, len(y) / FS - 0.02, step):
        F = [fm.get_value_at_time(k, t) for k in (1, 2, 3)]
        print(f"    {t * 1000:5.0f} {it.get_value(t):5.1f} " +
              " ".join(f"{(v if v == v else 0):5.0f}" for v in F))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/v2")
    ap.add_argument("--praat", action="store_true")
    ap.add_argument("--f0", type=float, default=225.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eng = VoiceEngine(EngineConfig())
    items = [("01_ara_tap", phones.ara), ("02_ra_lateral", phones.ra),
             ("03_sa_sibilant", phones.sa), ("04_na_nasal", phones.na)]
    ys = {}
    for name, fn in items:
        tr = fn(f0=args.f0)
        y = eng.render(tr)
        ys[name] = y
        sf.write(os.path.join(args.out, name + ".wav"), y / (np.abs(y).max() + 1e-9) * 0.7, FS)
        e = env_db(y)
        print(f"{name}: {len(y) / FS:.2f} s, peak {np.abs(y).max():.2f}")
        print("   10 ms RMS dB:", " ".join(f"{v:.0f}" for v in e))
    # 스트리밍으로도 같은 소리가 나는지
    tr = phones.ra(f0=args.f0)
    off = eng.render(tr)
    st = np.concatenate(list(eng.stream(tr, chunk_ms=20)))
    print(f"streaming vs offline max diff: {np.abs(st - off).max():.2e} (peak {np.abs(off).max():.2f})")
    e = env_db(ys["01_ara_tap"])
    print(f"tap dip: {e[10:20].mean() - e[23:31].min():.1f} dB (실측 3.5~5.5)")
    e = env_db(ys["02_ra_lateral"])
    print(f"lateral hold − vowel: {e[8:16].mean() - e[30:45].mean():.1f} dB (실측 −4~−5)")
    if args.praat:
        formant_table(ys["01_ara_tap"], "아라 v2")
        formant_table(ys["02_ra_lateral"], "라 v2")


if __name__ == "__main__":
    main()
