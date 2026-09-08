"""치찰음의 **페이드 인**을 잰다 — v1 이 쓰던 그 자.

    python scripts/probe_sibilant_fade.py --profile profiles/yang_female.json

무엇을 재는가
-------------
대역별로 포락선이 **자기 최대의 −6 dB 에 처음 닿는 시각**을 재고, 고역(11~16 kHz)이
중역(2~4 kHz)보다 얼마나 **늦게** 서는지를 본다. 사람의 /s/ 는 고역이 한참 늦다.

    (같은 화자, 긴 /s/ 두 토큰)      0-2k  2-4k  4-7k  7-11k 11-16k   고역−중역
      사람 실측 #1                    520   287    89    202    216      +127 ms
      사람 실측 #2                    343   467   258    521    554      +296 ms

왜 이 자인가
------------
치찰음은 **파형으로 비교할 수 없다** (engine/turbulence.py). 난류의 실현은 재현
불가라 위상은커녕 순시 진폭도 안 맞는다. 그래서 재현 가능한 것 — 대역 포락선의
**시간 구조** — 으로 잰다. 사람이 "치찰음이 안 맞는다" 고 듣는 것의 큰 몫이
"쉬익" 하고 밝아지는 그 과정이 없다는 것이다.

물리적으로 무엇이 이걸 만드는가
--------------------------------
협착이 닫혀 가는 동안 앞니 다이폴의 **기하 효율**이 커진다 (noise.OBSTACLE_JET_EXP).
협착이 넓으면 제트가 굵고 퍼져 앞니를 못 때리고, 좁아져야 때린다. 그동안 중역은
앞공동 극이 이미 내고 있으므로 중역이 먼저 서고 고역이 뒤따른다.

합격선: 고역−중역 > 80 ms (사람 127/296 ms 의 아래쪽).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from formant_ml.engine import EngineConfig, VoiceEngine, phones
from formant_ml.engine.profile import SpeakerProfile

FS = 48000.0
BANDS = [(0, 2000), (2000, 4000), (4000, 7000), (7000, 11000), (11000, 16000)]


def band_onsets(y, fs=FS, win_ms=5.0, drop_db=6.0):
    """대역 포락선이 자기 최대의 −drop_db 에 **처음** 닿는 시각 [ms]."""
    n = int(win_ms * 1e-3 * fs)
    m = len(y) // n
    fr = y[:m * n].reshape(m, n) * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, axis=-1)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    out = []
    for lo, hi in BANDS:
        e = S[:, (f >= lo) & (f < hi)].sum(1)
        if e.max() <= 0:
            out.append(float("nan")); continue
        thr = e.max() * 10 ** (-drop_db / 10)
        out.append(float(np.argmax(e >= thr) * win_ms))
    return out


def long_s(prof, dur_s=0.60, tense=False):
    """긴 /s/ 하나. `sibilant` 의 dur 를 늘려 고원을 길게 잡는다."""
    b = phones._b(prof, None, None)
    b.sibilant("a", tense=tense, dur=dur_s)
    b.vowel("a", final=True)
    return b.build()


def render(track, prof, seed=17):
    eng = VoiceEngine(EngineConfig(seed=seed), profile=prof)
    with torch.no_grad():
        return eng.render(track)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--dur", type=float, default=0.60)
    ap.add_argument("--rise", type=float, default=None, help="협착 전이 시간 [s] 강제")
    a = ap.parse_args()
    prof = SpeakerProfile.load(a.profile)
    from formant_ml.engine import noise as N, phones as P

    rows = []
    for jet in (0.0, N.OBSTACLE_JET_EXP):
        for vel in (0.0, 1.0):
            jo, vo = N.OBSTACLE_JET_EXP, N.OBSTACLE_VEL_EXP
            N.OBSTACLE_JET_EXP, N.OBSTACLE_VEL_EXP = jet, vel
            y = render(long_s(prof, a.dur), prof)
            N.OBSTACLE_JET_EXP, N.OBSTACLE_VEL_EXP = jo, vo
            rows.append((f"jet={jet:.1f} vel={vel:.1f}", band_onsets(y)))

    hdr = "  ".join(f"{lo//1000}-{hi//1000}k".rjust(7) for lo, hi in BANDS)
    print(f"{'조건':>22}  {hdr}   고역−중역")
    for name, o in rows:
        cells = "  ".join(f"{v:7.0f}" for v in o)
        print(f"{name:>22}  {cells}   {o[4] - o[1]:+7.0f} ms")
    print("\n사람 실측 #1: +127 ms   #2: +296 ms   합격선 > 80 ms")


if __name__ == "__main__":
    main()
