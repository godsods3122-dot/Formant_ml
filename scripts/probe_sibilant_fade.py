"""치찰음의 대역별 **개시 시각**을 잰다 — 그리고 그것을 **목표와 비교**한다.

    python scripts/probe_sibilant_fade.py --profile profiles/yang_female.json
    python scripts/probe_sibilant_fade.py --target out/sib/s101   # 녹음과 나란히

무엇을 재는가
-------------
대역별로 포락선이 **자기 최대의 −6 dB 에 처음 닿는 시각**을 재고, 고역(11~16 kHz)이
중역(2~4 kHz)보다 얼마나 늦게 서는지를 본다.

.. warning::
   **고정 합격선을 쓰지 말 것.** 예전에 v1 이 인용한 사람 실측(긴 지속 /s/ 두 토큰,
   +127 / +296 ms)을 합격선으로 삼았는데, 같은 자를 **이 코퍼스의 실제 녹음**에 대
   보니 완전히 다른 값이 나온다:

       s101 목표 (109 ms 마찰)   135 110  65  65  50   고역−중역 **−60 ms**
       s040 목표 (115 ms 마찰)   155  30  95  55  45   고역−중역 **+15 ms**

   이 화자의 연결발화 치찰음은 고역이 오히려 **먼저** 선다. v1 의 +127/+296 은
   길게 끈 독립 /s/ 의 값이고, 그건 다른 발화다. 그래서 이 자는 **같은 자를 목표와
   합성에 나란히 대는 데** 쓴다 — 절대값의 합격선으로 쓰면 안 된다.

   같은 자로 잰 합성: s101 −15 ms (목표 −60), s040 +15 ms (목표 +15).

물리적으로 무엇이 이 차를 만드는가
----------------------------------
협착이 닫혀 가는 동안 앞니 다이폴의 기하 효율이 커진다 (noise.OBSTACLE_JET_EXP).
협착이 넓으면 제트가 굵고 퍼져 앞니를 못 때리고, 좁아져야 때린다.

**이 자는 정적 스펙트럼 성형에는 둔하다.** 각 대역을 자기 최대로 정규화하므로,
소스를 통째로 밝게/어둡게 하는 조작은 분자와 분모를 같이 움직여 값이 안 변한다
(실측: 속도 의존 소스 기울기 0 → 3.0 에서 +30 ms 로 불변). 값을 움직이는 것은
**대역마다 시간 프로파일의 모양을 다르게 하는** 기제뿐이다.
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


def _compare_target(stem: str) -> None:
    """적합 결과의 목표·합성에 같은 자를 댄다. 마찰 구간을 트랙에서 찾아 잘라 쓴다."""
    import soundfile as sf

    from formant_ml.engine.control import INDEX

    d = np.load(stem + "_track.npz")
    vals, fm = d["values"], float(d["frame_ms"])
    hop = int(round(fm * FS / 1000.0))
    sel = (vals[:, INDEX["p_sub"]] > 2.0) & (vals[:, INDEX["a_c"]] < 0.5)
    idx = np.where(sel)[0]
    if not len(idx):
        print("마찰 프레임이 없다"); return
    runs = np.split(idx, np.where(np.diff(idx) > 5)[0] + 1)
    hdr = "  ".join(f"{lo//1000}-{hi//1000}k".rjust(6) for lo, hi in BANDS)
    print(f"{'구간':>18}  {hdr}   고역−중역")
    for tag, path in (("목표", stem + "_target.wav"), ("합성", stem + "_fit.wav")):
        y, sr = sf.read(path)
        y = np.asarray(y.mean(1) if y.ndim > 1 else y, float)
        for r in runs:
            if len(r) < 80:
                continue
            i0 = max(0, (r[0] - 30) * hop)
            i1 = min(len(y), (r[-1] + 30) * hop)
            o = band_onsets(y[i0:i1])
            cells = "  ".join(f"{v:6.0f}" for v in o)
            print(f"{tag} {len(r):4d} ms 마찰  {cells}   {o[4] - o[1]:+7.0f} ms")
    print("\n목표와 합성의 값을 나란히 볼 것 — 절대값의 합격선은 없다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="profiles/yang_female.json")
    ap.add_argument("--dur", type=float, default=0.60)
    ap.add_argument("--rise", type=float, default=None, help="협착 전이 시간 [s] 강제")
    ap.add_argument("--target", default=None, metavar="STEM",
                    help="적합 결과(copyfit --out 값)의 목표·합성에 같은 자를 대서 "
                         "나란히 본다. 이 자는 절대값의 합격선이 없으므로 "
                         "**이 쪽이 본래 용도**다")
    a = ap.parse_args()
    if a.target:
        _compare_target(a.target)
        return
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
    print("\n**절대값의 합격선은 없다.** 같은 자를 목표 녹음에 대서 나란히 볼 것 —\n"
          "이 화자의 연결발화 치찰음은 고역−중역이 −60 ~ +15 ms 다 (docstring 참조).")


if __name__ == "__main__":
    main()
