"""저장된 적합 결과의 **지글거림**을 여러 시드로 재서 중앙값과 폭을 낸다.

    python scripts/report_sizzle.py --profile profiles/yang_female.json \
        out/sib/w0 out/sib/r0.003 out/sib/r0.01

왜 여러 시드인가 — 단일 시드로는 못 읽는다
-------------------------------------------
난류의 실현은 재현 불가다(engine/turbulence.py). 그래서 **같은 트랙을 시드만 바꿔
렌더해도** 마찰 구간의 변조 지표가 크게 흔들린다. 실측 (같은 적합 트랙, 시드 6 개):

    out/sib/w0     193- 602 Hz  중앙 7.58x  범위 5.08~ 9.13x  (폭  53 %)
                    60- 150 Hz  중앙 9.18x  범위 5.57~17.58x  (폭 131 %)
                   150- 400 Hz  중앙 6.66x  범위 4.40~ 8.27x  (폭  58 %)
    out/sib/s101   194- 605 Hz  중앙 4.43x  범위 3.03~ 6.16x  (폭  71 %)

즉 **시드 하나가 웬만한 조작보다 크게 움직인다.** 단일 시드 값 두 개를 비교해서
"좋아졌다/나빠졌다" 를 말하면 거의 전부 이 잡음을 읽는 것이다. 실제로 그렇게 해서
결론 둘을 잘못 냈다 (docs/MEASUREMENTS.md §17).

그래서 여기서는 트랙을 **시드 여러 개로 다시 렌더**해 중앙값과 범위를 낸다. 두 적합을
비교할 때는 **범위가 겹치는지**를 먼저 보라 — 겹치면 그 차이는 아직 없는 것이다.

재렌더는 `copyfit` 과 같은 `EngineConfig`(residual=False, frame_ms 는 트랙의 값)를
쓴다. 시드만 바뀐다. 지표는 레벨 불변(변조 **지수**)이라 적합기의 전역 이득을
다시 걸 필요가 없다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine import turbulence as tb
from formant_ml.engine.control import ControlTrack, INDEX
from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine

FS = 48000.0
SEEDS = (0, 1, 2, 3, 5, 17)


def _read(p):
    y, sr = sf.read(p)
    if y.ndim > 1:
        y = y.mean(1)
    y = np.asarray(y, float)
    if sr != FS:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), int(FS))
        y = resample_poly(y, int(FS) // g, sr // g)
    return y


def sizzle(stem: str, prof, seeds=SEEDS) -> None:
    tgt = _read(stem + "_target.wav")
    d = np.load(stem + "_track.npz")
    vals, fm = d["values"], float(d["frame_ms"])
    hop = int(round(fm * FS / 1000.0))
    ac, ps, f0 = vals[:, INDEX["a_c"]], vals[:, INDEX["p_sub"]], vals[:, INDEX["f0_target"]]
    cfg = dict(sample_rate=48000, frame_ms=fm, residual=False,
               speaker="female" if prof.f0_nominal > 165 else "male")
    ys = []
    for s in seeds:
        with torch.no_grad():
            ys.append(np.asarray(
                VoiceEngine(EngineConfig(seed=s, **cfg), profile=prof)
                .render(ControlTrack(vals.copy(), frame_ms=fm)), float))
    print(os.path.basename(stem))
    for label, sel in (("마찰", (ps > 2.0) & (ac < 0.5)),
                       ("유성", (ps > 2.0) & (ac > 1.0))):
        if sel.sum() < 20:
            continue
        f0m = float(np.median(f0[sel]))
        bands = [(0.8 * f0m, 2.5 * f0m), (60.0, 150.0), (150.0, 400.0)]
        msk = np.repeat(sel.astype(float), hop)
        vt, _ = tb.env_modulation_index(tgt, FS, msk, bands)
        r = np.array([tb.env_modulation_index(y[:len(msk)], FS, msk[:len(y)], bands)[0]
                      / np.maximum(vt, 1e-30) for y in ys])
        cells = "  ".join(
            f"{name} {np.median(r[:, j]):5.2f}x [{r[:, j].min():.2f}~{r[:, j].max():.2f}]"
            for j, name in enumerate(("F0대역", "60-150", "150-400")))
        print(f"  {label}({sel.mean() * 100:2.0f}%, F0 {f0m:.0f} Hz)  {cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="시드 개수 (기본 6). 줄이면 빨라지지만 폭을 과소평가한다")
    a = ap.parse_args()
    prof = SpeakerProfile.load(a.profile) if a.profile else DEFAULT_PROFILE
    for s in a.stems:
        sizzle(s, prof, SEEDS[:max(2, a.seeds)])


if __name__ == "__main__":
    main()
