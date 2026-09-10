"""치찰음의 지글거림이 **어느 변조 항**에서 오는가 — 고친 자로 다시.

    python scripts/probe_sizzle_sources.py out/sib/s101 --profile profiles/yang_female.json

왜 다시 재는가
--------------
MEASUREMENTS §12.3 이 이 항들을 이미 기각했는데, 그 기각은 (a) **파일 전체**를 한
덩어리로 재는 자로, (b) **시드 하나**로 한 것이다. 그 뒤에 둘 다 틀린 방식임이
드러났다 — 유성 구간과 마찰 구간은 부호가 반대이고(§13), 시드 하나가 지표를
중앙값의 53~131 % 흔든다(§18). 그래서 같은 항들을 **프레임을 갈라 여러 시드로**
다시 잰다.

적합은 다시 하지 않는다. 이미 적합된 트랙을 놓고 **상수 하나씩만 바꿔 다시 렌더**
한다 — 그래야 적합기의 보상이 섞이지 않는다. 대신 레벨이 달라지므로 레벨 불변인
변조 **지수**로 재고, 대역 레벨도 같이 찍어 "그 대역을 비운 것" 을 가려낸다
(§12.4 가 마찰 0 에서 걸렸던 함정이다).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import soundfile as sf
import torch

from formant_ml.engine import glottis as G
from formant_ml.engine import noise as N
from formant_ml.engine import turbulence as tb
from formant_ml.engine.control import ControlTrack, INDEX
from formant_ml.engine.profile import DEFAULT_PROFILE, SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine

FS = 48000.0
SEEDS = (0, 1, 2, 3, 5, 17)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args()
    prof = SpeakerProfile.load(a.profile) if a.profile else DEFAULT_PROFILE
    seeds = SEEDS[:max(2, a.seeds)]

    tgt, sr = sf.read(a.stem + "_target.wav")
    if tgt.ndim > 1:
        tgt = tgt.mean(1)
    tgt = np.asarray(tgt, float)
    d = np.load(a.stem + "_track.npz")
    vals, fm = d["values"], float(d["frame_ms"])
    hop = int(round(fm * FS / 1000.0))
    ac, ps, f0 = vals[:, INDEX["a_c"]], vals[:, INDEX["p_sub"]], vals[:, INDEX["f0_target"]]
    sel = (ps > 2.0) & (ac < 0.5)
    if sel.sum() < 20:
        print("마찰 프레임이 너무 적다"); return
    f0m = float(np.median(f0[sel]))
    bands = [(0.8 * f0m, 2.5 * f0m), (60.0, 150.0), (150.0, 400.0)]
    msk = np.repeat(sel.astype(float), hop)
    vt, lt = tb.env_modulation_index(tgt, FS, msk, bands)
    cfg = dict(sample_rate=48000, frame_ms=fm, residual=False,
               speaker="female" if prof.f0_nominal > 165 else "male")

    def measure(fric_am, asp_am, mod_depth):
        f0_, a0_ = N.FRIC_AM_DEPTH, G.ASP_AM_DEPTH
        N.FRIC_AM_DEPTH, G.ASP_AM_DEPTH = fric_am, asp_am
        try:
            out = []
            for s in seeds:
                eng = VoiceEngine(EngineConfig(seed=s, **cfg), profile=prof)
                eng.frication.mod_depth = mod_depth
                with torch.no_grad():
                    y = np.asarray(eng.render(ControlTrack(vals.copy(), frame_ms=fm)), float)
                n = min(len(y), len(msk))
                out.append(tb.env_modulation_index(y[:n], FS, msk[:n], bands))
        finally:
            N.FRIC_AM_DEPTH, G.ASP_AM_DEPTH = f0_, a0_
        r = np.array([o[0] for o in out]) / np.maximum(vt, 1e-30)
        return r, float(np.median([o[1] for o in out]))

    base_fric, base_asp = N.FRIC_AM_DEPTH, G.ASP_AM_DEPTH
    base_mod = VoiceEngine(EngineConfig(**cfg), profile=prof).frication.mod_depth
    cases = [
        ("그대로", base_fric, base_asp, base_mod),
        ("마찰 F0 게이트 0", 0.0, base_asp, base_mod),
        ("기식 F0 AM 0", base_fric, 0.0, base_mod),
        ("둘 다 0", 0.0, 0.0, base_mod),
        ("느린 변조 0", base_fric, base_asp, 0.0),
        ("셋 다 0", 0.0, 0.0, 0.0),
    ]
    print(f"{a.stem}  마찰 {sel.mean()*100:.0f}%  F0 {f0m:.0f} Hz  시드 {len(seeds)} 개  "
          f"목표 대역레벨 {lt:.1f} dB")
    print(f"{'조건':>18}  {'F0대역':>20} {'60-150':>20} {'150-400':>20}  {'레벨':>7}")
    for name, fa, aa, md in cases:
        r, lvl = measure(fa, aa, md)
        cells = "  ".join(f"{np.median(r[:, j]):5.2f}x[{r[:, j].min():.2f}~{r[:, j].max():.2f}]"
                          for j in range(3))
        print(f"{name:>18}  {cells}  {lvl:7.1f}")
    print("\n레벨이 크게 바뀌면 '그 대역을 비운 것' 이지 변조를 없앤 것이 아니다 (§12.4).")


if __name__ == "__main__":
    main()
