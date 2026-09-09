#!/usr/bin/env python3
"""치찰음 전용 경로를 눈과 귀로 확인한다 (`engine/sibilant.py`).

    python scripts/sibilant_probe.py --out out/sib_path

자세별 파라미터 표를 찍고, 공용 엔진으로 렌더해서 무게중심·봉우리를 잰다.
DSP 를 새로 만들지 않으므로 **같은 엔진**이고, 다른 것은 손잡이를 채우는 방식이다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from formant_ml.engine import sibilant as sb                       # noqa: E402
from formant_ml.engine import turbulence as tb                     # noqa: E402
from formant_ml.engine.control import PARAMS, ControlTrack         # noqa: E402
from formant_ml.engine.profile import SpeakerProfile               # noqa: E402
from formant_ml.engine.voice import EngineConfig, VoiceEngine      # noqa: E402

FS, HOP = 48000, 48


def build_track(spec: sb.SibilantSpec, prof: SpeakerProfile, ms: int = 220,
                a_g: float = 0.20, p_sub: float = 8.0, fric_gain: float = 24.0):
    """자세 하나를 고원으로 유지하는 트랙. 개시/종결만 코사인으로 연다."""
    n = ms
    vals = np.zeros((n, len(PARAMS)), dtype=np.float32)
    names = list(PARAMS)
    for i, nm in enumerate(names):
        vals[:, i] = PARAMS[nm].default
    c = sb.control_values(spec, p_sub=p_sub, a_g=a_g)
    f1, f2, f3 = prof.vowels["eu"]                     # 중립 혀 자세의 성도
    rise, fall = int(0.20 * n), int(0.16 * n)          # §14.3 의 비례 전이
    ramp = np.ones(n)
    ramp[:rise] = 0.5 * (1 - np.cos(np.pi * np.arange(rise) / rise))
    ramp[n - fall:] = 0.5 * (1 + np.cos(np.pi * np.arange(fall) / fall))
    rest = 3.0
    set_ = lambda nm, v: vals.__setitem__((slice(None), names.index(nm)), v)
    set_("a_c", rest + (c["a_c"] - rest) * ramp)
    set_("c_place", c["c_place"]); set_("front_len", c["front_len"])
    set_("obstacle", c["obstacle"] * ramp); set_("back_leak", c["back_leak"])
    set_("fric_gain", fric_gain * ramp); set_("p_sub", p_sub)
    set_("f1", f1); set_("f2", f2); set_("f3", f3)
    # 유성이면 성대를 울리고, 무성/속삭임이면 끈다. 속삭임은 성문을 벌린 채 압력만.
    set_("adduction", 0.65 if spec.voiced else (0.10 if spec.name == "whisper" else 0.05))
    set_("tension", 0.5)
    if not spec.voiced:
        set_("f0_target", 0.0)
    return ControlTrack(values=vals, frame_ms=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/sib_path")
    ap.add_argument("--profile", default="profiles/yang_female.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    prof = SpeakerProfile.load(a.profile)

    print("자세별 파라미터 (Ps 8 cmH2O, 무성 Ag 0.20 / 유성 Ag 0.04)")
    print(f"{'':<10}{'위치':>6}{'소스':>10}{'a_c':>7}{'앞공동cm':>9}{'앞공동봉우리':>12}"
          f"{'back_leak':>10}{'제트 cm/s':>10}{'obst_eff':>10}")
    for nm, spec in {k: sb.from_profile(k, prof) for k in sb.PRESETS}.items():
        ag = 0.04 if spec.voiced else (0.06 if spec.place < 0.5 else 0.20)
        c = sb.control_values(spec, p_sub=8.0, a_g=ag)
        print(f"{nm:<10}{spec.place:6.2f}{spec.source:>10}{c['a_c']:7.2f}"
              f"{c['front_len']:9.2f}{c['front_peak_hz']:12.0f}{c['back_leak']:10.2f}"
              f"{c['jet_v']:10.0f}{c['obstacle']:10.5f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print("\n렌더 (공용 엔진, 같은 DSP — 손잡이만 이 모듈이 채운다)")
    print(f"{'':<10}{'무게중심':>10}{'봉우리':>9}{'치찰도 dB':>11}   파일")
    for nm in ("s", "ss", "sh", "z", "f", "whisper", "h"):
        spec = sb.from_profile(nm, prof)
        ag = 0.04 if spec.voiced else (0.06 if spec.place < 0.5 else 0.20)
        eng = VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, speaker="female",
                                       residual=False), prof)
        y = eng.render(build_track(spec, prof, a_g=ag))
        y = y / (np.abs(y).max() + 1e-9) * 0.7
        path = f"{a.out}_{nm}.wav"
        sf.write(path, y, FS)
        m = tb.measure(y, float(FS)) if hasattr(tb, "measure") else None
        c = tb.compare(y, y, float(FS))["target"]
        print(f"{nm:<10}{c['centroid']:10.0f}{c['peak']:9.0f}{c['sibilance']:11.1f}   {path}")


if __name__ == "__main__":
    main()
