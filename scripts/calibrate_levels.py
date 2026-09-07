"""제스처 세기 보정 — 렌더한 수준 차이를 프로파일의 실측 목표에 맞춘다.

    OMP_NUM_THREADS=2 python scripts/calibrate_levels.py [--write]

무엇을 보정하는가
    lateral.gain   설측 유지부가 뒤 모음보다 `lateral.level_db`
    tap.gain       탄음 골이 앞 모음보다 `tap.level_db`
    nasal.gain     비음 머머가 뒤 모음보다 `nasal.level_db`
    sibilant       `phones.SIB_REF_DB` 기준점 (전 화자 공통) — 여기서는 확인만 한다

왜 다시 해야 했나
    `f0_target` 위에 폐압-F0 결합이 곱해져 지정한 F0 보다 18 % 높게 울리던 버그가
    있었다(0.3.1 에서 고침). 세기 보정은 하모닉이 F1 에 얼마나 걸리느냐에 민감하므로
    F0 가 18 % 틀리면 보정값도 그만큼 틀린다. 실제로 설측이 "모음보다 조용해야"
    하는데 F0 를 고치자 1.4 dB 더 커졌다 — 이전 값이 하모닉 우연에 기대고 있었다.

이 스크립트는 이분법으로 gain 을 찾는다. 렌더 → 수준 차 측정 → 목표와 비교.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from formant_ml.engine import phones
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine

PROFILES = {"yang_female": "profiles/yang_female.json",
            "user_male": "profiles/user_male.json",
            "default": None}          # None = SpeakerProfile 의 dataclass 기본값


def _rms_db(y: np.ndarray, mask: np.ndarray, hop: int) -> float:
    """제어 프레임 마스크가 켜진 구간의 rms (dB). 창은 제스처가 정한다."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return float("nan")
    a, b = idx[0] * hop, (idx[-1] + 1) * hop
    seg = y[a:min(b, len(y))]
    return float(20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12))


def _eng(prof: SpeakerProfile) -> VoiceEngine:
    return VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0,
                                    speaker="female" if prof.f0_nominal > 165 else "male",
                                    residual=False), prof)


def _render(prof: SpeakerProfile, track):
    """(파형, 제어열, hop). 창은 하드코딩하지 않고 **제어열에서** 읽는다 —
    지속시간이 화자마다 다르므로 프레임 번호를 박아 두면 여성 프로파일에서 창이
    엉뚱한 데를 잡는다(실제로 겪었다: 이분법이 하한까지 밀렸다)."""
    eng = _eng(prof)
    return eng.render(track), track, eng.cfg.hop


def measure_lateral(prof: SpeakerProfile) -> float:
    """설측 유지부 − 뒤 모음 정상부 (dB). 유지부는 lat_mix>0.9 인 구간."""
    y, tr, hop = _render(prof, phones.ra(prof))
    mix, ps = tr["lat_mix"], tr["p_sub"]
    hold = (mix > 0.9) & (ps > 0)
    after = np.flatnonzero(mix > 0.1)
    v = np.zeros(tr.n_frames, dtype=bool)
    if len(after):
        v[after[-1] + 40:] = True                 # 전이 40 ms 를 건너뛴다
    v &= ps > 0
    return _rms_db(y, hold, hop) - _rms_db(y, v, hop)


def measure_tap(prof: SpeakerProfile) -> float:
    """탄음 골 − 앞 모음 정상부 (dB). 골은 협착이 가장 좁은 구간."""
    y, tr, hop = _render(prof, phones.ara(prof))
    ac, ps = tr["a_c"], tr["p_sub"]
    dip = (ac < 0.5) & (ps > 0)
    first = np.flatnonzero(dip)
    v = np.zeros(tr.n_frames, dtype=bool)
    if len(first):
        v[max(first[0] - 90, 0):max(first[0] - 20, 1)] = True
    v &= ps > 0
    return _rms_db(y, dip, hop) - _rms_db(y, v, hop)


def measure_nasal(prof: SpeakerProfile) -> float:
    """비음 머머 − 뒤 모음 (dB). 머머는 구강이 닫히고 연구개가 열린 구간."""
    y, tr, hop = _render(prof, phones.na(prof))
    vel, opn, ps = tr["velum"], tr["oral_open"], tr["p_sub"]
    mur = (vel > 0.8) & (opn < 0.2) & (ps > 0)
    m = np.flatnonzero(mur)
    v = np.zeros(tr.n_frames, dtype=bool)
    if len(m):
        v[m[-1] + 60:] = True                     # 개방 전이 60 ms 를 건너뛴다
    v &= (ps > 0) & (vel < 0.2)
    return _rms_db(y, mur, hop) - _rms_db(y, v, hop)


def solve(fn, prof: SpeakerProfile, group: str, target: float,
          lo: float = 0.05, hi: float = 8.0, iters: int = 18) -> tuple[float, float]:
    """gain 을 이분법으로 찾는다. 수준 차는 gain 에 단조 증가한다."""
    d = getattr(prof, group)
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        d["gain"] = mid
        got = fn(prof)
        if got < target:
            lo = mid
        else:
            hi = mid
    d["gain"] = (lo * hi) ** 0.5
    return d["gain"], fn(prof)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="프로파일 json 에 기록")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for name, path in PROFILES.items():
        prof = SpeakerProfile.load(path) if path else SpeakerProfile()
        print(f"\n=== {name} ===")
        for group, fn, key in (("lateral", measure_lateral, "level_db"),
                               ("tap", measure_tap, "dip_db"),
                               ("nasal", measure_nasal, "murmur_db")):
            d = getattr(prof, group)
            if key not in d:
                print(f"  {group:9s} 목표 없음 — 건너뜀")
                continue
            before_gain = d["gain"]
            before = fn(prof)
            g, after = solve(fn, prof, group, float(d[key]))
            print(f"  {group:9s} 목표 {d[key]:+6.1f} dB | 이전 gain {before_gain:5.2f} -> "
                  f"{before:+6.1f} dB | 새 gain {g:5.2f} -> {after:+6.1f} dB")
        if a.write and path:
            prof.save(path)
            print(f"  기록: {path}")
        elif a.write:
            print("  (기본 프로파일은 profile.py 의 dataclass 기본값 — 손으로 옮긴다)")


if __name__ == "__main__":
    main()
