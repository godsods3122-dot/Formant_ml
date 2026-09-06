"""한국어 음소 제스처 라이브러리 — "텍스트 -> ms 단위 물리 factor 스크립트" 의 첫 판.

LLM 연동 시 LLM 이 만드는 것도 결국 이 형식의 키프레임이다. 여기서는 규칙으로
만든다. **여성 화자 기본값** (Yang 1996 한국어 여성 모음 포먼트, 성도 14.6 cm).

각 음소는 키프레임 몇 개다. 파라미터마다 시간축이 독립이므로 "F3 는 F1/F2 보다
먼저 움직인다"(Ying 2026) 같은 대역별 시차를 그대로 적는다.

유음 /ㄹ/ — docs/RIEUL.md 와 기준 녹음 계측(2026-09-06, 남성 화자)에서:
  * 어두 설측음 [l]: 유지 80~150 ms. F1 ≈ 340, F2 ≈ 1400~1500, F3 ≈ 2450~2600 (남성).
    모음보다 4~5 dB 낮다. 측지 영점 2~5 kHz. 모음으로 90~100 ms 전이.
  * 모음 사이 탄음 [ɾ]: 전체 제스처 60~80 ms, 폐쇄 20~30 ms. F1 골 ≈ −300 Hz
    (700→330~400), F2 +150~250 (치경 로커스), F3 소폭 하강. 세기 골 3.5~5.5 dB.
    발성은 끊기지 않는다. 혀끝 접촉·해제 과도음이 붙는다.
여성 목표는 포먼트별 비(F1 1.35, F2 1.32, F3 1.14)로 옮겼다.
"""
from __future__ import annotations

import numpy as np

from .control import ControlTrack, track_from_keyframes

# 여성 (F1, F2, F3) Hz
VOWELS = {
    "a": (945, 1590, 2850), "eo": (668, 1187, 2882), "o": (486, 889, 2894),
    "u": (401, 1052, 2926), "eu": (464, 1608, 2906), "i": (344, 2814, 3336),
    "e": (494, 2453, 3138), "ae": (630, 2280, 3070),
}
# 유음 자세 (남성 계측 × 비)
LATERAL = (340 * 1.35, 1450 * 1.32, 2550 * 1.14)      # ≈ 459, 1914, 2907
TAP_LOCUS = (360 * 1.35, 1350 * 1.32, 2400 * 1.14)    # ≈ 486, 1782, 2736
ALVEOLAR_LOCUS = (350 * 1.35, 1750 * 1.15, 2650 * 1.14)

DEFAULT = dict(p_sub=0.0, adduction=0.6, tension=0.5, a_c=3.0, c_place=0.9, velum=0.0)


def _kf(t, **kw):
    return dict(t=t, **kw)


class Builder:
    """키프레임을 시간순으로 쌓는다. 파라미터별 독립 시간축."""

    def __init__(self, f0_hz: float = 220.0, p_sub: float = 7.5):
        self.kf: list[dict] = []
        self.t = 0.0
        self.f0, self.ps = f0_hz, p_sub
        self.add(0.0, p_sub=0.0, adduction=0.6, tension=0.5, f0_target=f0_hz,
                 a_c=3.0, c_place=0.9, velum=0.0, tract_gain=1.0, obstacle=0.0)

    def add(self, t, **kw):
        self.kf.append(_kf(round(t, 5), **kw))

    def event(self, t, name, **kw):
        self.kf.append(dict(t=round(t, 5), event=dict(name=name, **kw)))

    # ---------------------------------------------------------- 분절
    def silence(self, dur):
        self.add(self.t, p_sub=0.0); self.t += dur; self.add(self.t, p_sub=0.0)

    def vowel(self, name, dur, f0_end=None, onset_ms=0.0):
        f1, f2, f3 = VOWELS[name]
        t0 = self.t
        self.add(t0, p_sub=self.ps, adduction=0.6, a_c=3.0, f1=f1, f2=f2, f3=f3,
                 tract_gain=1.0, f0_target=self.f0)
        self.t += dur
        f0e = f0_end or self.f0
        self.add(self.t, f1=f1, f2=f2, f3=f3, f0_target=f0e, p_sub=self.ps)
        self.f0 = f0e
        return self

    def lateral_onset(self, hold=0.12, release=0.09, next_vowel="a"):
        """어두 [l]: 측지 영점 + 자세 유지 + 모음으로 전이 (F3 먼저)."""
        l1, l2, l3 = LATERAL
        v1, v2, v3 = VOWELS[next_vowel]
        t0 = self.t
        self.add(t0, p_sub=self.ps, adduction=0.6, a_c=0.6, c_place=0.85,
                 f1=l1, f2=l2, f3=l3, bw1=60, bw2=140, bw3=240,
                 lat_z1=3300.0, lat_z2=4400.0, lat_bw=350.0, tract_gain=1.05)
        self.event(t0 + 0.004, "tongue_contact", amp=0.05)
        t1 = t0 + hold
        self.add(t1, f1=l1, f2=l2, f3=l3, lat_z1=3300.0, lat_z2=4400.0,
                 tract_gain=1.05, a_c=0.6, bw1=60, bw2=140, bw3=240)
        self.add(t1 - 0.012, f3=l3, lat_bw=350.0)                      # F3 먼저 출발
        self.add(t1 + release * 0.6, f3=v3, lat_bw=4000.0)             # 영점이 넓어지며 사라진다
        self.add(t1 + release * 0.6 + 0.001, lat_z1=0.0, lat_z2=0.0)
        self.event(t1 + 0.008, "tongue_release", amp=0.06)
        self.add(t1 + release, f1=v1, f2=v2, tract_gain=1.0, a_c=3.0,
                 bw1=0.0, bw2=0.0, bw3=0.0, c_place=0.9)
        self.t = t1 + release
        return self

    def tap(self, prev_vowel="a", next_vowel="a", closure=0.025, approach=0.030, release=0.035):
        """모음 사이 [ɾ]: 짧은 골. 발성은 이어진다. 접촉/해제/입 바닥 과도음."""
        a1, a2, a3 = VOWELS[prev_vowel]; b1, b2, b3 = VOWELS[next_vowel]
        l1, l2, l3 = TAP_LOCUS
        t0 = self.t
        self.add(t0, f1=a1, f2=a2, f3=a3, tract_gain=1.0, a_c=3.0, p_sub=self.ps)
        tc = t0 + approach
        self.add(tc, f1=l1, f2=l2, f3=l3, tract_gain=0.7, a_c=0.25, c_place=0.87,
                 bw1=70, bw2=150, bw3=240)
        self.event(tc + 0.002, "tongue_contact", amp=0.15, rate=1.1)
        tr = tc + closure
        self.add(tr, f1=l1, f2=l2, f3=l3, tract_gain=0.7, a_c=0.25, bw1=70, bw2=150, bw3=240)
        self.event(tr + 0.003, "tongue_release", amp=0.1)
        self.event(tr + 0.022, "tongue_floor", amp=0.08, rate=0.9)
        self.add(tr + release * 0.7, f3=b3)
        self.add(tr + release, f1=b1, f2=b2, tract_gain=1.0, a_c=3.0,
                 bw1=0.0, bw2=0.0, bw3=0.0, c_place=0.9)
        self.t = tr + release
        return self

    def sibilant(self, dur=0.13, next_vowel="a", a_min=0.10, front_len=1.1):
        """/ㅅ/: 성문을 벌리고(내전↓) 혀가 협착을 만든다. 마찰은 레이놀즈가 정한다."""
        v1, v2, v3 = VOWELS[next_vowel]
        l1, l2, l3 = ALVEOLAR_LOCUS
        t0 = self.t
        close, hold, rel = dur * 0.30, dur * 0.45, dur * 0.25
        self.add(t0, p_sub=self.ps, adduction=0.15, a_c=1.5, c_place=0.91, back_leak=0.08,
                 front_len=front_len, obstacle=0.8, f1=l1, f2=l2, f3=l3, tract_gain=0.9)
        self.add(t0 + close, a_c=a_min, adduction=0.05)
        self.add(t0 + close + hold, a_c=a_min, adduction=0.08)
        self.add(t0 + dur, a_c=3.0, adduction=0.55, obstacle=0.0, front_len=0.0, back_leak=0.3)
        self.add(t0 + dur + 0.04, adduction=0.6, f1=v1, f2=v2, f3=v3, tract_gain=1.0)
        self.t = t0 + dur
        return self

    def nasal(self, dur=0.09, place="n", next_vowel="a"):
        v1, v2, v3 = VOWELS[next_vowel]
        nz = {"m": 1000.0, "n": 1700.0, "ng": 3000.0}[place]
        t0 = self.t
        self.add(t0, p_sub=self.ps, adduction=0.6, velum=1.0, nasal_f=280.0, nasal_z=nz,
                 f1=350.0, f2=1400.0 if place != "m" else 1100.0, f3=2600.0, tract_gain=1.1, a_c=0.05)
        self.add(t0 + dur, velum=1.0, f1=350.0, tract_gain=1.1, a_c=0.05)
        self.add(t0 + dur + 0.05, velum=0.0, f1=v1, f2=v2, f3=v3, tract_gain=1.0, a_c=3.0)
        self.t = t0 + dur
        return self

    def build(self, tail=0.05) -> ControlTrack:
        self.t += tail
        self.add(self.t, p_sub=0.0)
        return track_from_keyframes(self.kf, seconds=self.t, frame_ms=1.0)


def ra(f0=225.0, p_sub=7.5) -> ControlTrack:
    b = Builder(f0, p_sub); b.silence(0.05)
    b.lateral_onset(hold=0.12, release=0.09, next_vowel="a")
    b.vowel("a", 0.30, f0_end=f0 * 0.85)
    b.silence(0.05)
    return b.build()


def ara(f0=225.0, p_sub=7.5) -> ControlTrack:
    b = Builder(f0, p_sub); b.silence(0.05)
    b.vowel("a", 0.18); b.tap("a", "a"); b.vowel("a", 0.25, f0_end=f0 * 0.85)
    b.silence(0.05)
    return b.build()


def sa(f0=225.0, p_sub=7.5) -> ControlTrack:
    b = Builder(f0, p_sub); b.silence(0.05)
    b.sibilant(0.13, "a"); b.vowel("a", 0.30, f0_end=f0 * 0.85)
    b.silence(0.05)
    return b.build()


def na(f0=225.0, p_sub=7.5) -> ControlTrack:
    b = Builder(f0, p_sub); b.silence(0.05)
    b.nasal(0.09, "n", "a"); b.vowel("a", 0.30, f0_end=f0 * 0.85)
    b.silence(0.05)
    return b.build()
