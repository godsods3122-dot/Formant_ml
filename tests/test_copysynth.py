"""복사합성의 최소 조건 — 되합성이 원본을 따라가는가.

    PYTHONPATH=src python3 tests/test_copysynth.py

합성한 신호를 **원본과 대조**하는 유일한 검사다. 손으로 작곡한 음절에는 정답이
없어서 "지표는 맞는데 사람 소리가 아닌" 상태를 못 잡는다. 여기서는 정답이 있다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from formant_ml.analysis.track import track_formants

FS = 24000


def _synthetic_vowel(f0=120.0, formants=(700, 1200, 2500, 3500),
                     bws=(80, 90, 120, 160), dur=0.4, sr=FS):
    """알려진 포먼트를 가진 합성 모음 — 추적기의 정답을 우리가 안다."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    src = np.zeros(n)
    for k in range(1, int(sr / 2 / f0)):
        src += np.cos(2 * np.pi * k * f0 * t) / k
    y = src
    for f, b in zip(formants, bws):
        r = np.exp(-np.pi * b / sr)
        th = 2 * np.pi * f / sr
        a1, a2 = -2 * r * np.cos(th), r * r
        out = np.zeros(n)
        g = 1 + a1 + a2
        for i in range(2, n):
            out[i] = g * y[i] - a1 * out[i - 1] - a2 * out[i - 2]
        y = out
    return y / np.abs(y).max()


def test_tracker_finds_known_formants():
    """알려진 포먼트를 넣으면 그대로 찾아야 한다."""
    y = _synthetic_vowel()
    F, _ = track_formants(y, FS, n=6)
    got = np.median(F[5:-5], axis=0)
    for i, want in enumerate((700, 1200, 2500, 3500)):
        assert abs(got[i] - want) / want < 0.08, f"F{i+1} {got[i]:.0f} vs {want}"


def test_tracker_never_emits_zero_formants():
    """못 찾은 슬롯을 0 으로 두면 안 된다.

    합성기가 f_min 으로 클램프해서 저역에 가짜 공명기를 만들고, 하나당
    -12 dB/oct 씩 감쇠가 붙는다. 실제로 빈 슬롯 4 개가 150 Hz 유령 극이 되어
    고역을 40 dB 죽였고, 되합성 오차가 33 dB 였다 (고친 뒤 5 dB).
    """
    y = _synthetic_vowel()
    F, B = track_formants(y, FS, n=12)      # 실제보다 훨씬 많이 요구한다
    assert float(F.min()) > 200.0, f"최소 포먼트 {F.min():.0f} Hz"
    assert float(B.min()) > 0.0
    # 남는 슬롯은 아주 높고 아주 넓어야 한다(무해)
    assert float(F[:, -1].min()) > 5000.0


def test_tracked_formants_are_ordered_and_continuous():
    """포먼트는 교차하지 않고, 프레임 사이에서 튀지 않아야 한다."""
    y = _synthetic_vowel()
    F, _ = track_formants(y, FS, n=6)
    assert bool((np.diff(F, axis=1) >= 0).all()), "포먼트 순서가 뒤집혔다"
    jump = np.abs(np.diff(F[5:-5, :4], axis=0)).max()
    assert jump < 400.0, f"프레임 간 최대 도약 {jump:.0f} Hz"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
