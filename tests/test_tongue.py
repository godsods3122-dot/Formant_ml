"""혀끝 유동유발 진동 검증 — 조음 방식이 '파라미터'가 아니라 '결과'인지 본다.

    PYTHONPATH=src python3 tests/test_tongue.py

핵심 주장: 목표 간극 h0(t) **하나**만 바꾸면 접근음/탄음이 갈린다. 방식을 고르는
if 문이 없다. 접촉 횟수는 우리가 정하지 않고 방정식이 낸다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.dsp.tongue import (TipParams, contact_events, gesture,
                                   simulate_tip)

FS = 24000
DUR = 0.35


def _run(points, **kw):
    n = int(DUR * FS)
    return simulate_tip(gesture(n, points, FS), TipParams(**kw), FS)


def _hold(h0, **kw):
    return _run([(0.0, h0), (DUR, h0)], **kw)


def _dip(depth=-0.04, t0=0.15, rise=0.015, hold=0.03, open_gap=0.30):
    """탄도 제스처: 넓게 열려 있다가 짧게 구개로 눌렀다 놓는다."""
    return _run([(0.0, open_gap), (t0, open_gap), (t0 + rise, depth),
                 (t0 + rise + hold, depth), (t0 + rise + hold + rise, open_gap),
                 (DUR, open_gap)])


def test_approximant_never_touches_the_palate():
    """간극을 넓게 유지하면 접촉이 0 회 — 접근음."""
    out = _hold(0.30)
    assert contact_events(out["contact"]) == 0
    assert float(out["gap"].min()) > 0.0


def test_tap_makes_exactly_one_contact():
    """짧은 탄도 제스처 하나가 접촉 한 번을 만든다 — 탄음.

    '한 번' 을 우리가 지정하지 않았다는 것이 요점이다. 접촉 감쇠가 모자라면
    혀끝이 구개에서 튀어 한 번이 여러 번으로 갈라진다(실제로 겪었다).
    """
    assert contact_events(_dip()["contact"]) == 1


def test_tap_closure_duration_matches_measurement():
    """폐쇄 지속이 실측대 20~50 ms 안에 있어야 한다.

    Cathcart(2012)가 여러 언어에서 '대략 1/24 초'(≈42 ms)로 보고한 값이다.
    """
    ms = float(_dip()["contact"].sum()) / FS * 1000.0
    assert 20.0 <= ms <= 50.0, f"{ms:.1f} ms"


def test_manner_is_monotonic_in_the_single_gesture_parameter():
    """h0 하나를 좁혀 가면 접촉이 없다가 생긴다 — 방식이 연속체 위에 있다."""
    wide = contact_events(_hold(0.30)["contact"])
    narrow = contact_events(_hold(0.08)["contact"])
    assert wide == 0 and narrow > 0, (wide, narrow)


def test_flow_stays_physiological():
    """유량이 사람의 발화 범위(수백~1500 cm^3/s)를 벗어나면 안 된다."""
    for h0 in (0.30, 0.12, 0.05):
        u = float(_hold(h0)["flow"].max())
        assert 0.0 < u < 2000.0, (h0, u)


def test_gap_stays_anatomically_possible():
    """혀끝이 몇 cm 씩 움직이면 안 된다. 점성항이 빠지면 실제로 그렇게 된다."""
    for h0 in (0.30, 0.12, 0.05):
        out = _hold(h0)
        assert float(out["gap"].max()) <= h0 + 1e-6
        assert float(out["gap"].min()) > -0.1


def test_closed_tip_shuts_the_flow_off():
    """접촉 중에는 유량이 0 이어야 한다 (협착이 닫히면 공기가 못 지난다)."""
    out = _dip()
    touching = out["contact"] > 0
    assert bool(touching.any())
    assert float(out["flow"][touching].max()) == 0.0


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
