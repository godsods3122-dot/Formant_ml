"""혀끝 유동유발 진동 검증 — 조음 방식이 '파라미터'가 아니라 '결과'인지 본다.

    PYTHONPATH=src python3 tests/test_tongue.py

핵심 주장: 목표 간극 h0(t) **하나**만 바꾸면 접근음/탄음이 갈린다. 방식을 고르는
if 문이 없다. 접촉 횟수는 우리가 정하지 않고 방정식이 낸다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from formant_ml.dsp.tongue import (TipParams, contact_events, contact_rate,
                                   gesture, simulate_tip, simulate_tip_chain)

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




# ------------------------------------------------- 다질량 혀끝 (전동음이 나온다)
TRILL_HOLD = 0.03      # 좁은 유지 자세 [cm]
PO_SPEECH = 8000.0     # 발화 중 구강압 (~8 cmH2O)
PO_TAP = 2000.0        # 탄음은 폐쇄가 짧아 압력이 안 쌓인다 (~2 cmH2O)


def _chain(points, nm=5, dur=0.25, **kw):
    n = int(dur * FS)
    return simulate_tip_chain(gesture(n, points, FS), TipParams(**kw), nm, FS,
                              oversample=4)


def test_single_mass_tip_never_self_oscillates():
    """질량 하나짜리 혀끝은 좁게 잡아도 자가진동하지 않는다.

    이것이 전동음이 안 나오던 이유였다. 흡입이 닫는 계는 위상차가 없으면
    기류에서 순 에너지를 못 받는다 (vocalfold.py 서두와 같은 이유).
    """
    out = _chain([(0.0, TRILL_HOLD), (0.25, TRILL_HOLD)], nm=1)
    assert contact_events(out["contact"]) == 0


def test_two_masses_are_not_enough():
    """질량 2 개는 자가진동은 하지만 진동수가 너무 빠르다 — 실측대를 벗어난다.

    측정: n=2 -> 48 Hz, n>=3 -> 28 Hz (전동음 실측 25~35 Hz).
    2 개는 이웃이 하나뿐이라 결합 강성이 파동을 만드는 게 아니라 두 점을
    붙였다 뗐다 할 뿐이고, 파동을 낼 만큼 결합을 풀면 고차 모드로 뛴다.
    """
    r = contact_rate(_chain([(0.0, TRILL_HOLD), (0.25, TRILL_HOLD)],
                            nm=2)["contact"], FS)
    assert r > 40.0, f"{r:.1f} Hz — n=2 가 실측대에 들어오면 이 주석이 틀린 것이다"


def test_three_or_more_masses_hit_the_measured_trill_rate():
    """질량 3 개 이상에서 진동수가 실측대 25~35 Hz 에 들어온다."""
    for nm in (3, 5, 8):
        r = contact_rate(_chain([(0.0, TRILL_HOLD), (0.25, TRILL_HOLD)],
                                nm=nm)["contact"], FS)
        assert 25.0 <= r <= 35.0, f"n={nm}: {r:.1f} Hz"


def test_trill_rate_does_not_drift_with_mass_count():
    """질량 수를 늘려도 진동수가 흐르면 안 된다 (분할 규칙이 맞다는 검사).

    m 과 k 를 함께 1/n 로 줄이지 않으면 F0 가 질량 수에 끌려간다 —
    vocalfold.simulate_stack 주석이 기록한 실패다.
    """
    rates = [contact_rate(_chain([(0.0, TRILL_HOLD), (0.25, TRILL_HOLD)],
                                 nm=nm)["contact"], FS) for nm in (3, 5, 8)]
    assert max(rates) - min(rates) <= 4.0, rates


def test_contact_travels_from_back_to_front():
    """접촉이 후단에서 시작해 전단으로 진행해야 한다.

    이건 우리가 넣은 게 아니라 방정식이 내는 **예측**이고, Cathcart(2012)가
    초음파로 flap 의 back-to-front 운동을 보고한 것과 방향이 같다.
    """
    out = _chain([(0.0, 0.02), (0.30, 0.02)], nm=8, dur=0.30)
    tr, start = out["traj"], int(0.12 * FS)
    firsts = []
    for i in range(tr.shape[1]):
        hit = (tr[start:, i] <= 0).nonzero()
        assert len(hit) > 0, f"마디 {i} 가 닿지 않았다"
        firsts.append(float(hit[0]))
    assert firsts[-1] > firsts[0], firsts          # 전단이 후단보다 늦다
    assert firsts[len(firsts) // 2] >= firsts[0]   # 중간이 뒤집히지 않는다


def test_low_oral_pressure_turns_a_trill_into_a_single_tap():
    """같은 탄도 제스처라도 구강압이 낮으면 접촉이 한 번뿐이다.

    조음 방식을 고르는 스위치가 없다는 것이 요점이다. 탄음은 폐쇄가 20~50 ms
    라 압력이 안 쌓이고(Cathcart 2012), 그래서 자가진동이 성립하지 않는다.
    """
    dip = [(0.0, 0.30), (0.10, 0.30), (0.115, -0.04), (0.145, -0.04),
           (0.16, 0.30), (0.25, 0.30)]
    assert contact_events(_chain(dip, po=PO_TAP)["contact"]) == 1
    assert contact_events(_chain(dip, po=PO_SPEECH)["contact"]) > 1


def test_tap_closure_duration_matches_measurement_in_the_chain():
    """다질량 모델에서도 폐쇄가 실측대 20~50 ms 안에 있어야 한다."""
    dip = [(0.0, 0.30), (0.10, 0.30), (0.115, -0.04), (0.145, -0.04),
           (0.16, 0.30), (0.25, 0.30)]
    ms = float(_chain(dip, po=PO_TAP)["contact"].sum()) / FS * 1000.0
    assert 20.0 <= ms <= 50.0, f"{ms:.1f} ms"


def test_trill_needs_pressure_to_sustain():
    """압력이 모자라면 전동음이 성립하지 않는다 (Solé 2002 의 공기역학적 요구)."""
    hold = [(0.0, TRILL_HOLD), (0.25, TRILL_HOLD)]
    assert contact_events(_chain(hold, po=PO_SPEECH)["contact"]) > 3
    assert contact_events(_chain(hold, po=500.0)["contact"]) == 0


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
