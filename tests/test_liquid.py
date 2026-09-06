"""유음 궤적 — `docs/LIQUID.md` 의 측정과 코드가 어긋나지 않는지.

    PYTHONPATH=src python3 tests/test_liquid.py

여기서 지키는 것은 **측정으로 정한 값들**이다. 상수를 바꾸고 싶으면 문서의
해당 절을 먼저 고쳐라 — 근거 없이 움직이면 이 시험이 잡는다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from formant_ml.liquid import (F1_TARGET, F2_FLAT, F2_RAISED, LiquidGesture,
                               PRESETS, constriction, liquid_track)

SR = 24000
HOP = 240


def _vowel(t, k=4):
    """이 화자의 /아/ (LIQUID.md §2.1)."""
    return np.tile([700.0, 1200.0, 2450.0, 3300.0][:k], (t, 1))


def test_gesture_durations_match_the_recording():
    """탄음 40 ms, 설측음 255 ms — 실측 그대로여야 한다 (LIQUID.md §2.2).

    두 토큰이 독립적으로 40.0 ms 로 나왔고 문헌의 ~42 ms 와도 맞는다.
    """
    assert LiquidGesture.preset("tap").total_ms == 40.0
    assert LiquidGesture.preset("lateral").total_ms == 255.0


def test_tap_is_deeper_than_the_lateral():
    """탄음이 설측음보다 **깊다** (LIQUID.md §2.2).

    처음에 반대로 결론냈다가 고쳤다 — 20 ms 분석창이 40 ms 짜리 탄음 골을
    뭉갠 것이었다. 10 ms 창(F0 108 Hz 의 한 주기 9.3 ms 보다 길어야 한다)으로
    재면 탄음 −16.6 / −10.9, 설측음 −8.0 이다. 물리적으로도 탄음이 완전
    폐쇄에 가깝다.
    """
    tap = LiquidGesture.preset("tap")
    lat = LiquidGesture.preset("lateral")
    assert tap.depth_db < lat.depth_db - 3.0, (
        f"탄음 {tap.depth_db} vs 설측음 {lat.depth_db}")


def test_body_raise_moves_f2_and_nothing_else():
    """설체 상승은 F2 만 움직인다 — Lee(2015) 의 범주적 축 (LIQUID.md §1.1).

    설측음은 F2 가 올라가고(1190 -> 1445) 탄음은 거의 안 움직인다.
    """
    flat = LiquidGesture(body_raise=0.0)
    raised = LiquidGesture(body_raise=1.0)
    assert abs(flat.f2 - F2_FLAT) < 1e-6
    assert abs(raised.f2 - F2_RAISED) < 1e-6
    assert flat.f1 == raised.f1 == F1_TARGET
    assert flat.f3 == raised.f3


def test_f3_leads_the_constriction():
    """F3 가 조음보다 **먼저** 움직인다 (LIQUID.md §2.3, Ying 2026).

    실측 5 ms(아라#1) / 25 ms(아라#2), Ying 은 15~30 ms. F1·F2 는 반대로
    뒤진다 — 그래서 F3 만 앞세운다. 궤적을 하나로 묶으면 이 구조가 사라진다.
    """
    g = LiquidGesture.preset("lateral")
    t = 200
    F, gain = liquid_track(_vowel(t), g, t, HOP, SR)
    # F1 과 F3 가 목표에 **도달하는 시점**을 비교한다
    def reach(col, target, vowel):
        x = (F[:, col] - vowel) / (target - vowel)
        return int(np.argmax(x > 0.5))
    i1 = reach(0, g.f1, 700.0)
    i3 = reach(2, g.f3, 2450.0)
    assert i3 < i1, f"F3 가 F1 보다 늦게 움직인다 (F3 {i3}, F1 {i1} 프레임)"
    lead_ms = (i1 - i3) * HOP / SR * 1000.0
    assert 5.0 <= lead_ms <= 40.0, f"선행 {lead_ms:.0f} ms 가 실측 범위 밖"


def test_constriction_function_is_smooth():
    """협착 **함수**가 각지면 안 된다 (충분히 촘촘히 재서 확인).

    각진 사다리꼴은 기울기가 꺾이는 지점이 포먼트 궤적의 2 차 미분에 임펄스로
    남고, 그게 소리에서 '툭' 하고 들린다. smoothstep 으로 들어가고 나온다.
    """
    fine = 24                                   # 1 ms 간격
    for name in PRESETS:
        g = LiquidGesture.preset(name)
        c = constriction(4000, fine, SR, g, center_s=2.0)
        d2 = np.abs(np.diff(c, 2))
        assert d2.max() < 0.02, f"{name}: 2 차 미분 최대 {d2.max():.4f} — 꺾임"
        assert 0.0 <= c.min() and c.max() <= 1.0 + 1e-9


def test_liquid_transitions_are_undersampled_at_the_default_frame_rate():
    """**유음 전이는 기본 프레임률에서 표본이 모자란다** — 알려진 한계다.

    협착으로 들어가고 나오는 시간이 20 ms 인데(실측: 탄음 전체가 40 ms 이고
    유지가 없다, LIQUID.md §2.2) 기본 프레임이 10 ms 라 전이 하나가 **2 점**
    이다. 함수가 아무리 매끄러워도 샘플된 궤적은 꺾이고(2 차 미분 0.6),
    이건 탄음만이 아니라 **모든 유음 제스처**에 해당한다.

    지금은 고치지 않고 기록한다 — 프레임률을 올리면 합성 경로 전체가 같이
    움직이고 그 영향을 따로 재야 한다. 유음의 전이를 다듬을 때 여기부터 보라.
    """
    coarse = {}
    for name in PRESETS:
        g = LiquidGesture.preset(name)
        c = constriction(400, HOP, SR, g, center_s=2.0)
        coarse[name] = float(np.abs(np.diff(c, 2)).max())
    assert all(v > 0.4 for v in coarse.values()), (
        f"전이가 더 이상 표본 부족이 아니다 — 프레임률이나 TRANSIT_MS 를 "
        f"바꿨다면 이 시험과 docs/LIQUID.md 를 같이 고쳐라: {coarse}")
    # **표본화의 문제라는 증거**: 프레임을 촘촘히 하면 줄어든다.
    # 함수에 진짜 꺾임이 있으면 안 줄어든다.
    g = LiquidGesture.preset("tap")
    d = [float(np.abs(np.diff(constriction(400 * k, HOP // k, SR, g, 2.0), 2)).max())
         for k in (1, 2, 4, 8)]
    for a, b in zip(d, d[1:]):
        assert b < a * 0.5, f"촘촘히 해도 안 줄어든다 (진짜 꺾임): {d}"


def test_tap_never_fully_reaches_its_f3_target():
    """탄음에서는 F3 가 목표에 **못 닿는다** — 선행 때문이다.

    실측에서도 F3 봉우리가 세기 최소보다 먼저 오고, 최소점에서는 이미
    되돌아오는 중이다(2430 -> 2339, §2.3). 이건 손으로 넣은 것이 아니라
    선행 오프셋에서 저절로 나와야 한다.
    """
    g = LiquidGesture.preset("tap")
    t = 200
    F, gain = liquid_track(_vowel(t), g, t, HOP, SR)
    i = int(np.argmin(gain))
    assert F[i, 2] > g.f3 + 30.0, (
        f"세기 최소점의 F3 {F[i, 2]:.0f} 가 목표 {g.f3:.0f} 에 닿아 버렸다")


def test_presets_are_one_code_path():
    """설측음과 탄음이 **같은 코드 경로**여야 한다 (LIQUID.md §4-2).

    지운 구현은 자세마다 별도 면적함수와 영점을 들고 있었다. 여기서는
    `hold_ms`/`depth_db`/`body_raise` 세 숫자만 다르다.
    """
    t = 200
    out = {}
    for name in PRESETS:
        F, gain = liquid_track(_vowel(t), LiquidGesture.preset(name), t, HOP, SR)
        out[name] = (F, gain)
    for name, (F, gain) in out.items():
        assert np.isfinite(F).all() and np.isfinite(gain).all()
        assert abs(F[0, 0] - 700.0) < 1.0, f"{name}: 가장자리가 모음이 아니다"
    # 유지 시간이 길수록 골이 넓다
    width = {n: float((g < 0.9).sum()) for n, (_, g) in out.items()}
    assert width["tap"] < width["onset"] < width["lateral"], width


def test_no_antiresonance_is_used():
    """영점을 쓰지 않는다 (LIQUID.md §2.4 — 증거가 없다).

    지운 구현은 유음마다 영점 쌍을 들고 있었고, 그 위에 맞춘 극이 부서졌다.
    영점을 다시 넣으려면 `scripts/measure_liquid.py --zeros` 로 **먼저 증거를
    만든 뒤**여야 한다.
    """
    import formant_ml.liquid as m
    src = open(m.__file__, encoding="utf-8").read()
    for word in ("antiformant", "antiresonator", "zero_freq", "영점 주파수"):
        assert word not in src, f"영점을 쓰고 있다: {word}"


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
