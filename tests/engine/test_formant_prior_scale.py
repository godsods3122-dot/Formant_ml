"""높은 포먼트의 사전 가중은 **Hz 당 벌점**으로 정해져 있다 (MEASUREMENTS §52.260·263).

사전 항은 `(u − u0) · prior_w` 로 **raw**(로그 범위의 로짓) 거리를 문다. 포먼트 넷은 상한이 같은 로그 축이라
같은 Hz 오차라도 높은 포먼트일수록 raw 거리가 작다. 가중을 그 기울기로 보정하지 않으면 f4 는 사실상 안 묶인다 —
실측으로 F4 가 목표(LPC)에서 평균 864 Hz, 한 판은 1657 Hz 날아갔다.
"""
import math

from formant_ml.engine import fit as F
from formant_ml.engine.control import PARAMS


def _raw(v, spec):
    lo, hi = spec.lo, spec.hi
    x = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if spec.log else (v - lo) / (hi - lo)
    x = min(max(x, 1e-4), 1 - 1e-4)
    return math.log(x / (1 - x))


def _penalty_per_100hz(name, f_hz):
    """그 포먼트가 100 Hz 틀렸을 때의 사전 벌점 `w·d²`."""
    spec = PARAMS[name]
    d = _raw(f_hz + 100.0, spec) - _raw(f_hz, spec)
    return F.PRIOR_W[name] * d * d


# 이 화자의 대표 자리 (yang_female, `wavs/yang_00000000.wav` 0.451~1.293 s 의 LPC 중앙값 언저리)
TYPICAL = {"f1": 700.0, "f2": 1600.0, "f3": 3800.0, "f4": 4950.0}


def test_high_formants_are_not_left_effectively_unconstrained():
    """f3·f4 의 Hz 당 벌점이 f1 의 1/10 아래로 떨어지면 안 된다 — 예전 20·10 은 1/24·1/55 였다."""
    p = {k: _penalty_per_100hz(k, v) for k, v in TYPICAL.items()}
    for k in ("f3", "f4"):
        assert p[k] / p["f1"] > 0.1, (k, p[k] / p["f1"], p)


def test_f1_stays_the_most_tightly_bound_and_the_rest_share_one_rung():
    """f1 은 가장 세게 묶는다(분석이 가장 믿을 만하다). f2·f3·f4 는 **같은 단**이다 — 겨냥한 설계가 그것이다."""
    p = {k: _penalty_per_100hz(k, v) for k, v in TYPICAL.items()}
    assert p["f1"] > 2.0 * max(p["f2"], p["f3"], p["f4"]), p
    rung = [p["f2"], p["f3"], p["f4"]]
    assert max(rung) / min(rung) < 1.2, p


def test_f3_f4_sit_near_the_f2_rung():
    """겨냥한 값은 'Hz 당 벌점을 f2 수준으로' 다 — 0.5~2 배 안에 있어야 한다."""
    p = {k: _penalty_per_100hz(k, v) for k, v in TYPICAL.items()}
    for k in ("f3", "f4"):
        assert 0.5 <= p[k] / p["f2"] <= 2.0, (k, p[k] / p["f2"], p)
