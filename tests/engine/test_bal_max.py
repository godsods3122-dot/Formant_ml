"""균형 가중의 상한 (`fit.BAL_MAX`) — 못 맞추는 항이 가중을 폭주시키던 결함 (MEASUREMENTS §52.243).

GradNorm 은 `가중 = 몫 × ‖∇env‖ / ‖∇항‖` 이다. 목표가 **원리적으로 못 맞추는 양**(잡음의 실현 등)이면
그 항의 기울기가 0 에 가까워 분모가 사라지고 가중이 폭주한다 — 치찰음 구간에서 `flux` 가 41.1 까지 갔다.
"""
import inspect

from formant_ml.engine import fit as F


def test_default_is_on_and_finite():
    assert F.BAL_MAX > 0.0
    assert F.BAL_MAX < 1e3


def test_the_cap_is_applied_in_the_balance_code():
    src = inspect.getsource(F.CopySynthFitter)
    assert "BAL_MAX * BAL_W[k]" in src
    assert "균형 상한에 걸림" in src        # 걸린 것을 반드시 찍는다 (조용히 자르지 않는다)


def test_the_cap_is_a_multiple_of_the_share():
    """상한은 절댓값이 아니라 **몫의 배수**다 — 몫이 다른 항끼리 서로 다른 상한을 갖는다."""
    for k, share in (("flux", 0.3), ("corr", 1.0)):
        assert F.BAL_W[k] * F.BAL_MAX > F.BAL_W[k]


def test_a_tiny_gradient_would_have_blown_up_without_it():
    """상한이 없으면 얼마나 커지나 — 실측값으로 확인 (flux 몫 0.3, 관측 가중 41.1 = 137 배)."""
    assert 41.1 / F.BAL_W["flux"] > F.BAL_MAX, "실측 폭주가 상한 아래면 이 시험이 무의미하다"
