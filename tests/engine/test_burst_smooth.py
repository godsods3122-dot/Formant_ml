"""파열 자리의 조음 궤적 이어 붙이기 (`analyze.smooth_over_bursts`, MEASUREMENTS §52.329)."""
import numpy as np
import pytest

from formant_ml.engine.analyze import BURST_SMOOTH_PARAMS, smooth_over_bursts
from formant_ml.engine.control import INDEX, PARAMS, ControlTrack


def _track(n=200, burst=100, spike=900.0):
    vals = np.tile(np.array([p.default for p in PARAMS.values()], float), (n, 1))
    vals[:, INDEX["f2"]] = 1500.0
    vals[burst - 2:burst + 3, INDEX["f2"]] = spike          # 파열이 만든 급발진
    vals[:, INDEX["f0_target"]] = 300.0
    vals[burst, INDEX["f0_target"]] = 420.0
    tr = ControlTrack(vals, 1.0)
    tr.bursts = np.array([burst], dtype=int)
    return tr


def test_off_when_half_is_zero():
    tr = _track()
    assert smooth_over_bursts(tr, 0.0) == 0
    assert tr.values[100, INDEX["f2"]] == pytest.approx(900.0)


def test_no_bursts_means_no_change():
    tr = _track()
    tr.bursts = np.zeros(0, dtype=int)
    assert smooth_over_bursts(tr, 12.0) == 0


def test_spike_is_replaced_by_a_straight_line():
    tr = _track()
    n = smooth_over_bursts(tr, 12.0)
    assert n == 25
    f2 = tr.values[:, INDEX["f2"]]
    assert np.allclose(f2[88:113], 1500.0)          # 양끝이 같으니 직선이 곧 상수
    assert f2[80] == pytest.approx(1500.0)


def test_pitch_is_smoothed_too():
    tr = _track()
    smooth_over_bursts(tr, 12.0)
    assert tr.values[100, INDEX["f0_target"]] == pytest.approx(300.0)


def test_endpoints_are_preserved_and_the_line_connects_them():
    tr = _track()
    tr.values[88, INDEX["f2"]] = 1200.0
    tr.values[112, INDEX["f2"]] = 1800.0
    smooth_over_bursts(tr, 12.0)
    f2 = tr.values[:, INDEX["f2"]]
    assert f2[88] == pytest.approx(1200.0)
    assert f2[112] == pytest.approx(1800.0)
    assert f2[100] == pytest.approx(1500.0, abs=1.0)


def test_only_the_named_controls_move():
    tr = _track()
    before = tr.values[:, INDEX["tract_gain"]].copy()
    smooth_over_bursts(tr, 12.0)
    assert np.allclose(tr.values[:, INDEX["tract_gain"]], before)
    assert "f2" in BURST_SMOOTH_PARAMS and "tract_gain" not in BURST_SMOOTH_PARAMS


def test_constriction_controls_are_smoothed_too():
    """협착을 놔두면 레이놀즈가 터져도 성도에서 눌린다 (§52.338)."""
    for k in ("a_c", "obstacle", "front_len"):
        assert k in BURST_SMOOTH_PARAMS


def test_constriction_spike_is_flattened():
    tr = _track()
    tr.values[:, INDEX["a_c"]] = 0.4
    tr.values[98:103, INDEX["a_c"]] = 0.05          # 파열이 만든 조임
    smooth_over_bursts(tr, 12.0)
    assert tr.values[100, INDEX["a_c"]] == pytest.approx(0.4, abs=1e-6)
