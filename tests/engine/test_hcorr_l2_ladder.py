"""보정 L2 의 **배음 사다리** (`fit.HCORR_L2_KNEE/SLOPE`, MEASUREMENTS §52.297)."""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F


class _Fit:
    def __init__(self, k=8, c=3):
        self.hcorr = torch.ones((2, k, c), dtype=torch.float64)


def _kw(k=8, knee=0, slope=0.0):
    f = _Fit(k)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(F, "HCORR_L2_KNEE", knee)
        m.setattr(F, "HCORR_L2_SLOPE", slope)
        return F.CopySynthFitter._hcorr_kw(f)


def test_off_by_default():
    assert F.HCORR_L2_SLOPE == 0.0
    assert F.HCORR_L2_KNEE == 0


def test_flat_when_slope_is_zero():
    assert _kw(slope=0.0) == 1.0


def test_weight_grows_linearly_above_the_knee():
    w = _kw(k=8, knee=4, slope=0.5).reshape(-1).numpy()
    assert np.allclose(w[:4], 1.0)                       # H1~H4 (k-knee ≤ 0) 는 그대로
    assert np.allclose(w[4:], [1.5, 2.0, 2.5, 3.0])      # H5 부터 0.5 씩


def test_knee_zero_charges_every_harmonic():
    w = _kw(k=4, knee=0, slope=1.0).reshape(-1).numpy()
    assert np.allclose(w, [2.0, 3.0, 4.0, 5.0])


def test_shape_broadcasts_over_cycles():
    w = _kw(k=6, knee=2, slope=0.25)
    assert tuple(w.shape) == (1, 6, 1)
    assert (torch.ones((2, 6, 5), dtype=torch.float64) * w).shape == (2, 6, 5)


def test_high_harmonics_cost_more_in_the_penalty():
    """같은 크기라도 높은 배음에 놓으면 벌점이 커야 한다 — 싼 길을 막는 것이 목적이다."""
    lo = torch.zeros((2, 8, 1), dtype=torch.float64)
    hi = torch.zeros((2, 8, 1), dtype=torch.float64)
    lo[0, 0, 0] = 1.0
    hi[0, 7, 0] = 1.0
    with pytest.MonkeyPatch.context() as m:
        m.setattr(F, "HCORR_L2_KNEE", 2)
        m.setattr(F, "HCORR_L2_SLOPE", 0.5)
        f = _Fit()
        f.hcorr = hi
        w = F.CopySynthFitter._hcorr_kw(f)
        assert float((hi ** 2 * w).mean()) > 3.0 * float((lo ** 2 * w).mean())
