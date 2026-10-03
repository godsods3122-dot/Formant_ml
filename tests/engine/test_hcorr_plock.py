"""낮은 배음의 **위상만** 묶는 항 (`fit.HCORR_PLOCK`, MEASUREMENTS §52.334)."""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F


def _pen(h, w=1.0, kk=None):
    a_, b_ = h[0][:kk], h[1][:kk]
    mag = torch.sqrt(a_ * a_ + b_ * b_ + 1e-20)
    res = torch.sqrt(a_.sum(-1) ** 2 + b_.sum(-1) ** 2 + 1e-20)
    return float(w * (1.0 - res / mag.sum(-1).clamp_min(1e-12)).mean())


def _h(ang, mag=None):
    """(2, K, C) — 배음 4 개, 주기 len(ang) 개. ang[c] 는 주기 c 의 위상."""
    ang = np.asarray(ang, float)
    mag = np.ones_like(ang) if mag is None else np.asarray(mag, float)
    a = np.tile(mag * np.cos(ang), (4, 1))
    b = np.tile(mag * np.sin(ang), (4, 1))
    return torch.as_tensor(np.stack([a, b]), dtype=torch.float64)


def test_default_is_off():
    assert F.HCORR_PLOCK == 0.0
    assert F.HCORR_PLOCK_K == 12


def test_same_phase_costs_nothing():
    assert _pen(_h([0.3] * 20)) == pytest.approx(0.0, abs=1e-6)


def test_scattered_phase_costs_nearly_one():
    rng = np.random.default_rng(0)
    assert _pen(_h(rng.uniform(0, 2 * np.pi, 400))) > 0.9


def test_magnitude_is_free():
    """크기가 주기마다 크게 달라도 위상이 같으면 값이 0 이다 — 크기 보정을 막지 않는다."""
    rng = np.random.default_rng(1)
    mag = rng.uniform(0.1, 5.0, 40)
    assert _pen(_h([1.1] * 40, mag)) == pytest.approx(0.0, abs=1e-6)


def test_partial_scatter_is_between():
    rng = np.random.default_rng(2)
    tight = _pen(_h(rng.normal(0.0, 0.2, 200)))
    loose = _pen(_h(rng.normal(0.0, 1.2, 200)))
    assert 0.0 < tight < loose < 1.0


def test_only_the_low_harmonics_are_locked():
    """상한 위의 배음은 흩어져도 값에 안 들어간다."""
    rng = np.random.default_rng(3)
    n = 200
    a = np.stack([np.cos(np.full(n, 0.4))] * 2 + [np.cos(rng.uniform(0, 6.28, n))] * 2)
    b = np.stack([np.sin(np.full(n, 0.4))] * 2 + [np.sin(rng.uniform(0, 6.28, n))] * 2)
    h = torch.as_tensor(np.stack([a, b]), dtype=torch.float64)
    assert _pen(h, kk=2) == pytest.approx(0.0, abs=1e-6)
    assert _pen(h, kk=4) > 0.4


def test_gradient_flows():
    rng = np.random.default_rng(4)
    h = _h(rng.uniform(0, 2 * np.pi, 50)).clone().requires_grad_(True)
    a_, b_ = h[0], h[1]
    mag = torch.sqrt(a_ * a_ + b_ * b_ + 1e-20)
    res = torch.sqrt(a_.sum(-1) ** 2 + b_.sum(-1) ** 2 + 1e-20)
    (1.0 - res / mag.sum(-1).clamp_min(1e-12)).mean().backward()
    assert h.grad is not None and float(h.grad.abs().sum()) > 0.0
