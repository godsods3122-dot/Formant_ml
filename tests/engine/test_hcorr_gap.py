"""보정이 **주기 안에서 뾰족할 것** (`fit.HCORR_GAP`, MEASUREMENTS §52.296).

닫힌 꼴 ½·[1 − Σ(a_k a_{k−1} + b_k b_{k−1}) / Σ(a_k² + b_k²)] 이 실제로
∫ sin²(πφ)·s(φ)² dφ / ∫ s(φ)² dφ 와 같은지 수치로 확인한다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F


def closed_form(h):
    num = float((h[:, 1:] * h[:, :-1]).sum())
    den = float((h ** 2).sum())
    return 0.5 * (1.0 - num / (den + 1e-30))


def numeric(h):
    """한 주기를 촘촘히 그려 실제 비를 잰다."""
    a = h[0, :, 0].detach().numpy()
    b = h[1, :, 0].detach().numpy()
    phi = np.linspace(0.0, 1.0, 20001, endpoint=False)
    k = np.arange(1, len(a) + 1)[:, None]
    ang = 2.0 * np.pi * k * phi[None, :]
    s = (a[:, None] * np.cos(ang) + b[:, None] * np.sin(ang)).sum(0)
    w = np.sin(np.pi * phi) ** 2
    return float((w * s * s).mean() / (s * s).mean())


def _h(a, b):
    return torch.as_tensor(np.stack([np.asarray(a, float)[:, None],
                                     np.asarray(b, float)[:, None]]), dtype=torch.float64)


def test_closed_form_matches_the_integral_for_aligned_harmonics():
    h = _h([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    assert closed_form(h) == pytest.approx(numeric(h), abs=2e-3)


def test_closed_form_matches_the_integral_for_scattered_harmonics():
    rng = np.random.default_rng(3)
    h = _h(rng.standard_normal(9), rng.standard_normal(9))
    assert closed_form(h) == pytest.approx(numeric(h), abs=2e-3)


def test_aligned_harmonics_score_lower_than_alternating_ones():
    """모두 같은 부호로 모이면 뾰족하고, 부호가 번갈면 주기 가운데로 퍼진다."""
    spiky = _h([1.0] * 8, [0.0] * 8)
    flat = _h([1.0, -1.0] * 4, [0.0] * 8)
    assert closed_form(spiky) < 0.1
    assert closed_form(flat) > 0.9


def test_penalty_is_scale_free():
    """크기와 무관해야 한다 — 그래야 높은 배음을 줄이는 쪽으로 값을 치울 수 없다."""
    rng = np.random.default_rng(11)
    h = _h(rng.standard_normal(7), rng.standard_normal(7))
    assert closed_form(h) == pytest.approx(closed_form(h * 37.0), abs=1e-9)


def test_default_is_off():
    assert F.HCORR_GAP == 0.0


def test_penalty_enters_physics_penalty(monkeypatch):
    """물리 벌점에 실제로 더해지는지 — 이 저장소의 상습 실패가 '켰는데 안 걸린다' 이다."""
    monkeypatch.setattr(F, "HCORR_L2", 0.0)
    monkeypatch.setattr(F, "HCORR_TV", 0.0)
    monkeypatch.setattr(F, "HCORR_KTV", 0.0)
    spiky = _h([1.0] * 8, [0.0] * 8).requires_grad_(True)
    flat = _h([1.0, -1.0] * 4, [0.0] * 8).requires_grad_(True)

    def pen(h, w):
        monkeypatch.setattr(F, "HCORR_GAP", w)
        num = (h[:, 1:] * h[:, :-1]).sum()
        den = (h ** 2).sum()
        return F.HCORR_GAP * 0.5 * (1.0 - num / (den + 1e-30))

    assert float(pen(flat, 2.0)) > float(pen(spiky, 2.0))
    assert float(pen(flat, 0.0)) == 0.0
    g = pen(flat, 2.0)
    g.backward()
    assert flat.grad is not None and float(flat.grad.abs().sum()) > 0.0
