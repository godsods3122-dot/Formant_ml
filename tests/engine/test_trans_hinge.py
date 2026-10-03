"""튐 벌점(`fit.transient_loss`)의 경첩 — 문턱 아래는 기울기가 정확히 0 이다 (MEASUREMENTS §52.472)."""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.fit import CopySynthFitter

FS = 48000


def _voice(dur=0.4, f0=180.0, seed=0):
    from scipy.signal import lfilter
    g = np.random.default_rng(seed)
    n = int(dur * FS)
    x = np.zeros(n)
    x[(np.arange(0, dur, 1 / f0) * FS).astype(int)] = 1.0
    for f, bw in ((700, 90), (1800, 120), (3000, 200)):
        r = np.exp(-np.pi * bw / FS)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r], x)
    x = x / np.abs(x).max() * 0.3
    return x + 3e-3 * g.standard_normal(n)


class _Stand:
    """`transient_loss` 가 쓰는 것만 가진 대역."""

    def __init__(self, target):
        self.fs = FS
        self.target = torch.as_tensor(target, dtype=torch.float64).unsqueeze(0)
        self._trans_env = CopySynthFitter._trans_env.__get__(self)
        self._soft_over = CopySynthFitter._soft_over


def _loss_grad(target, y, leak, monkeypatch):
    monkeypatch.setattr(F, "TRANS_LEAK", leak)
    s = _Stand(target)
    yy = torch.as_tensor(y, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    l = CopySynthFitter.transient_loss(s, yy)
    (g,) = torch.autograd.grad(l, yy, allow_unused=True)
    return float(l), (0.0 if g is None else float(g.abs().max()))


def test_below_threshold_has_exactly_zero_gradient(monkeypatch):
    x = _voice()
    y = 0.9 * x                                     # 어디서나 목표보다 작다 — 문턱 아래
    l, g = _loss_grad(x, y, False, monkeypatch)
    assert l == 0.0 and g == 0.0


def test_old_softplus_hinge_leaks_below_threshold(monkeypatch):
    """되살리기용 옛 경첩은 같은 입력에서 기울기가 샌다 — 이것이 §52.472 의 결함이었다."""
    x = _voice()
    l, g = _loss_grad(x, 0.9 * x, True, monkeypatch)
    assert l > 0.0 and g > 0.0


def test_a_click_absent_from_the_target_is_penalised(monkeypatch):
    x = _voice()
    y = x.copy()
    i = int(0.2 * FS)
    y[i:i + 24] += 0.3 * np.hanning(24) * np.sign(np.random.default_rng(3).standard_normal(24))
    l, g = _loss_grad(x, y, False, monkeypatch)
    assert l > 0.0 and g > 0.0


def test_hinge_is_c2_at_the_threshold():
    """z³/(z²+k²) 는 0 에서 값·1 차·2 차 도함수가 모두 0 — 왼쪽(0)과 이어진다."""
    k = F.TRANS_KNEE
    z = torch.tensor([1e-4], dtype=torch.float64, requires_grad=True)
    f = z ** 3 / (z ** 2 + k * k)
    (d1,) = torch.autograd.grad(f, z, create_graph=True)
    (d2,) = torch.autograd.grad(d1, z)
    assert abs(float(f)) < 1e-11 and abs(float(d1)) < 1e-7 and abs(float(d2)) < 1e-3


# ------------------------------------------------------------ 치찰 사전 가장자리 (같은 §52.472 의 두 번째 결함)
def test_sib_edge_weights_ramp_both_ends_of_each_run():
    m = np.zeros(100, bool)
    m[10:60] = True                                   # 50 틀 구간
    m[80:84] = True                                   # 4 틀 — 경사가 겹친다
    w = F.sib_edge_weights(m, 10.0)
    assert np.all(w[~m] == 0.0)
    assert w[10] < 0.05 and w[59] < 0.05              # 양 끝이 가장 가볍다
    assert np.allclose(w[20:50], 1.0)                 # 가장자리에서 10 틀 들어가면 1
    assert np.all(np.diff(w[10:21]) > 0) and np.all(np.diff(w[49:60]) < 0)
    assert w[80:84].max() < 0.2                        # 짧은 구간은 가운데도 1 에 못 닿는다
    assert np.array_equal(F.sib_edge_weights(m, 0.0), m.astype(float))   # 0 이면 예전 그대로
