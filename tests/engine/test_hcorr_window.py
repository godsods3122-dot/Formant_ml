"""보정을 **성문 폐쇄 둘레에만** 놓는 창 (`fit.HCORR_WIN`, MEASUREMENTS §52.299)."""
import math

import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F


def window(frac, half):
    d = np.minimum(frac, 1.0 - frac)
    return 0.5 + 0.5 * np.cos(np.pi * np.clip(d / half, None, 1.0))


class _Fit:
    """`_hcorr_signal` 이 쓰는 것만 갖춘 최소 대역물."""
    def __init__(self, k, n_cycles, n, phi):
        self.hcorr = torch.zeros((2, k, n_cycles), dtype=torch.float64)
        self.hcorr[0, :, :] = 1.0                       # 모든 배음을 같은 위상으로
        self._hc_c0 = 0
        self._hc_scale = 1.0
        self._pulse_phase = torch.as_tensor(phi, dtype=torch.float64).reshape(1, -1)
        self.pulse_phi0 = torch.zeros(1, dtype=torch.float64)


def _sig(half, k=6, cycles=4, per=200):
    n = cycles * per
    phi = 2.0 * math.pi * np.arange(n) / per
    f = _Fit(k, cycles + 1, n, phi)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(F, "HCORR_WIN", half)
        return F.CopySynthFitter._hcorr_signal(f)[0].numpy(), (np.arange(n) % per) / per


def test_default_is_off():
    assert F.HCORR_WIN == 0.0


def test_off_leaves_the_signal_untouched():
    a, _ = _sig(0.0)
    b, _ = _sig(0.0)
    assert np.allclose(a, b)
    assert np.abs(a).max() > 0.0


def test_window_zeroes_the_middle_of_the_cycle():
    y, frac = _sig(0.25)
    mid = (frac > 0.30) & (frac < 0.70)
    assert np.abs(y[mid]).max() < 1e-9


def test_closure_is_left_alone():
    """창의 값이 폐쇄에서 1 이라 그 자리의 보정은 그대로다."""
    off, _ = _sig(0.0)
    on, frac = _sig(0.25)
    at = frac < 0.002
    assert np.allclose(on[at], off[at], atol=1e-9)


def test_window_matches_the_raised_cosine_it_claims():
    off, frac = _sig(0.0)
    on, _ = _sig(0.25)
    w = window(frac, 0.25)
    keep = np.abs(off) > 1e-6
    assert np.allclose(on[keep], off[keep] * w[keep], atol=1e-9)


def test_narrower_window_keeps_less_energy():
    wide, _ = _sig(0.4)
    narrow, _ = _sig(0.1)
    assert (narrow ** 2).sum() < (wide ** 2).sum()


def test_gradients_flow_through_the_window():
    n, per, k = 400, 200, 5
    phi = 2.0 * math.pi * np.arange(n) / per
    f = _Fit(k, 3, n, phi)
    f.hcorr = f.hcorr.clone().requires_grad_(True)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(F, "HCORR_WIN", 0.25)
        F.CopySynthFitter._hcorr_signal(f).pow(2).sum().backward()
    assert f.hcorr.grad is not None
    assert float(f.hcorr.grad.abs().sum()) > 0.0


def _sig_k(half, win_k, k=8, cycles=4, per=200):
    n = cycles * per
    phi = 2.0 * math.pi * np.arange(n) / per
    f = _Fit(k, cycles + 1, n, phi)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(F, "HCORR_WIN", half)
        m.setattr(F, "HCORR_WIN_K", win_k)
        return F.CopySynthFitter._hcorr_signal(f)[0].numpy(), (np.arange(n) % per) / per


def test_win_k_default_is_zero():
    assert F.HCORR_WIN_K == 0


def test_win_k_leaves_the_low_harmonics_in_the_middle_of_the_cycle():
    """H1~H4 는 주기 한가운데에도 남아야 한다 — F5~F8 대역이 거기서 할 일이 있다."""
    y, frac = _sig_k(0.25, 4)
    mid = (frac > 0.35) & (frac < 0.65)
    assert np.abs(y[mid]).max() > 1e-6


def test_win_k_zero_still_zeroes_the_middle():
    y, frac = _sig_k(0.25, 0)
    mid = (frac > 0.35) & (frac < 0.65)
    assert np.abs(y[mid]).max() < 1e-9


def test_win_k_above_all_harmonics_is_the_unwindowed_signal():
    on, _ = _sig_k(0.25, 99)
    off, _ = _sig_k(0.0, 0)
    assert np.allclose(on, off, atol=1e-9)


def test_win_k_middle_holds_only_the_low_part():
    """한가운데 남은 것은 H1~Hk 의 합과 같아야 한다."""
    k_keep = 3
    y, frac = _sig_k(0.25, k_keep, k=8)
    per, n = 200, 800
    ph = (np.arange(n) % per) / per
    lo = sum(np.cos(2 * np.pi * (j + 1) * ph) for j in range(k_keep))
    mid = (frac > 0.40) & (frac < 0.60)
    assert np.allclose(y[mid], lo[mid], atol=1e-9)
