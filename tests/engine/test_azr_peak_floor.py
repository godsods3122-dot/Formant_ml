"""보조 영점 봉우리형(`tract.AZR_PEAK`)과 F5~F8 폭 하한 분리(`tract.HF_FORMANT_BW_FLOOR`) (MEASUREMENTS §52.245)."""
import numpy as np
import torch

from formant_ml.engine import tract as T
from formant_ml.engine.tviir import notch_coeffs, peaking_coeffs

FS = 48000.0


def _g(coef, mix, f):
    b0, b1, b2, a1, a2 = (float(v) for v in coef)
    z = np.exp(-1j * 2 * np.pi * f / FS)
    H = (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
    return 20 * np.log10(abs(1 + mix * (H - 1)))


def _c(fn, fz, *args):
    return fn(torch.tensor(fz, dtype=torch.float64), fz * T.AZR_WFRAC, FS, *args)


def test_old_notch_moves_the_far_band_with_mix():
    """결함의 재현: mix 0.2 에서 15 kHz 가 0.3 dB 넘게 움직인다."""
    assert _g(_c(notch_coeffs, 2600.0, T.AZR_RATIO), 0.2, 15000.0) > 0.3


def test_peaking_cut_leaves_the_far_band_alone():
    for fz in (1200.0, 2600.0, 4300.0):
        for mix in (-0.2, 0.1, 0.4, 1.0):
            for f in (13000.0, 15000.0, 18000.0):
                assert abs(_g(_c(peaking_coeffs, fz, 1.0 / T.AZR_RATIO), mix, f)) < 0.15, (fz, mix, f)


def test_peaking_cut_keeps_the_local_depth():
    for fz in (1200.0, 2600.0):
        old = _g(_c(notch_coeffs, fz, T.AZR_RATIO), 0.4, fz)
        new = _g(_c(peaking_coeffs, fz, 1.0 / T.AZR_RATIO), 0.4, fz)
        assert abs(old - new) < 1.0, (fz, old, new)


def test_defaults_are_unchanged():
    assert T.AZR_PEAK is False
    assert T.HF_FORMANT_BW_FLOOR is None
    assert abs(float(T._extra_mult(torch.tensor(-5.0))) - T.EXTRA_BW_FLOOR) < 1e-6


def test_floor_argument_moves_only_the_floor():
    lo = float(T._extra_mult(torch.tensor(-5.0), 0.8))
    assert abs(lo - 0.8) < 1e-6
    top_default = float(T._extra_mult(torch.tensor(5.0)))
    top_low = float(T._extra_mult(torch.tensor(5.0), 0.8))
    assert top_low <= T.EXTRA_BW_CEIL + 1e-6 and top_default <= T.EXTRA_BW_CEIL + 1e-6
    xs = torch.linspace(-4, 4, 41)
    v = T._extra_mult(xs, 0.8)
    assert bool((v[1:] >= v[:-1] - 1e-9).all())
