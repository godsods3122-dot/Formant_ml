"""Linearized modal summary of the 3D vocal-fold FEM (MEASUREMENTS §52.537)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla
import torch

from formant_ml.physics.fold_modal import FoldLinearizer, build_solid

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lin():
    return FoldLinearizer(build_solid(n_ap=4, n_si=3, cells_per_layer=(1, 1, 1)))


def test_tangent_matches_finite_difference_of_internal_force(lin):
    lin.solve(0.1, 0.0)
    ctl = lin.control(0.1, 0.0)
    K = lin._tangent(lin.x, ctl)[lin.free_idx][:, lin.free_idx]
    d = np.random.default_rng(0).normal(size=len(lin.free_idx)) * 1e-9
    plus, minus = lin.x.clone(), lin.x.clone()
    plus.view(-1)[lin.free_idx] += torch.as_tensor(d)
    minus.view(-1)[lin.free_idx] -= torch.as_tensor(d)
    fd = (lin._residual(lin._mirror(plus), ctl) - lin._residual(lin._mirror(minus), ctl)) / 2
    np.testing.assert_allclose(-(K @ d), fd, rtol=1e-6, atol=1e-12 * np.abs(fd).max())


def test_prestrain_equilibrium_raises_first_mode_and_lowers_compliance(lin):
    lin.solve(0.1, 0.0)
    soft = lin.summary()
    lin.solve(0.4, 0.0)
    stiff = lin.summary()
    assert soft.newton_residual_n < 1e-9 and stiff.newton_residual_n < 1e-9
    assert np.all(np.diff(stiff.freq_hz) >= 0)
    assert stiff.freq_hz[0] > 1.3 * soft.freq_hz[0]
    assert stiff.compliance_m_per_pa.mean() < soft.compliance_m_per_pa.mean()
    assert np.all(stiff.compliance_m_per_pa > 0)
    assert np.all(stiff.ap_purity >= 0.8)
    assert 0 < stiff.damping_ratio[0] < 1          # Kelvin–Voigt: higher modes may be overdamped


def test_ta_activation_bulges_the_medial_surface(lin):
    lin.solve(0.1, 0.5)
    m = lin.summary()
    assert m.bulge_m.mean() > 0
    assert lin.ctl == (pytest.approx(0.1), pytest.approx(0.5))


def test_mode3_matched_pitch_and_static_compliance_are_consistent():
    spec = importlib.util.spec_from_file_location("fem_fold_calib", ROOT / "scripts" / "fem_fold_calib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sc = dict(len=1.0, thick=1.0, kc=1.0, zeta=0.1, bulge=0.1, shear=1.0)
    e = mod.matched_eps(250.0, 0.0, sc, 6)
    f, z, prof, comp, bulge = mod.mode3(e, 0.0, sc, 6)
    assert f[0] == pytest.approx(250.0, rel=1e-6)
    assert np.all(comp > 0) and np.all(np.abs(prof).max(1) == pytest.approx(1.0))
    assert mod.mode3(e + 0.1, 0.0, sc, 6)[0][0] > f[0]
