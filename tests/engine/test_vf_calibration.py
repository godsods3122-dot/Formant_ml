"""Numerical calibration is tied to actual solver coordinates, not human norms."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from formant_ml.engine import fit as fm, voice as voice, voice_td as vt
from formant_ml.engine.control import ControlTrack, INDEX, PARAM_NAMES, default_vector
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.vf_calibration import (
    check_calibration, grid_index, locked_names, model_signature, pitch_coverage,
    validate_pitch_table,
)
from formant_ml.physics import beam_membrane as bm


def _fitter(n):
    fit = object.__new__(CopySynthFitter)
    fit.track = SimpleNamespace(voiced=np.ones(n, bool))
    return fit


def _controls(f0, add, rdo, ta, pressure=0.05):
    cols = np.broadcast_arrays(f0, add, rdo, ta)
    c = torch.tensor(np.tile(default_vector(), (cols[0].size, 1)), dtype=torch.float64)[None]
    for name, value in zip(("f0_target", "adduction", "rd_offset", "vf_ta"), cols):
        c[0, :, INDEX[name]] = torch.as_tensor(value.ravel())
    c[0, :, INDEX["p_sub"]] = pressure
    return c


@pytest.mark.parametrize("name", [
    "KWF_n10_ns_calib.npz", "KWF_m21_ns_calib_e50_r100.npz",
])
def test_saved_threshold_knots_are_recovered(name, monkeypatch):
    path = Path("profiles") / "vf" / name
    monkeypatch.setattr(fm, "PTH_MAP", str(path))
    with np.load(path, allow_pickle=False) as z:
        grids = np.meshgrid(*(z[k] for k in ("f0", "add", "rdo", "ta")), indexing="ij")
        for knot in np.ndindex(z["pth"].shape):
            if not np.isfinite(z["pth"][knot]):
                continue
            fit = _fitter(1)
            c = _controls(*(g[knot] for g in grids))
            loss = fit.pth_loss(c)
            recovered = math.exp(math.sqrt(float(loss))) * 0.05 / fm.PTH_MARGIN
            assert recovered == pytest.approx(z["pth"][knot], rel=1e-10)


def test_nonuniform_four_axis_interpolation_and_gradients(tmp_path, monkeypatch):
    axes = ([180., 230., 410.], [.3, .35, .45, .55], [-1., .2, 1.], [.1, .3, .8])
    f, a, r, t = np.meshgrid(*axes, indexing="ij")
    pth = np.exp(.2 * np.log(f) + a + .1 * r + .5 * t)
    path = tmp_path / "map.npz"
    np.savez(path, pth=pth, **dict(zip(("f0", "add", "rdo", "ta"), axes)))
    monkeypatch.setattr(fm, "PTH_MAP", str(path))
    c = _controls(215., .4, .3, .4, pressure=1.).requires_grad_()
    fit = _fitter(1)
    loss = fit.pth_loss(c)
    expected = (.2 * math.log(215) + .4 + .1 * .3 + .5 * .4
                + math.log(fm.PTH_MARGIN)) ** 2
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-10)
    assert torch.autograd.gradcheck(fit.pth_loss, (c,), eps=1e-5, atol=1e-5)


def test_singleton_threshold_axes_and_invalid_grid(tmp_path, monkeypatch):
    path = tmp_path / "map.npz"
    np.savez(path, pth=np.full((1, 1, 1, 1), 3.), f0=[200.], add=[.35], rdo=[0.], ta=[.2])
    monkeypatch.setattr(fm, "PTH_MAP", str(path))
    c = _controls([100., 500.], [.2, .9], [-1., 1.], [.1, .8])
    assert float(_fitter(2).pth_loss(c)) == pytest.approx(math.log(3 * fm.PTH_MARGIN / .05) ** 2)
    np.savez(path, pth=np.ones((2, 1, 1, 1)), f0=[200., 200.], add=[.35], rdo=[0.], ta=[.2])
    with pytest.raises(ValueError, match="Invalid NS threshold grid"):
        _fitter(2).pth_loss(c)


def test_grid_index_border_and_nonuniform_knots():
    x = torch.tensor([-.1, .3, .35, .4, .45, .55, .8], dtype=torch.float64, requires_grad=True)
    y = grid_index(x, [.3, .35, .45, .55])
    torch.testing.assert_close(y, torch.tensor([0., 0., 1., 1.5, 2., 3., 3.], dtype=torch.float64))
    y.sum().backward()
    assert x.grad[3] == pytest.approx(10.)
    assert x.grad[0] == x.grad[-1] == 0


def test_threshold_uses_rendered_not_raw_controls(tmp_path, monkeypatch):
    path = tmp_path / "map.npz"
    np.savez(path, pth=np.array([2., 8.]).reshape(1, 2, 1, 1),
             f0=[200.], add=[.3, .5], rdo=[0.], ta=[.3])
    monkeypatch.setattr(fm, "PTH_MAP", str(path))
    raw = _controls(200., .5, 0., .3, pressure=1.)
    rendered = _controls(200., .3, 0., .3, pressure=1.5).requires_grad_()
    fit = _fitter(1)
    fit.eng = SimpleNamespace(td=SimpleNamespace(last_vf_controls={
        k: rendered[:, :, INDEX[k]] for k in ("f0_target", "adduction", "rd_offset", "vf_ta", "p_sub")
    }))
    loss = fit.pth_loss(raw)
    assert float(loss.detach()) == pytest.approx(math.log(2 * fm.PTH_MARGIN / 1.5) ** 2)
    loss.backward()
    assert rendered.grad[0, 0, INDEX["p_sub"]] < 0


def test_ns_render_caches_filtered_controls_without_preroll(monkeypatch):
    for name, value in dict(GLOTTIS="vf", VF_NS=True, VF_BODY_DRIVE=False,
                            VF_NEURAL=False, ARTIC="cos", SLEW_ON=True,
                            VF_ADD_SMOOTH_MS=3., VF_TA_SMOOTH_MS=3.,
                            VF_PS_SMOOTH_MS=3.).items():
        monkeypatch.setattr(vt, name, value)
    T, hop = 80, 48
    td = vt.TDPath(48000, hop).double()
    raw = torch.tensor(np.tile(default_vector(), (T, 1)), dtype=torch.float64)[None]
    c = {k: raw[:, :, i].clone() for i, k in enumerate(PARAM_NAMES)}
    c["p_sub"][:, :40], c["p_sub"][:, 40:] = 6., 10.
    c["adduction"][:, :40], c["adduction"][:, 40:] = .3, .5
    c["vf_ta"][:, :40], c["vf_ta"][:, 40:] = .2, .5
    c["p_sub"].requires_grad_()
    td.vf_pscorr = torch.full((1, T), 1.2, dtype=torch.float64)
    st = dict(f0=torch.full((1, T), 230., dtype=torch.float64),
              amp=torch.ones(1, T, dtype=torch.float64), ag_dc=torch.zeros(1, T, dtype=torch.float64))
    captured = {}
    original = vt.td.tube_torch

    def solver(*args, **kwargs):
        captured["pressure"] = args[3]
        return original(*args, **kwargs)

    monkeypatch.setattr(vt.td, "tube_torch", solver)
    out = td(c, st, torch.zeros(1, T * hop, dtype=torch.float64), T * hop)
    assert torch.isfinite(out["audio"]).all()
    effective = td.last_vf_controls
    assert all(x.shape == (1, T) for x in effective.values())
    assert not torch.equal(effective["adduction"], c["adduction"])
    assert not torch.equal(effective["vf_ta"], c["vf_ta"])
    assert float(effective["p_sub"][0, 10].detach()) == pytest.approx(7.2)
    reconstructed = vt.spline_up(effective["p_sub"][:, :, None], hop * vt.OS)[0, :, 0]
    npre = td._n_pre_frames * hop * vt.OS
    actual = captured["pressure"][npre:] / vt.td.CMH2O
    # Ignore the spline support that crosses the removed pre-roll.
    torch.testing.assert_close(reconstructed[3 * hop * vt.OS:], actual[3 * hop * vt.OS:])
    effective["p_sub"].sum().backward()
    assert c["p_sub"].grad.abs().sum() > 0


def test_ns_bounds_recover_warm_start_and_preserve_rendered_scales(monkeypatch):
    monkeypatch.setattr(vt, "VF_NS_HARD", dict(vt.VF_NS_HARD_DEFAULT))
    td = vt.TDPath(48000, 48).double()
    with torch.no_grad():
        td.log_ns_len.fill_(math.log(1.26028044))
    before = td.ns_scales()["len"].detach().clone()
    fit = _fitter(1)
    fit.eng = SimpleNamespace(td=td)
    prior = fit.vf_ns_anat_loss()
    prior.backward()
    assert td.log_ns_len.grad > 0
    changed = td.project_ns_bounds()
    assert changed["len"][0] == pytest.approx(1.26028044)
    torch.testing.assert_close(td.ns_scales()["len"], before)
    td.log_ns_len.grad = None
    td.ns_scales()["len"].backward()
    assert td.log_ns_len.grad > 0
    opt = torch.optim.SGD([td.log_ns_len], lr=.01)
    opt.step()
    td.project_ns_bounds()
    assert float(td.log_ns_len.detach().exp()) < 1.15


def test_fitter_projection_uses_ns_bounds(monkeypatch):
    monkeypatch.setattr(voice, "TD_TUBE", True)
    monkeypatch.setattr(vt, "GLOTTIS", "vf")
    monkeypatch.setattr(vt, "VF_NS", True)
    monkeypatch.setattr(vt, "VF_NS_HARD", dict(vt.VF_NS_HARD_DEFAULT))
    fit = _fitter(1)
    fit.eng = SimpleNamespace(td=vt.TDPath(48000, 48).double())
    with torch.no_grad():
        fit.eng.td.log_ns_len.fill_(math.log(1.26))
    fit._project_ns_bounds()
    assert float(fit.eng.td.log_ns_len.detach()) == pytest.approx(math.log(1.15))
    assert not fit.eng.td.project_ns_bounds()


def test_optimizer_cannot_leave_ns_hard_bounds(monkeypatch):
    monkeypatch.setattr(voice, "TD_TUBE", True)
    monkeypatch.setattr(vt, "GLOTTIS", "vf")
    monkeypatch.setattr(vt, "VF_NS", True)
    monkeypatch.setattr(vt, "VF_NS_HARD", dict(vt.VF_NS_HARD_DEFAULT))
    monkeypatch.setattr(fm, "BALANCE", False)
    monkeypatch.setattr(fm, "LR_BUMP_MAX", 0)
    track = ControlTrack(np.tile(default_vector(), (80, 1)))
    eng = voice.VoiceEngine(voice.EngineConfig(residual=False))
    fit = CopySynthFitter(eng, np.zeros(3840), 48000, track, initial_gain_db=0.)
    p = eng.td.log_ns_len
    seen = []

    def loss():
        seen.append(float(p.detach().exp()))
        fit._last_db = 0.
        return (p.exp() - 2.) ** 2, p.detach() * 0., p.detach() * 0., {}

    monkeypatch.setattr(fit, "loss", loss)
    fit.fit(3, lr=.5, params=[p], verbose=False)
    assert len(seen) == 3
    assert seen[1] == pytest.approx(1.15)
    assert max(seen) <= 1.15 + 1e-12
    assert float(p.detach().exp()) <= 1.15 + 1e-12


def _signature_table(path, td):
    np.savez(path, eps_tab=[-.3, 0., .7], f0_tab=[160., 220., 600.],
             signature=json.dumps(model_signature(td)))


def test_calibration_rejects_drift_and_unlocked_constants(tmp_path, monkeypatch):
    monkeypatch.setattr(vt, "GLOTTIS", "vf")
    monkeypatch.setattr(vt, "VF_NS", True)
    monkeypatch.setattr(vt, "VF_NS_EPS_TAB", (-.3, 0., .7))
    monkeypatch.setattr(vt, "VF_NS_F0_TAB", (160., 220., 600.))
    td = vt.TDPath(48000, 48).double()
    path = tmp_path / "calib.npz"
    _signature_table(path, td)
    check_calibration(path, td, locked_names(), log=lambda _: None)
    with pytest.raises(ValueError, match="not locked"):
        check_calibration(path, td, ())
    monkeypatch.setattr(bm, "CLOSURE_EDGE_CM", bm.CLOSURE_EDGE_CM * 2)
    with pytest.raises(ValueError, match="closure_edge_cm"):
        check_calibration(path, td, locked_names())
    logs = []
    check_calibration(path, td, (), approximate=True, log=logs.append)
    assert any("WARNING: APPROXIMATE" in line for line in logs)


def test_legacy_metadata_and_lr_dependencies_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(vt, "GLOTTIS", "vf")
    monkeypatch.setattr(vt, "VF_NS", True)
    monkeypatch.setattr(bm, "LR_FOLDS", True)
    td = vt.TDPath(48000, 48).double()
    path = tmp_path / "old.npz"
    np.savez(path, consts=json.dumps(model_signature(td)["consts"]),
             cond=json.dumps(dict(edge_um=bm.CLOSURE_EDGE_CM * 1e4, rest_modal_um=0.)))
    logs = []
    check_calibration(path, td, locked_names(), log=logs.append)
    assert "td_ns_lr_q" in locked_names() and "td_ns_lr_m" in locked_names()
    assert any("not fully recorded" in line for line in logs)
    monkeypatch.setattr(vt, "VF_R_MODAL", .01)
    with pytest.raises(ValueError, match="rest_modal_um"):
        check_calibration(path, td, locked_names())


def test_pitch_coverage_reports_saturation_without_widening_strain(monkeypatch):
    with np.load(Path("profiles") / "vf" / "KWF_n10_ns_calib.npz") as z:
        monkeypatch.setattr(vt, "VF_NS_EPS_TAB", tuple(z["eps_tab"]))
        monkeypatch.setattr(vt, "VF_NS_F0_TAB", tuple(z["f0_tab"]))
    report = pitch_coverage([168., 180., 190., 320.], [True] * 4)
    assert report["outside"] == report["saturated"] == 3
    assert pitch_coverage([168., 320.], [False, True])["saturated"] == 0
    assert pitch_coverage([], [])["frames"] == 0
    for eps, f0 in [([-.31, .2], [100., 200.]), ([0., .2], [200., 100.]),
                    ([0., 0.], [100., 200.]), ([0., .2], [100., np.nan])]:
        with pytest.raises(ValueError):
            validate_pitch_table(eps, f0)


def test_pitch_table_excludes_irregular_branches():
    path = Path("scripts") / "vf_ns_calib.py"
    spec = importlib.util.spec_from_file_location("vf_ns_calib_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Bench:
        def threshold(self, ps_max, eps):
            return 3.

        def f0_at(self, p, eps):
            return 200 + 100 * eps, (.5 if eps == 0 else .95), eps == .1, 100.

    rows, e, f = module.f0_table(Bench(), [-.3, 0., .1, .7], 15., log=lambda _: None)
    assert len(rows) == 4
    np.testing.assert_array_equal(e, [-.3, .7])
    np.testing.assert_array_equal(f, [170., 270.])


def test_reference_frame_uses_dump_sample_rate_and_preroll(tmp_path):
    spec = importlib.util.spec_from_file_location("vf_ns_calib_reference", Path("scripts") / "vf_ns_calib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "input.npz"
    A = np.arange(1., 1001.)[:, None] * np.ones((1, 4))
    L = np.full(1000, 15.)
    np.savez(path, A=A, L=L, fs=2000., n_pre=200)
    area, length = module.reference_frame(path, .4, .1)
    np.testing.assert_array_equal(area, A[400])
    assert length == 15.
    np.savez(path, A=A, L=L)
    with pytest.raises(ValueError, match="TD_DUMP_IN"):
        module.reference_frame(path, .4, .1)
