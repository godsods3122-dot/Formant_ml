"""Small numerical and integration checks for the default-off loaded source."""
import numpy as np
import pytest
import torch

from formant_ml.engine import glottis, tract
from formant_ml.engine.control import ControlTrack, default_vector
from formant_ml.engine.loaded_source import (
    RHO, C_SOUND, TRACT_AREA_CM2, ImpedanceLoadedSource,
    load_coefficients, loaded_flow, njit,
)
from formant_ml.engine.voice import EngineConfig, VoiceEngine

backend = pytest.mark.skipif(njit is None, reason="optional numba backend")
FS = 48000


def _inputs(n=12):
    ref = torch.linspace(-2.0, 5.0, n, dtype=torch.float64)[None]
    one = torch.ones_like(ref)
    tracks = [(one * f, one * bw) for f, bw in ((500, 40), (1500, 85), (2500, 120))]
    coeffs = load_coefficients(tracks, FS, 14.6)
    return (ref, one * 400, one * 1.0, one * 0.1, one * 30,
            *coeffs, torch.linspace(-0.3, 0.5, 8, dtype=torch.float64)[None])


@backend
def test_implicit_adjoint_including_terminal_and_initial_state():
    args = tuple(t.clone().requires_grad_() for t in _inputs(5))

    def run(*x):
        return loaded_flow(*x[:5], x[5:9], x[9], 0.7, 0.3)

    assert torch.autograd.gradcheck(run, args, eps=1e-5, atol=3e-5, rtol=2e-4)


@backend
def test_zero_load_tracks_reference_and_recurrence_streams():
    args = _inputs(29)
    initial = torch.zeros_like(args[-1])
    out, _, _ = loaded_flow(*args[:5], args[5:9], initial, 0.0, 0.3)
    torch.testing.assert_close(out, args[0], atol=2e-12, rtol=1e-12)
    full = loaded_flow(*args[:5], args[5:9], initial, 0.8, 0.3)
    first = loaded_flow(*[x[:, :11] for x in args[:5]],
                        [x[:, :11] for x in args[5:9]], initial, 0.8, 0.3)
    last = loaded_flow(*[x[:, 11:] for x in args[:5]],
                       [x[:, 11:] for x in args[5:9]], first[-1], 0.8, 0.3)
    for i in (0, 1):
        torch.testing.assert_close(torch.cat((first[i], last[i]), 1), full[i], rtol=0, atol=0)
    torch.testing.assert_close(last[-1], full[-1], rtol=0, atol=0)


def test_discrete_impedance_response_is_positive_real():
    args = _inputs(1)
    aa, bb, zz, ww = [t.numpy()[0, 0] for t in args[5:9]]
    frequencies = np.array([50.0, 350.0, 500.0, 700.0, 1500.0, 2500.0, 9000.0])
    residue = 2 * RHO * C_SOUND**2 / (TRACT_AREA_CM2 * 14.6)
    for f in frequencies:
        z = np.exp(2j * np.pi * f / FS)
        midpoint_input = (1 + z) / 2
        response = 0.3
        analytic = 0.3
        s = 2 * FS * (z - 1) / (z + 1)
        for j, (freq, bw) in enumerate(((500, 40), (1500, 85), (2500, 120))):
            a, b, gain, w = aa[j], bb[j], zz[j], ww[j]
            matrix = np.array([[2*a - 1, 2*b], [2*w*a, 1 + 2*w*b]])
            forcing = np.array([2*gain, 2*w*gain])
            state = np.linalg.solve(z * np.eye(2) - matrix, forcing * midpoint_input)
            response += (a*state[0] + b*state[1] + gain*midpoint_input) / midpoint_input
            analytic += residue*s / (s*s + 2*np.pi*bw*s + (2*np.pi*freq)**2)
        np.testing.assert_allclose(response, analytic, rtol=1e-11, atol=1e-11)
        assert response.real > 0
    # The load has both signs of reactance around a resonance, not just damping.
    s = 2j*np.pi*np.array([350, 700])
    mode = residue*s / (s*s + 2*np.pi*40*s + (2*np.pi*500)**2)
    assert mode[0].imag > 0 and mode[1].imag < 0


@backend
def test_fixed_geometry_unforced_energy_decays():
    n = 2400
    args = list(_inputs(n))
    args[0] = torch.zeros_like(args[0])
    args[4] = torch.zeros_like(args[4])
    initial = torch.tensor([[4., 0., 30., 20., -10., 40., 10., -20.]], dtype=torch.float64)
    residue = 2 * RHO * C_SOUND**2 / (TRACT_AREA_CM2 * 14.6)
    inertance = args[1][0, 0].item() / (2 * FS)
    coupling = 0.8

    def energy(state):
        return 0.5 * inertance * state[0, 0]**2 + coupling * state[0, 2:].square().sum() / (2*residue)

    old = energy(initial)
    start = old
    for begin in range(0, n, 48):
        end = begin + 48
        _, pressure, initial = loaded_flow(
            *[x[:, begin:end] for x in args[:5]],
            [x[:, begin:end] for x in args[5:9]], initial, coupling, 0.3)
        new = energy(initial)
        assert new <= old + 1e-12
        assert torch.isfinite(pressure).all()
        old = new
    assert old < start * 0.01


def _track():
    tr = ControlTrack(np.tile(default_vector(), (83, 1)))
    tr["p_sub"] = 7
    tr["adduction"] = 0.6
    tr["f0_target"] = np.linspace(185, 225, 83)
    tr["f1"], tr["f2"], tr["f3"], tr["f4"] = 650, 1500, 2600, 3600
    tr["f1"] += 70 * np.sin(np.arange(83) / 12)
    tr["bw1"] = np.linspace(35, 120, 83)
    tr["rd_offset"] = np.linspace(0.0, 0.3, 83)
    tr["jitter"], tr["shimmer"] = 0.002, 0.01
    tr["adduction"][32:47] = 0.05
    tr["p_sub"][60:] = 0
    tr["oral_open"][62:] = 0
    return tr


def test_disabled_identity_and_invalid_configuration(monkeypatch):
    legacy = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0))
    bypass = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0,
                                      glottal_source="loaded", load_coupling=0))
    bypass.load_state_dict(legacy.state_dict(), strict=True)
    a, parts = legacy.render(_track(), return_parts=True)
    b, bparts = bypass.render(_track(), return_parts=True)
    np.testing.assert_array_equal(a, b)
    for key in parts:
        np.testing.assert_array_equal(parts[key], bparts[key])
    assert "loaded" not in bypass.state
    assert not bypass.glottis.flow_reference
    for coupling in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="load_coupling"):
            EngineConfig(load_coupling=coupling)
    with pytest.raises(ValueError, match="glottal_source"):
        EngineConfig(glottal_source="typo")
    monkeypatch.setattr(tract, "OPEN_DAMP", True)
    with pytest.raises(ValueError, match="OPEN_DAMP"):
        VoiceEngine(EngineConfig(glottal_source="loaded"))


def test_missing_backend_is_explicit_but_zero_coupling_needs_none(monkeypatch):
    from formant_ml.engine import loaded_source
    monkeypatch.setattr(loaded_source, "njit", None)
    with pytest.raises(RuntimeError, match="realtime"):
        VoiceEngine(EngineConfig(glottal_source="loaded"))
    VoiceEngine(EngineConfig(glottal_source="loaded", load_coupling=0))


@backend
@pytest.mark.parametrize("chunk_ms", [1, 7])
def test_loaded_engine_streams_changing_controls_and_backpropagates(monkeypatch, chunk_ms):
    monkeypatch.setattr(glottis, "HARM_BLOCK", 0)
    eng = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0,
                                   glottal_source="loaded", load_coupling=0.7))
    tr = _track()
    offline, parts = eng.render(tr, return_parts=True)
    final = eng.state["loaded"]["flow"].clone()
    streaming = np.concatenate(list(eng.stream(tr, chunk_ms=chunk_ms)))
    np.testing.assert_allclose(streaming, offline, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(eng.state["loaded"]["flow"], final, rtol=1e-6, atol=1e-6)
    assert np.isfinite(offline).all()
    assert np.max(np.abs(parts["load_pressure"])) > 0.1
    assert np.max(np.abs(parts["flow"] - parts["reference_flow"])) > 0.01
    eng.reset()
    ctrl = tr.to_tensor()[:, :9].requires_grad_()
    out = eng(ctrl)
    loss = out["du"].square().mean() + out["load_pressure"].square().mean() * 1e-6
    loss.backward()
    assert torch.isfinite(ctrl.grad).all()
    assert ctrl.grad.abs().max() > 0


@backend
def test_blocked_flow_reference_matches_sequential(monkeypatch):
    eng = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0,
                                   glottal_source="loaded"))
    monkeypatch.setattr(glottis, "HARM_BLOCK", 0)
    _, a = eng.render(_track(), return_parts=True)
    monkeypatch.setattr(glottis, "HARM_BLOCK", 16)
    _, b = eng.render(_track(), return_parts=True)
    np.testing.assert_allclose(a["reference_flow"], b["reference_flow"], atol=8e-5, rtol=1e-5)
    np.testing.assert_allclose(a["du"], b["du"], atol=3e-5, rtol=1e-4)


@backend
def test_dynamic_reference_no_load_preserves_legacy_component():
    n = 79
    t = torch.linspace(0, 1, n, dtype=torch.float64)[None]
    ref = 20 * (1 + t) * torch.sin(17 * t)
    legacy = torch.cos(31 * t)  # Does not need to equal a discrete reference derivative.
    scale, rate, area, ps = 100 + 40*t, 180 + 90*t, 0.03 + 0.1*t, 4 + 5*t
    tracks = [(500 + 80*t, 35 + 40*t), (1500 + 50*t, 60 + 50*t),
              (2600 - 100*t, 80 + 60*t)]
    source = ImpedanceLoadedSource(FS, 14.6, 0.0)
    args = (ref, scale, rate, area, ps)
    full = source(*args, tracks, {}, legacy_du=legacy)
    torch.testing.assert_close(full["flow"], ref, atol=2e-12, rtol=1e-12)
    torch.testing.assert_close(full["du"], legacy, atol=2e-12, rtol=1e-12)
    first = source(*[x[:, :31] for x in args],
                   [(f[:, :31], bw[:, :31]) for f, bw in tracks], {}, legacy_du=legacy[:, :31])
    last = source(*[x[:, 31:] for x in args],
                  [(f[:, 31:], bw[:, 31:]) for f, bw in tracks],
                  first["state"], legacy_du=legacy[:, 31:])
    torch.testing.assert_close(torch.cat((first["du"], last["du"]), 1), legacy,
                               atol=2e-12, rtol=1e-12)


@backend
def test_tiny_positive_coupling_converges_to_legacy(monkeypatch):
    monkeypatch.setattr(glottis, "HARM_BLOCK", 0)
    tr = _track()
    legacy = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0))
    base, base_parts = legacy.render(tr, return_parts=True)
    errors = []
    for coupling in (1e-3, 1e-6, 1e-9):
        eng = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0,
                                       glottal_source="loaded", load_coupling=coupling))
        y, parts = eng.render(tr, return_parts=True)
        errors.append(np.linalg.norm(y - base))
        if coupling == 1e-9:
            np.testing.assert_allclose(parts["du"], base_parts["du"], atol=2e-7, rtol=1e-6)
    assert errors[1] < errors[0] * 0.01
    assert errors[2] < errors[1] * 0.1


@backend
def test_external_pulse_rate_defect_scaling_streaming_and_gradient(monkeypatch):
    tr = _track()
    ctrl = tr.to_tensor()[:, :25]
    n = ctrl.shape[1] * 48
    rates = torch.linspace(260, 340, n, dtype=torch.float64)[None]
    phase = torch.cumsum(rates * (2 * np.pi / FS), 1) + 31 * 2 * np.pi
    phase = phase.requires_grad_()
    eng = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0,
                                   glottal_source="loaded", load_coupling=0.7))
    captured = {}
    original = ImpedanceLoadedSource.__call__

    def inspect(self, reference, scale, rate, *args, **kwargs):
        captured.update(scale=scale, legacy=kwargs["legacy_du"])
        return original(self, reference, scale, rate, *args, **kwargs)

    monkeypatch.setattr(ImpedanceLoadedSource, "__call__", inspect)
    full = eng(ctrl, pulse_phase=phase)
    expected_rate = torch.cat((rates[:, 1:2], rates[:, 1:]), 1)
    torch.testing.assert_close(full["source_rate"], expected_rate, atol=2e-9, rtol=1e-11)
    defect = full["flow"] - full["reference_flow"].double()
    prev = torch.cat((torch.zeros_like(defect[:, :1]), defect[:, :-1]), 1)
    expected_du = captured["legacy"].double() + (defect - prev) * FS / (
        captured["scale"].double() * expected_rate)
    torch.testing.assert_close(full["du"], expected_du.float(), atol=1e-7, rtol=1e-6)
    full["du"].square().mean().backward()
    assert phase.grad is not None and torch.isfinite(phase.grad).all()
    assert phase.grad.abs().max() > 0
    eng.reset()
    chunks = []
    with torch.no_grad():
        for start in range(0, ctrl.shape[1], 7):
            end = min(ctrl.shape[1], start + 7)
            out = eng(ctrl[:, start:min(ctrl.shape[1], end + 1)],
                      pulse_phase=phase.detach(), emit=end - start)
            chunks.append(out["audio"])
    torch.testing.assert_close(torch.cat(chunks, 1), full["audio"].detach(),
                               atol=2e-6, rtol=1e-5)
    eng.reset()
    with pytest.raises(ValueError, match="unwrapped"):
        eng(ctrl, pulse_phase=torch.remainder(phase.detach(), 2*np.pi))
