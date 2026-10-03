"""Source-knob range limits, fast-motion prior and physiological start smoothing (MEASUREMENTS §52.536)."""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as fit_module
from formant_ml.engine.control import INDEX, ControlTrack, default_vector
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.voice import EngineConfig, VoiceEngine

T = 400


def _engine():
    return VoiceEngine(EngineConfig(residual=False))


def _track():
    track = ControlTrack(np.tile(default_vector(), (T, 1)))
    track["p_sub"] = 7.0
    track["adduction"] = 0.6
    track["f0_target"] = 200.0
    track["f1"], track["f2"], track["f3"], track["f4"] = 700, 1200, 2600, 3600
    track.voiced = np.ones(T, dtype=bool)
    return track


def _fitter(track, params):
    eng = _engine()
    return CopySynthFitter(eng, eng.render(track), 48000, track, params=params, harmonic_weight=0)


def test_src_bounds_narrow_control_range_and_pull_start_inside(monkeypatch):
    monkeypatch.setattr(fit_module, "SRC_BOUNDS", {"voice_gain": (-12.0, 12.0), "tilt": (-4.0, 8.0)})
    track = _track()
    vg = np.zeros(T)
    vg[100:150] = 20.0
    vg[200:250] = -20.0
    track["voice_gain"] = vg
    track["tilt"] = 11.0
    f = _fitter(track, ("voice_gain", "tilt"))
    specs = dict(zip(f.names, f.specs))
    assert (specs["voice_gain"].lo, specs["voice_gain"].hi) == (-12.0, 12.0)
    assert (specs["tilt"].lo, specs["tilt"].hi) == (-4.0, 8.0)
    out = f.result_track()
    assert out["voice_gain"].max() == pytest.approx(11.4, abs=1e-3)
    assert out["voice_gain"].min() == pytest.approx(-11.4, abs=1e-3)
    assert out["tilt"].max() == pytest.approx(7.7, abs=1e-3)
    np.testing.assert_allclose(out["voice_gain"][:100], 0.0, atol=1e-6)


def test_src_bounds_must_narrow_the_existing_range(monkeypatch):
    monkeypatch.setattr(fit_module, "SRC_BOUNDS", {"voice_gain": (-30.0, 12.0)})
    with pytest.raises(ValueError, match="must narrow"):
        _fitter(_track(), ("voice_gain",))


def _controls(f, voice_gain):
    c = torch.as_tensor(np.tile(default_vector(), (T, 1)), dtype=torch.float64).unsqueeze(0).clone()
    c[0, :, INDEX["p_sub"]] = 7.0
    c[0, :, INDEX["voice_gain"]] = torch.as_tensor(voice_gain, dtype=torch.float64)
    return c


def test_src_move_penalizes_cycle_wobble_not_slow_contour():
    f = _fitter(_track(), ("f1",))
    t = np.arange(T) / 1000.0
    flat = f.src_move_loss(_controls(f, np.zeros(T)))
    assert float(flat) < 1e-20
    slow = float(f.src_move_loss(_controls(f, 6.0 * np.sin(2 * np.pi * 4.0 * t))))
    wobble = 2.0 * np.where((np.arange(T) // 3) % 2 == 0, 1.0, -1.0)   # ±2 dB every 3 ms period
    fast = float(f.src_move_loss(_controls(f, wobble)))
    assert fast > 20.0 * slow
    assert f._last_srcmove["voice_gain"] > 0.5


def test_src_move_ignores_unvoiced_frames():
    track = _track()
    track.voiced = np.zeros(T, dtype=bool)
    f = _fitter(track, ("f1",))
    wobble = 2.0 * np.where((np.arange(T) // 3) % 2 == 0, 1.0, -1.0)
    assert float(f.src_move_loss(_controls(f, wobble))) == 0.0


def test_base_tau_smooths_listed_start_columns_only(monkeypatch):
    monkeypatch.setattr(fit_module, "BASE_TAU", True)
    track = _track()
    zig = np.where(np.arange(T) % 2 == 0, 1.1, 0.9)
    track["bw1"] = 80.0 * zig
    track["f1"] = 700.0 * zig
    f = _fitter(track, ("bw1", "f1"))
    raw = {n: f._to_raw(torch.as_tensor(track[n], dtype=torch.float64), sp) for n, sp in zip(f.names, f.specs)}
    i_bw, i_f1 = f.names.index("bw1"), f.names.index("f1")
    step = lambda x: float(torch.diff(x).abs().mean())
    assert step(f.u0[:, i_bw]) < 0.01 * step(raw["bw1"])
    torch.testing.assert_close(f.u0[:, i_f1], raw["f1"])
