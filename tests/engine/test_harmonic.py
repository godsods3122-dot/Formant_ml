"""Individual harmonic errors must not disappear into envelope averages."""
import numpy as np
import pytest
import torch

from formant_ml.engine.harmonic import HarmonicMagnitudeLoss


FS, HOP, FRAMES = 48000, 48, 160


def _signal(f0=295.0, amplitudes=None, offsets=None):
    pitch = np.broadcast_to(f0, (FRAMES,)).copy()
    samples = np.arange(FRAMES * HOP)
    instant = np.interp(samples, (np.arange(FRAMES) + 0.5) * HOP, pitch)
    phase = 2 * np.pi * np.cumsum(instant) / FS
    amplitudes = 1.0 / np.arange(1, 9) if amplitudes is None else amplitudes
    offsets = np.zeros(8) if offsets is None else offsets
    basis = np.cos(phase[:, None] * np.arange(1, 9) + offsets)
    return basis @ amplitudes, pitch, basis


def _objective(target, pitch, voiced=None, fricative=None):
    return HarmonicMagnitudeLoss(
        target, FS, HOP, pitch,
        np.ones(FRAMES, bool) if voiced is None else voiced,
        np.zeros(FRAMES, bool) if fricative is None else fricative)


@pytest.mark.parametrize("f0", [295.0, 306.71875, 400.0])
def test_harmonic_amplitude_is_not_fft_bin_peak(f0):
    target, pitch, _ = _signal(f0)
    objective = _objective(target, pitch)
    amps = 1.0 / np.arange(1, 9)
    amps[4] *= 10 ** (-3.0 / 20)
    synth, _, _ = _signal(f0, amps)
    rep = objective.report(torch.tensor(synth, dtype=torch.float32))
    assert objective.observations > 0
    assert rep["by_order"]["5"]["mae_db"] == pytest.approx(3.0, abs=0.002)
    assert rep["by_order"]["1"]["mae_db"] < 0.002
    assert float(objective(torch.tensor(target, dtype=torch.float32))) < 1e-7


def test_relative_error_weights_do_not_follow_harmonic_energy():
    target, pitch, _ = _signal()
    objective = _objective(target, pitch)
    losses = []
    for k in (0, 6):
        amplitudes = 1.0 / np.arange(1, 9)
        amplitudes[k] *= 10 ** (3 / 20)
        synth, _, _ = _signal(amplitudes=amplitudes)
        losses.append(float(objective(torch.tensor(synth, dtype=torch.float32))))
    assert losses[0] == pytest.approx(losses[1], rel=0.002)


def test_fixed_target_phase_handles_pitch_motion_and_phase_offsets():
    pitch = np.linspace(260.0, 330.0, FRAMES)
    target, _, _ = _signal(pitch)
    objective = _objective(target, pitch)
    synth, _, _ = _signal(pitch, offsets=np.linspace(-1.0, 1.0, 8))
    rep = objective.report(torch.tensor(synth, dtype=torch.float32))
    assert rep["observations"] > 0
    assert rep["max_db"] < 0.002


def test_recorded_pulses_take_precedence_over_warm_start_f0():
    target, _, _ = _signal(295.0)
    objective = HarmonicMagnitudeLoss(
        target, FS, HOP, np.full(FRAMES, 310.0), np.ones(FRAMES, bool),
        np.zeros(FRAMES, bool), pulses=np.arange(0, FRAMES * HOP / FS, 1 / 295.0))
    synth, _, _ = _signal(295.0, offsets=np.linspace(-0.5, 1.2, 8))
    rep = objective.report(torch.tensor(synth, dtype=torch.float32))
    assert rep["observations"] > 0
    assert rep["max_db"] < 0.002


def test_amplitude_gradients_and_curvature_are_finite():
    target, pitch, basis = _signal()
    objective = _objective(target, pitch)
    amplitudes = torch.tensor(1.0 / np.arange(1, 9), requires_grad=True)
    with torch.no_grad():
        amplitudes[4] *= 1.4
    loss = objective(torch.tensor(basis) @ amplitudes)
    grad = torch.autograd.grad(loss, amplitudes, create_graph=True)[0]
    curvature = torch.autograd.grad(grad.sum(), amplitudes)[0]
    assert torch.isfinite(grad).all() and torch.isfinite(curvature).all()
    assert grad[4] > 0.0
    silent = torch.zeros(FRAMES * HOP, requires_grad=True)
    objective(silent).backward()
    assert torch.isfinite(silent.grad).all()


@pytest.mark.parametrize("excluded", ["unvoiced", "fricative", "silent"])
def test_missing_observations_are_not_reported_as_perfect(excluded):
    target, pitch, _ = _signal()
    if excluded == "silent":
        target = np.zeros_like(target)
    objective = _objective(target, pitch,
                           voiced=np.full(FRAMES, excluded != "unvoiced"),
                           fricative=np.full(FRAMES, excluded == "fricative"))
    y = torch.tensor(target, requires_grad=True)
    objective(y).backward()
    rep = objective.report(y)
    assert rep["observations"] == 0 and rep["mae_db"] is None
    assert not objective.coverage.any()
    assert torch.isfinite(y.grad).all()


def test_window_crossing_a_fricative_is_not_used():
    target, pitch, _ = _signal()
    fricative = np.zeros(FRAMES, bool)
    fricative[65:95] = True
    objective = _objective(target, pitch, fricative=fricative)
    assert objective.observations > 0
    assert not objective.coverage[65:95].any()


def test_projection_rejects_changed_sample_grid():
    target, pitch, _ = _signal()
    objective = _objective(target, pitch)
    with pytest.raises(ValueError, match="sample grid"):
        objective(torch.tensor(target[:-1]))
