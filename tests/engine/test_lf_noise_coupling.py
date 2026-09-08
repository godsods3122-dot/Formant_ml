"""Opt-in LF noise coupling: source consistency, gradients, and causal chunks."""
import math

import numpy as np
import pytest
import torch

from formant_ml.engine.control import ControlTrack, INDEX, PARAM_NAMES, default_vector
from formant_ml.engine.glottis import GlottalSource, lf_pulse
from formant_ml.engine.noise import FricationNoise, slow_modulation
from formant_ml.engine.rng import NoiseBank
from formant_ml.engine.tviir import lowpass_coeffs, tv_biquad
from formant_ml.engine.voice import EngineConfig, VoiceEngine

FS, HOP = 48000, 48


def _track(n=100, **values):
    tr = ControlTrack(np.tile(default_vector(), (n, 1)), 1.0)
    defaults = dict(p_sub=8.0, adduction=0.5, a_c=0.08, f0_target=200.0,
                    jitter=0.0, shimmer=0.0, f1=600, f2=1400, f3=2600, f4=3600,
                    residual_mix=0.0)
    for name, value in (defaults | values).items():
        tr[name] = value
    return tr


def _controls(track):
    ctrl = track.to_tensor()
    return {name: ctrl[..., INDEX[name]] for name in PARAM_NAMES}


@pytest.fixture(scope="module")
def glottis():
    return GlottalSource(FS, HOP, noise_modulation="lf")


def test_lf_envelope_tracks_integrated_source_and_rd_opening(glottis):
    n = glottis.lf_flow.shape[1]
    phase = (torch.arange(n, dtype=torch.float64) * (2 * math.pi / n))[None]
    peaks = []
    for index in (0, 8, 16, 23):
        rd = glottis.rd_grid[index].double().expand_as(phase)
        env = glottis.lf_noise_envelope(phase, rd, torch.ones_like(phase))
        e = lf_pulse(float(rd[0, 0]), n)
        flow = np.concatenate(([0.0], np.cumsum(e)[:-1]))
        flow /= flow.max()
        expected = 1 + 0.7 * (flow - flow.mean())
        np.testing.assert_allclose(env[0], expected, atol=2e-7)
        assert abs(float(env.mean()) - 1.0) < 1e-7
        assert float(env.min()) > 0
        r = float(rd[0, 0])
        ra, rk = (-1 + 4.8 * r) / 100, (22.4 + 11.8 * r) / 100
        rg = (rk / 4) * (0.5 + 1.2 * rk) / (0.11 * r - ra * (0.5 + 1.2 * rk))
        tp = 1 / (2 * rg)
        # Tp itself is not monotonic across the full Rd range; match the LF
        # timing rather than imposing that assumption on the source model.
        assert abs(int(env.argmax()) / n - tp) < 2 / n
        peaks.append(int(env.argmax()) / n)
    assert peaks[-1] - peaks[0] > 0.2, peaks


def test_lf_envelope_is_periodic_c1_and_unvoiced_is_unmodulated(glottis):
    phase = torch.tensor([[-1e-5, 0.0, 1e-5, 2 * math.pi]], dtype=torch.float64,
                         requires_grad=True)
    rd = torch.full_like(phase, 1.25)
    env = glottis.lf_noise_envelope(phase, rd, torch.ones_like(phase))
    grad, = torch.autograd.grad(env.sum(), phase)
    assert abs(float((env[0, 0] - env[0, 2]).detach())) < 1e-7
    assert abs(float(grad[0, 0] - grad[0, 2])) < 1e-4
    assert abs(float((env[0, 1] - env[0, 3]).detach())) < 1e-12
    torch.testing.assert_close(
        glottis.lf_noise_envelope(phase, rd, torch.zeros_like(phase)),
        torch.ones_like(phase), rtol=0, atol=0)


def test_lf_envelope_rd_and_phase_gradients_are_live(glottis):
    phase = torch.tensor([[0.7, 1.3, 2.4]], dtype=torch.float64, requires_grad=True)
    rd = torch.full_like(phase, 1.23, requires_grad=True)
    env = glottis.lf_noise_envelope(phase, rd, torch.ones_like(phase))
    grads = torch.autograd.grad(env.square().sum(), (phase, rd))
    for grad in grads:
        assert torch.isfinite(grad).all() and grad.abs().min() > 1e-5
    changed = glottis.lf_noise_envelope(phase, rd + 0.1, torch.ones_like(phase))
    assert float((changed - env).detach().abs().max()) > 0.01


def test_lf_envelope_preserves_full_cycle_level_not_chunk_mean(glottis):
    phase = torch.arange(4096, dtype=torch.float64)[None] * (2 * math.pi / 4096)
    for r in (0.3, 0.91, 1.7, 2.7):
        rd = torch.full_like(phase, r)
        env = glottis.lf_noise_envelope(phase, rd, torch.ones_like(phase))
        assert abs(float(env.mean()) - 1) < 1e-7
        # Both paths retain legacy voiced cycle-mean amplitude.
        assert float((0.8775 * env).mean()) == pytest.approx(0.8775, abs=1e-7)
        assert float((0.825 * env).mean()) == pytest.approx(0.825, abs=1e-7)
        chunks = [glottis.lf_noise_envelope(p, d, torch.ones_like(p))
                  for p, d in zip(phase.split(137, -1), rd.split(137, -1))]
        torch.testing.assert_close(torch.cat(chunks, -1), env, rtol=0, atol=0)


def test_shared_envelope_reaches_both_noise_sources(glottis):
    c = _controls(_track())
    g = glottis(c, amp0=torch.tensor([0.8]))
    fr = FricationNoise(FS, HOP, mod_depth=0.0)
    args = (c, g["ag_dc_frames"], g["phase"], g["voiced"])
    neutral = fr(*args, noise_am=torch.ones_like(g["phase"]))
    linked = fr(*args, noise_am=g["noise_am"])
    torch.testing.assert_close(linked["source"], neutral["source"] * g["noise_am"])
    base_asp = g["physiology"]["asp"][:, :1]
    torch.testing.assert_close(g["asp_env"], base_asp * 0.8775 * g["noise_am"])
    assert float((linked["source"] - neutral["source"]).abs().max()) > 1e-5


def test_rd_control_changes_noise_output_with_finite_nonzero_gradient(glottis):
    c = _controls(_track(80))
    c["rd_offset"] = c["rd_offset"].clone().requires_grad_(True)
    g = glottis(c, amp0=torch.tensor([0.8]))
    fr = FricationNoise(FS, HOP, mod_depth=0.0)
    out = fr(c, g["ag_dc_frames"], g["phase"], g["voiced"],
             noise_am=g["noise_am"])["source"]
    for loss in (out.square().mean(), g["asp_env"].square().mean()):
        grad, = torch.autograd.grad(loss, c["rd_offset"], retain_graph=True)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 1e-10
    changed = glottis(c | {"rd_offset": c["rd_offset"] + 0.4},
                      amp0=torch.tensor([0.8]))
    other = fr(c, changed["ag_dc_frames"], changed["phase"], changed["voiced"],
               noise_am=changed["noise_am"])["source"]
    assert float((other - out).abs().max()) > 1e-5


@pytest.mark.parametrize("fast", [False, True])
def test_knee_matches_legacy_forward_and_has_finite_difference_gradient(fast, monkeypatch):
    from formant_ml.engine import tviir
    monkeypatch.setattr(tviir, "USE_FAST_PATH", fast)
    white = NoiseBank(4).white("mod", 0, 301, 1, torch.float64, "cpu")
    log_knee = torch.tensor(math.log(8.0), dtype=torch.float64, requires_grad=True)
    y, _ = slow_modulation(white, 1000.0, log_knee.exp())
    old_k = float(log_knee.detach().exp())
    old, _ = tv_biquad(white, *lowpass_coeffs(old_k, 0.707, 1000.0))
    old = old * math.sqrt(1000.0 / (2 * math.pi * old_k))
    torch.testing.assert_close(y, old, rtol=1e-10, atol=1e-10)
    loss = y.square().mean()
    grad, = torch.autograd.grad(loss, log_knee)
    eps = 1e-4
    losses = [slow_modulation(white, 1000.0, (log_knee.detach() + delta).exp())[0]
              .square().mean() for delta in (-eps, eps)]
    numerical = (losses[1] - losses[0]) / (2 * eps)
    assert torch.isfinite(grad) and grad.abs() > 1e-4
    torch.testing.assert_close(grad, numerical, rtol=1e-5, atol=1e-7)


def test_noise_knee_changes_output_and_beta_is_checkpoint_only(glottis):
    c = _controls(_track(150))
    g = glottis(c, amp0=torch.tensor([0.8]))
    fr = FricationNoise(FS, HOP)
    assert "log_beta" not in dict(fr.named_parameters())
    assert "log_beta" in dict(fr.named_buffers())
    old_state = fr.state_dict()
    old_state["log_beta"] = torch.tensor(math.log(1.7))
    fr.load_state_dict(old_state, strict=True)
    args = (c, g["ag_dc_frames"], g["phase"], g["voiced"])
    out = fr(*args, noise_am=g["noise_am"])["source"]
    grad, = torch.autograd.grad(out.square().mean(), fr.log_knee)
    assert torch.isfinite(grad) and grad.abs() > 1e-10
    with torch.no_grad():
        fr.log_knee.add_(math.log(2))
    changed = fr(*args, noise_am=g["noise_am"])["source"]
    assert float((changed - out).abs().max()) > 1e-5


def test_knee_filter_chunk_state_preserves_output_and_gradient():
    white = NoiseBank(9).white("mod", 0, 281, 1, torch.float64, "cpu")
    log_knee = torch.tensor(math.log(8.0), dtype=torch.float64, requires_grad=True)
    full, zfull = slow_modulation(white, 1000.0, log_knee.exp())
    chunks, state = [], None
    for part in white.split(17, -1):
        y, state = slow_modulation(part, 1000.0, log_knee.exp(), zi=state)
        chunks.append(y)
    joined = torch.cat(chunks, -1)
    torch.testing.assert_close(joined, full, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(state, zfull, rtol=1e-10, atol=1e-10)
    full_grad, = torch.autograd.grad(full.square().mean(), log_knee)
    chunk_grad, = torch.autograd.grad(joined.square().mean(), log_knee)
    assert torch.isfinite(chunk_grad) and chunk_grad.abs() > 1e-4
    torch.testing.assert_close(chunk_grad, full_grad, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("knee", [0.0, -8.0, float("nan"), float("inf"), 500.0])
def test_knee_helper_rejects_invalid_cutoffs(knee):
    with pytest.raises(ValueError, match="knee_hz"):
        slow_modulation(torch.ones(1, 10), 1000.0, knee)


def test_knee_small_matched_noise_recovery():
    """Mechanism identifiability only, not held-out speech or slope fitting."""
    white = NoiseBank(17).white("mod", 0, 601, 1, torch.float64, "cpu")
    target, _ = slow_modulation(white, 1000.0, 8.0)
    log_knee = torch.tensor(math.log(4.0), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_knee], lr=0.06)
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        output, _ = slow_modulation(white, 1000.0, log_knee.exp())
        loss = (output - target).square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert abs(float(log_knee.detach().exp()) - 8.0) < 0.4
    assert losses[-1] < losses[0] * 0.01


def test_opt_in_default_compatibility_and_unvoiced_noise():
    default = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0))
    legacy = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0,
                                     noise_modulation="legacy"))
    lf = VoiceEngine(EngineConfig(residual=False, n_extra_formants=0, noise_modulation="lf"))
    lf.load_state_dict(legacy.state_dict(), strict=True)
    for add in (0.05, 0.5):
        tr = _track(100, adduction=add)
        a, a_parts = default.render(tr, return_parts=True)
        b = legacy.render(tr)
        c, c_parts = lf.render(tr, return_parts=True)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a_parts["du"], c_parts["du"])
        if add == 0.05:
            np.testing.assert_array_equal(a, c)
            assert np.max(np.abs(c_parts["fric"])) > 0
            assert np.max(np.abs(c_parts["asp"])) > 0
        else:
            assert np.max(np.abs(a - c)) > 1e-5
            for key in ("fric", "asp"):
                level_db = 10 * np.log10(np.mean(c_parts[key] ** 2)
                                        / np.mean(a_parts[key] ** 2))
                assert abs(level_db) < 1.0, (key, level_db)
    with pytest.raises(ValueError, match="noise_modulation"):
        VoiceEngine(EngineConfig(noise_modulation="typo"))


@pytest.mark.parametrize("chunk_ms", [1, 7, 53])
def test_lf_voice_long_short_chunks_equal_offline(chunk_ms):
    eng = VoiceEngine(EngineConfig(residual=False, n_extra_formants=2,
                                  noise_modulation="lf", seed=3))
    tr = _track(127, rd_offset=np.linspace(-0.15, 0.8, 127),
                jitter=0.003, shimmer=0.015)
    tr["adduction"][40:75] = 0.05
    tr["a_c"] = np.linspace(0.04, 0.12, 127)
    tr["f0_target"] = np.linspace(180, 230, 127)
    offline = eng.render(tr)
    streamed = np.concatenate(list(eng.stream(tr, chunk_ms=chunk_ms)))
    assert np.isfinite(streamed).all()
    assert np.max(np.abs(streamed - offline)) < 1e-5 * np.max(np.abs(offline))
