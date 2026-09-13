"""Shared calibration must not absorb utterance controls or truncate models."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from formant_ml.engine import fit as fit_module
from formant_ml.engine import tract as tract_module
from formant_ml.engine.calibration import (
    apply_calibration, capture_calibration, combine_calibrations, engine_parameters,
    legacy_fields, load_calibration, normalize_calibration,
)
from formant_ml.engine.control import ControlTrack, default_vector
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.voice import EngineConfig, VoiceEngine


def _engine():
    return VoiceEngine(EngineConfig(residual=False))


def _track():
    track = ControlTrack(np.tile(default_vector(), (90, 1)))
    track["p_sub"] = 7.0
    track["adduction"] = 0.6
    track["f0_target"] = 200.0
    track["f1"], track["f2"], track["f3"], track["f4"] = 700, 1200, 2600, 3600
    return track


def test_capture_separates_three_scopes():
    data = capture_calibration(_engine(), gain_db=-12, pulse_phi0=0.4,
                               room_ir=np.array([1.0, 0.1]))
    assert "log_front_bw" in data["speaker"]
    assert set(data["recording"]) == {"hf_eq_db", "hf_eq_enabled", "room_ir"}
    assert {"log_mvf", "glottis_hjit_log", "glottis_src_eq_db",
            "gain_db", "pulse_phi0"} == set(data["utterance"])
    json.dumps(data, allow_nan=False)


def test_shared_median_does_not_pool_source_or_recording():
    a = capture_calibration(_engine(), gain_db=-20)
    b = copy.deepcopy(a)
    a["speaker"]["log_front_bw"] = -0.2
    b["speaker"]["log_front_bw"] = 0.6
    b["utterance"]["log_mvf"] = 20
    b["utterance"]["gain_db"] = 60
    b["recording"]["hf_eq_db"] = [9.0] * 8
    result = combine_calibrations([a, b])
    assert result["speaker"]["log_front_bw"] == pytest.approx(0.2)
    assert result["utterance"] == {}
    assert result["recording"] == {}


def test_partial_lock_copies_only_requested_scope():
    eng = _engine()
    old = float(eng.aspiration.log_mvf.detach())
    data = normalize_calibration({"log_front_bw": 0.3, "log_mvf": 2.0,
                                  "hf_eq_db": [1.0] * 8})
    loaded = apply_calibration(eng, data, ("speaker",))
    assert loaded == ("log_front_bw",)
    assert float(eng.tract.log_front_bw.detach()) == pytest.approx(0.3)
    assert float(eng.aspiration.log_mvf.detach()) == old
    assert torch.count_nonzero(eng.tract.hf_eq_db) == 0


def test_recording_eq_load_is_instance_local_and_does_not_add_zero_eq_controls(monkeypatch):
    monkeypatch.setattr(tract_module, "HF_EQ", False)
    eng, other, track = _engine(), _engine(), _track()
    base = eng.render(track)
    data = normalize_calibration({"hf_eq_db": [6.0] * 8})
    apply_calibration(eng, data, ("recording",))
    assert eng.tract.recording_eq_enabled
    assert not other.tract.recording_eq_enabled and not tract_module.HF_EQ
    assert not np.allclose(eng.render(track), base)
    np.testing.assert_array_equal(other.render(track), base)
    apply_calibration(eng, normalize_calibration({"hf_eq_db": [0.0] * 8}), ("recording",))
    assert not eng.tract.recording_eq_enabled
    np.testing.assert_array_equal(eng.render(track), base)


def test_saved_disabled_eq_does_not_activate_stale_parameters(monkeypatch):
    monkeypatch.setattr(tract_module, "HF_EQ", False)
    eng = _engine()
    with torch.no_grad():
        eng.tract.hf_eq_db.fill_(6.0)
    saved = capture_calibration(eng)
    assert not saved["recording"]["hf_eq_enabled"]
    target = _engine()
    apply_calibration(target, saved)
    assert not target.tract.output_eq_enabled
    np.testing.assert_array_equal(target.render(_track()), eng.render(_track()))


def test_shape_failure_is_atomic_not_silent_truncation():
    eng = _engine()
    before = eng.tract.log_front_bw.detach().clone()
    data = capture_calibration(eng)
    data["speaker"]["log_front_bw"] = 0.4
    data["speaker"]["hf_log_df"].append(0.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_calibration(eng, data)
    torch.testing.assert_close(eng.tract.log_front_bw, before)


@pytest.mark.parametrize("edit,match", [
    (lambda d: d["speaker"].update(log_front_bw=float("nan")), "finite"),
    (lambda d: d["speaker"].update(log_mvf=8.0), "belongs"),
    (lambda d: d.update(schema_version=2), "Unsupported"),
    (lambda d: d["recording"].update(room_ir=[]), "finite"),
    (lambda d: d["utterance"].update(gain_db=[1, 2]), "scalar"),
])
def test_bad_calibration_is_rejected(edit, match):
    data = capture_calibration(_engine())
    edit(data)
    with pytest.raises(ValueError, match=match):
        normalize_calibration(data)


def test_wrong_engine_or_mixed_shapes_cannot_be_combined():
    a = capture_calibration(_engine())
    b = copy.deepcopy(a)
    b["model"]["tract_length_cm"] = 12
    with pytest.raises(ValueError, match="model mismatch"):
        apply_calibration(_engine(), b)
    with pytest.raises(ValueError, match="model signatures"):
        combine_calibrations([a, b])
    b = copy.deepcopy(a)
    b["speaker"]["hf_log_bw"] = [0.0]
    with pytest.raises(ValueError, match="different shapes"):
        combine_calibrations([a, b])
    b = copy.deepcopy(a)
    del b["speaker"]["log_front_bw"]
    with pytest.raises(ValueError, match="incomplete"):
        combine_calibrations([a, b])


def test_different_room_responses_are_not_averaged():
    a = capture_calibration(_engine(), room_ir=np.array([1.0, 0.2]))
    b = capture_calibration(_engine(), room_ir=np.array([1.0, -0.2]))
    with pytest.raises(ValueError, match="impulse responses"):
        combine_calibrations([a, b], "recording")
    result = combine_calibrations([a], "recording")
    assert result["recording"]["room_ir"] == [1.0, 0.2]
    assert result["speaker"] == result["utterance"] == {}


@pytest.mark.parametrize("versioned", [False, True])
def test_archive_round_trip_restores_source_and_output_eq(tmp_path, versioned):
    source = _engine()
    with torch.no_grad():
        for i, p in enumerate(engine_parameters(source).values()):
            p.add_(0.01 * (i + 1))
    data = capture_calibration(source, gain_db=-7, pulse_phi0=0.3)
    old = {k: np.array(json.dumps(v)) for k, v in legacy_fields(data).items()}
    old["gain_db"] = np.array(-7.0)
    if versioned:
        old["acoustic_constants"] = np.array(json.dumps(data))
    path = tmp_path / "voice_track.npz"
    np.savez(path, **old)
    loaded = load_calibration(path)
    target = _engine()
    apply_calibration(target, loaded)
    for name, p in engine_parameters(source).items():
        torch.testing.assert_close(p, engine_parameters(target)[name])
    assert loaded["utterance"]["gain_db"] == -7


def test_legacy_room_sidecar_is_recording_only(tmp_path):
    path = tmp_path / "voice_track.npz"
    np.savez(path, engine_params=np.array('{"log_front_bw": 0.2}'))
    np.save(tmp_path / "voice_room.npy", np.array([1.0, 0.2]))
    loaded = load_calibration(path)
    assert loaded["recording"]["room_ir"] == [1.0, 0.2]
    assert "room_ir" not in combine_calibrations([loaded])["speaker"]


def test_legacy_band_metadata_survives_without_becoming_locked_constants(tmp_path):
    bands, q = {"f1": [400.0, 1400.0]}, {"f1": [3.0, 15.0]}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"log_front_bw": 0.2, "formant_band": bands,
                                "q_band": q}), encoding="utf-8")
    data = normalize_calibration(load_calibration(path))
    assert data["formant_band"] == bands
    assert data["q_band"] == q
    assert apply_calibration(_engine(), data, ("speaker",)) == ("log_front_bw",)


def test_fitter_locks_only_supplied_constants_and_retains_utterance(monkeypatch):
    monkeypatch.setattr(fit_module, "MVF_FROZEN", False)
    monkeypatch.setattr(fit_module, "SPEAKER_LOCK", False)
    eng, track = _engine(), _track()
    target = eng.render(track)
    f = CopySynthFitter(eng, target, 48000, track, params=("f1",),
                       harmonic_weight=0, locked_constants=("log_front_bw",))
    ids = {id(p) for p in f.opt_params()}
    assert id(eng.tract.log_front_bw) not in ids
    assert id(eng.tract.log_extra_bw) in ids
    assert id(eng.aspiration.log_mvf) in ids
    assert f.names == ["f1"]
    assert f.constant_budget()["speaker"]["locked"] == {"log_front_bw": 1}


def test_shared_formant_band_preserves_in_band_initial_value():
    eng, track = _engine(), _track()
    f = CopySynthFitter(eng, eng.render(track), 48000, track, params=("f1",),
                       harmonic_weight=0, formant_band={"f1": (400.0, 1400.0)})
    np.testing.assert_allclose(f.result_track()["f1"], track["f1"], rtol=1e-6)


def test_saved_gain_skips_recalibration_and_zero_budget_does_not_optimize(monkeypatch):
    monkeypatch.setattr(fit_module, "PULSE_LOCK", True)
    monkeypatch.setattr(fit_module, "HF_STAGE_ITERS", 0)
    eng, track = _engine(), _track()
    track.pulses = np.arange(0.005, 0.085, 0.005)
    target = eng.render(track)

    def forbidden(*args, **kwargs):
        raise AssertionError("Render-only must not recalibrate or optimize")

    monkeypatch.setattr(CopySynthFitter, "calibrate_gain", forbidden)
    f = CopySynthFitter(eng, target, 48000, track, params=("f1",),
                       harmonic_weight=0, initial_gain_db=-7.0, initial_pulse_phi0=0.37)
    before = [p.detach().clone() for p in f.opt_params()]
    monkeypatch.setattr(f, "pick_lr_global", forbidden)
    monkeypatch.setattr(f, "set_grid", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    report = f.fit_staged(global_iters=0, stage_iters=0, phase_iters=0, verbose=False)
    assert report.iters == 0
    assert np.isfinite(report.loss)
    assert f.gain_db() == pytest.approx(-7)
    assert f.pulse_phi0.item() == pytest.approx(0.37)
    for p, expected in zip(f.opt_params(), before):
        torch.testing.assert_close(p, expected, rtol=0, atol=0)


def test_explicit_hf_only_budget_does_not_sweep_phase_or_change_grid(monkeypatch):
    monkeypatch.setattr(fit_module, "PULSE_LOCK", True)
    monkeypatch.setattr(fit_module, "HF_STAGE_ITERS", 3)
    eng, track = _engine(), _track()
    track.pulses = np.arange(0.005, 0.085, 0.005)
    f = CopySynthFitter(eng, eng.render(track), 48000, track, params=("f1",),
                       harmonic_weight=0, initial_gain_db=0, initial_pulse_phi0=0.37)
    calls = []

    def fit_spy(iters, *args, **kwargs):
        calls.append((iters, getattr(f, "_hf_focus", False)))
        return fit_module.FitReport(0, 0, 0, {}, 0, iters, [])

    def forbidden(*args, **kwargs):
        raise AssertionError("HF-only must not sweep phase, LR, or control grids")

    monkeypatch.setattr(f, "fit", fit_spy)
    monkeypatch.setattr(f, "loss", forbidden)
    monkeypatch.setattr(f, "pick_lr_global", forbidden)
    monkeypatch.setattr(f, "set_grid", forbidden)
    f.fit_staged(global_iters=0, stage_iters=0, phase_iters=0, verbose=False)
    assert calls == [(0, False), (3, True)]
    assert f.pulse_phi0.item() == pytest.approx(0.37)


def test_speaker_script_writes_scoped_profile(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "speaker_profile.py"
    spec = importlib.util.spec_from_file_location("speaker_profile_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    src = tmp_path / "voice_constants.json"
    src.write_text(json.dumps(capture_calibration(_engine())), encoding="utf-8")
    data = module.combine([str(src)])
    assert data["schema_version"] == 1
    assert data["speaker"]
    assert not data["utterance"] and not data["recording"]


def test_sparse_events_reach_only_observed_controls_with_finite_gradients(monkeypatch):
    monkeypatch.setattr(fit_module, "EVENT_W", 1.0)
    monkeypatch.setattr(fit_module, "MOVE_FINE_PCT", 0.0)
    eng, track = _engine(), _track()
    track["front_len"] = 4.0
    track["front_len"][20:60] = 2.0
    track["obstacle"] = 0.0
    track["obstacle"][40:] = 0.2
    f = CopySynthFitter(eng, eng.render(track), 48000, track,
                       params=("f1", "f2", "front_len", "obstacle"),
                       harmonic_weight=0, initial_gain_db=0.0)
    assert f.w_ev.shape == (3,)
    assert [f.names[i] for i in f._ev_cols.tolist()] == [
        "front_len", "obstacle", "front_len"]
    with torch.no_grad():
        f.w_ev.copy_(torch.tensor([0.1, 0.2, -0.1], dtype=f.w_ev.dtype))
    delta = f._delta()
    torch.testing.assert_close(delta[:, :2], torch.zeros_like(delta[:, :2]))
    assert torch.count_nonzero(delta[:, 2:]) > 0
    f.loss()[0].backward()
    assert f.w_ev.grad is not None and torch.isfinite(f.w_ev.grad).all()
    assert torch.isfinite(f.event_loss())


def test_copyfit_render_only_round_trip_preserves_recording_chain(tmp_path, monkeypatch):
    import soundfile as sf

    path = Path(__file__).resolve().parents[2] / "scripts" / "copyfit.py"
    spec = importlib.util.spec_from_file_location("copyfit_calibration_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    track = _track()
    audio = _engine().render(track)
    audio = 0.1 * audio / np.max(np.abs(audio))
    wav = tmp_path / "input.wav"
    sf.write(wav, audio, 48000, subtype="FLOAT")
    channel = tmp_path / "recording.json"
    channel.write_text(json.dumps(normalize_calibration(
        {"hf_eq_db": [3.0] * 8, "room_ir": [1.0, 0.02]})), encoding="utf-8")
    monkeypatch.setattr(cli, "analyze", lambda *a, **kw: copy.deepcopy(track))
    common = ["copyfit.py", str(wav), "--no-denoise", "--harmonic", "0",
              "--global-iters", "0", "--stage-iters", "0", "--phase-iters", "0"]
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(cli.sys, "argv", common + [
        "--recording-lock", str(channel), "--out", str(first)])
    cli.main()
    monkeypatch.setattr(cli.sys, "argv", common + [
        "--init", str(first), "--out", str(second)])
    cli.main()
    y1, _ = sf.read(str(first) + "_fit.wav")
    y2, _ = sf.read(str(second) + "_fit.wav")
    np.testing.assert_array_equal(y2, y1)
    for stem in (first, second):
        constants = load_calibration(str(stem) + "_track.npz")
        assert constants["recording"]["hf_eq_enabled"]
        assert constants["recording"]["room_ir"] == pytest.approx([1, 0.02])
        assert Path(str(stem) + "_constants.json").exists()
