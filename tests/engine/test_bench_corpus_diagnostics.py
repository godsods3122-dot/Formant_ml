"""CLI validation and same-fit seed reuse without requiring corpus recordings."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def script(name="bench_corpus"):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preprocessing_modes_are_explicit(monkeypatch):
    m = script()
    calls = []
    x = np.ones(5000)
    monkeypatch.setattr(m, "noise_profile", lambda y, sr: "profile")

    def denoise(y, sr, profile, **kwargs):
        calls.append(kwargs)
        return y * .5

    monkeypatch.setattr(m, "denoise", denoise)
    np.testing.assert_array_equal(m.prepare_audio(x, 48000, "raw"), x)
    assert not calls
    m.prepare_audio(x, 48000, "weak")
    m.prepare_audio(x, 48000, "current")
    assert calls == [{"over": .5, "floor_db": -6.}, {}]
    with pytest.raises(ValueError):
        m.prepare_audio(x, 48000, "unknown")


@pytest.mark.parametrize("args", [
    ["--budget", "1,2"], ["--budget", "-1,2,3"], ["--seeds", "1,1"],
    ["--seeds", "x"], ["--patience", "-1"], ["--denoise", "other"],
])
def test_cli_rejects_bad_arguments(args):
    with pytest.raises(SystemExit) as exc:
        script().main(args)
    assert exc.value.code == 2


def test_seed_renders_reuse_fit_and_raw_masks(monkeypatch):
    m = script()
    x = np.random.default_rng(1).normal(size=12000)
    monkeypatch.setattr(m.sf, "read", lambda path: (x, 48000))

    class Track(dict):
        frame_ms = 1.
        voiced = np.zeros(250, bool)

    track = Track(f0_target=np.full(250, 200.), f1=np.ones(250), f2=np.ones(250))
    monkeypatch.setattr(m, "analyze", lambda *a, **k: track)
    monkeypatch.setattr(m, "VoiceEngine", lambda cfg, prof: SimpleNamespace(cfg=cfg))
    events = []

    class Fitter:
        def __init__(self, eng, seg, sr, tr):
            assert eng.cfg.noise_modulation == "lf"
            self.target = torch.tensor(seg[None, :])
            self.f_max = 20000.
            events.append("construct")

        def fit_staged(self, **kwargs):
            events.append(("fit", kwargs["global_iters"], kwargs["phase_iters"]))
            return SimpleNamespace(env=70.)

        def render(self, seed=None):
            events.append(("render", seed))
            return x + .01 * np.random.default_rng(seed).normal(size=len(x))

        def result_track(self):
            return track

    monkeypatch.setattr(m, "CopySynthFitter", Fitter)
    r = m.bench("unused.wav", 0, .25, None, iters=(1, 0, 0), probe=False,
                denoise_mode="raw", seeds=(4, 8, 12), coupling="lf")
    assert events == ["construct", ("fit", 1, 0), ("render", 4), ("render", 8), ("render", 12)]
    assert r["fits"] == 1
    assert r["render_seeds"] == [4, 8, 12]
    assert r["fidelity_summary"]["fine"]["count"] == 3
    assert r["mask_definition"]["source"].startswith("raw target")
    assert np.isnan(r["nh_target"])  # Unvoiced frication must not become voiced residual.
    json.dumps(m.json_safe(r), allow_nan=False)


def test_cli_json_missing_corpus_and_unavailable_values(monkeypatch, tmp_path):
    m = script()
    monkeypatch.setattr(m, "SEGMENTS", [("missing", "missing.wav", 0, .2, "sib")])
    monkeypatch.setattr(m.SpeakerProfile, "load", lambda path: None)
    output = tmp_path / "report.json"
    m.main(["--budget", "1,0,0", "--no-probe", "--json", str(output)])
    report = json.loads(output.read_text())
    assert report["budget"] == [1, 0, 0]
    assert report["segments"][0]["status"] == "unavailable"
    assert m.json_safe({"x": np.nan, "y": np.inf}) == {"x": None, "y": None}


def test_cli_dispatches_conditions_without_refitting_each_seed(monkeypatch, tmp_path):
    m = script()
    audio = tmp_path / "exists.wav"
    audio.touch()
    monkeypatch.setattr(m, "SEGMENTS", [("fake", str(audio), 0, .2, "sib")])
    monkeypatch.setattr(m.SpeakerProfile, "load", lambda path: None)
    calls = []

    def bench(path, t0, t1, prof, verbose, iters, **kwargs):
        calls.append((iters, kwargs))
        return dict(env=0., fine=0., fine_corr=100., correction_status="unresolved",
                    noise_ratio=1., phase=np.nan, snr=np.nan, centroid_err=0.,
                    band_mae=0., mod_mae=0., nh_synth=np.nan, nh_target=np.nan,
                    region_summary={})

    monkeypatch.setattr(m, "bench", bench)
    output = tmp_path / "report.json"
    m.main(["--budget", "2,1,0", "--seeds", "4,8", "--no-probe",
            "--denoise", "all", "--coupling", "all", "--json", str(output)])
    assert len(calls) == 6
    assert {(kw["denoise_mode"], kw["coupling"]) for _, kw in calls} == {
        (mode, coupling) for mode in ("raw", "weak", "current") for coupling in ("legacy", "lf")}
    assert all(iters == (2, 1, 0) and kw["seeds"] == (4, 8) and not kw["probe"]
               for iters, kw in calls)
    assert len(json.loads(output.read_text())["segments"]) == 6


def test_listening_audio_uses_shared_gain_and_anonymous_names(tmp_path):
    m = script()
    x = np.sin(np.arange(2000)) * 2
    m.write_listening(tmp_path, x, x * .5, [x * .25], [7])
    key = json.loads((tmp_path / "key.json").read_text())
    peaks = {}
    for filename, label in key["key"].items():
        assert filename.startswith("sample_")
        wave, sr = m.sf.read(tmp_path / filename)
        assert sr == 48000
        peaks[label] = np.max(np.abs(wave))
    assert peaks["raw_target"] <= .951
    assert peaks["synth_seed_7"] / peaks["raw_target"] == pytest.approx(.25, abs=1e-5)


def test_parametrize_log_does_not_claim_lower_bound(capsys):
    m = script("parametrize_corpus")
    row = dict(stem="fake", kind="fricative", env=90., resolved=False, fine_corr=100.,
               noise_ratio=1., ok=True)
    m._log(io.StringIO(), row, 1, 1, m.time.time())
    text = capsys.readouterr().out
    assert "unresolved estimate" in text
    assert ">" not in text and "≥" not in text
