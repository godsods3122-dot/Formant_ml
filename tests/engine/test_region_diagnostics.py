"""Target-defined, contiguous and explicitly limited pronunciation diagnostics."""
import numpy as np
import pytest

from formant_ml.engine import turbulence as tb

FS = 48000


def noise(seed=0, seconds=0.15):
    return np.random.default_rng(seed).normal(size=int(FS * seconds))


def region(spans):
    return {"regions": {"frication_core": {
        "spans": spans, "definition": "test target mask", "confidence": "heuristic"}}}


def test_masks_distinguish_unvoiced_frication_overlap_and_onset():
    from scipy.signal import butter, sosfilt
    n = int(FS * 0.2)
    t = np.arange(n) / FS
    high = sosfilt(butter(4, [4000, 10000], fs=FS, btype="band", output="sos"),
                   np.random.default_rng(1).normal(size=n))
    x = high.copy()
    x[int(.12 * FS):] = np.sin(2 * np.pi * 200 * t[int(.12 * FS):])
    voiced = np.arange(200) >= 90
    masks = tb.target_regions(x, FS, voiced)["regions"]
    assert all(masks[k]["spans"] for k in masks)
    core = masks["frication_core"]["spans"]
    overlap = masks["voiced_overlap"]["spans"]
    onset = masks["vowel_onset"]["spans"]
    assert max(b for _, b in core) <= .09 * FS
    assert min(a for a, _ in overlap) >= .09 * FS
    assert min(a for a, _ in onset) >= min(a for a, _ in overlap)


def test_vowel_clip_has_no_invented_onset():
    t = np.arange(9600) / FS
    masks = tb.target_regions(np.sin(2 * np.pi * 200 * t), FS, np.ones(200, bool))
    assert masks["regions"]["vowel_onset"]["spans"] == []
    assert masks["regions"]["frication_core"]["spans"] == []


def test_regions_are_contiguous_not_spliced(monkeypatch):
    calls = []

    def measure(t, s, fs, bandwidth):
        calls.append((t.copy(), s.copy()))
        return {"status": "available", "reason": None, "metrics": {"metric": float(t.mean())}}

    monkeypatch.setattr(tb, "diagnostic_metrics", measure)
    x = np.arange(10000, dtype=float)
    result = tb.region_diagnostics(x, -x, FS, region([[0, 1000], [5000, 7000]]))
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][0], x[5000:7000])
    np.testing.assert_array_equal(calls[1][1], -x[5000:7000])
    r = result["frication_core"]
    assert r["metrics"]["metric"] == pytest.approx((499.5 + 2 * 5999.5) / 3)
    assert r["metric_duration_s"]["metric"] == pytest.approx(3000 / FS)


@pytest.mark.parametrize("x,reason", [(np.zeros(4800), "silent target"),
                                     (np.ones(200), "less than 20 ms contiguous data")])
def test_insufficient_data_unavailable(x, reason):
    r = tb.diagnostic_metrics(x, x, FS)
    assert r["status"] == "unavailable"
    assert r["reason"] == reason
    assert not r["metrics"]


def test_short_islands_do_not_become_sufficient_by_concatenation():
    x = noise()
    r = tb.region_diagnostics(x, x, FS, region([[0, 500], [1000, 1500], [2000, 2500]]))
    assert r["frication_core"]["status"] == "unavailable"
    assert not r["frication_core"]["metrics"]


def test_narrow_bandwidth_and_short_modulation_are_unavailable():
    x = noise()
    m = tb.diagnostic_metrics(x, x, FS, bandwidth=2000)["metrics"]
    assert m["envelope_corr_4000_12000"] is None
    assert m["mod_error_pp_4000_12000_60_400"] is None
    assert m["mod_error_pp_300_1000_5_60"] is None
    assert m["mod_error_pp_300_1000_60_400"] == pytest.approx(0)
    assert m["envelope_corr_300_1000"] == pytest.approx(1)


def test_silent_synth_is_not_perfect_spectrum_or_envelope():
    x = noise()
    m = tb.diagnostic_metrics(x, np.zeros_like(x), FS)["metrics"]
    assert m["rms_error_db"] < -100
    assert m["band_mae_db"] is None
    assert m["spectrum_match"] is None
    assert m["envelope_corr_4000_12000"] is None


def test_identical_signal_matches_but_modulated_mismatch_is_detectable():
    t = np.arange(int(FS * .25)) / FS
    x = noise(seconds=.25) * (1 + .7 * np.sin(2 * np.pi * 90 * t))
    same = tb.diagnostic_metrics(x, x, FS)["metrics"]
    wrong = tb.diagnostic_metrics(x, noise(9, .25), FS)["metrics"]
    assert same["band_mae_db"] == pytest.approx(0)
    assert same["spectrum_match"] == pytest.approx(100)
    assert same["envelope_corr_4000_12000"] > .99
    assert wrong["envelope_corr_4000_12000"] < .5
    assert wrong["mod_error_pp_4000_12000_60_400"] > 1


def test_seed_summaries_are_descriptive_and_skip_unavailable():
    r = tb.seed_summary([{"x": 1, "z": None}, {"x": 3, "z": np.nan}])
    assert r["x"] == dict(count=2, median=2., min=1., max=3., std=1.)
    assert r["z"] == dict(count=0, median=None, min=None, max=None, std=None)
    assert tb.seed_summary([{"x": 2}])["x"]["std"] is None


def test_spectral_fidelity_status_for_missing_short_silent_and_unresolved():
    x = noise()
    assert tb.spectral_fidelity(x, x, None, FS)["correction_status"] == "uncorrected"
    for x in (np.zeros(5000), np.ones(10)):
        r = tb.spectral_fidelity(x, x, x, FS)
        assert r["correction_status"] == "unavailable"
        assert not r["resolved"]
        assert np.isnan(r["fine_corr"])
    r = tb.spectral_fidelity(noise(), noise(1), noise(2), FS)
    assert r["correction_status"] == "unresolved"
    assert not r["resolved"]
