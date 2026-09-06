"""VoiceEngine 통합 — 스크립트 -> 오디오, 스트리밍 = 오프라인, 유음 지표."""
import numpy as np
import pytest
import torch

from formant_ml.engine import VoiceEngine, EngineConfig, phones
from formant_ml.engine.control import track_from_keyframes


@pytest.fixture(scope="module")
def eng():
    return VoiceEngine(EngineConfig(n_extra_formants=3))


def _env_db(y, fs=48000, win=480):
    return np.array([20 * np.log10(np.sqrt((y[i:i + win] ** 2).mean()) + 1e-9)
                     for i in range(0, len(y) - win, win)])


def test_render_is_finite_and_bounded(eng):
    y = eng.render(phones.ara())
    assert np.isfinite(y).all() and np.abs(y).max() < 20


def test_streaming_equals_offline(eng):
    tr = phones.ra()
    off = eng.render(tr)
    chunks = np.concatenate(list(eng.stream(tr, chunk_ms=17)))
    assert chunks.shape == off.shape
    assert np.abs(chunks - off).max() < 1e-3 * np.abs(off).max()


def test_tap_makes_a_short_dip_with_continuous_voicing(eng):
    y = eng.render(phones.ara())
    e = _env_db(y)
    v = e[10:20].mean()                       # 첫 모음 정상부
    dip = e[23:31].min()
    assert 2.0 < v - dip < 12.0               # 실측 3.5~5.5 dB 골 (여유 있게)
    assert (e[23:31] > v - 25).all()          # 발성이 끊기지 않는다


def test_lateral_onset_is_quieter_than_vowel_but_voiced(eng):
    y = eng.render(phones.ra())
    e = _env_db(y)
    hold = e[8:16].mean(); vowel = e[30:45].mean()
    assert 1.0 < vowel - hold < 15.0


def test_keyframe_track_defaults_before_first_key_and_zero_hold():
    tr = track_from_keyframes([dict(t=0.1, lat_z1=3000.0, f1=500.0),
                               dict(t=0.2, lat_z1=0.0, f1=800.0)], seconds=0.3)
    assert tr["lat_z1"][50] == 0.0 and tr["lat_z1"][150] == 3000.0 and tr["lat_z1"][250] == 0.0
    assert 500 < tr["f1"][150] < 800 and tr["f1"][50] == 0.0
    assert abs(tr["f1"][250] - 800.0) < 1e-6
