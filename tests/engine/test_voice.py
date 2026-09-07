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
    assert np.abs(chunks - off).max() < 1e-4 * np.abs(off).max()


def test_tap_makes_a_short_dip_with_continuous_voicing(eng):
    y = eng.render(phones.ara())
    e = _env_db(y)
    v = e[10:20].mean()                       # 첫 모음 정상부
    dip = e[23:31].min()
    assert 2.0 < v - dip < 12.0               # 실측 3.5~5.5 dB 골 (여유 있게)
    assert (e[23:31] > v - 25).all()          # 발성이 끊기지 않는다


def test_lateral_onset_is_quieter_than_vowel_but_voiced(eng):
    """설측 유지부가 뒤 모음보다 프로파일의 `lateral.level_db` 만큼 조용하다.

    창을 프레임 번호로 박아 두면 안 된다. 지속시간이 화자마다 다르고, 그 창이
    엇나가면 테스트가 **엉뚱한 이유로** 통과한다(실제로 겪었다: `f0_target` 위에
    폐압 결합이 곱해져 F0 가 18 % 높던 시절, 하모닉이 F1 을 비껴가 유지부가 우연히
    조용했고 그걸로 통과하고 있었다). 창은 제어열에서 읽는다.
    """
    tr = phones.ra()
    y = eng.render(tr)
    hop = eng.cfg.hop
    mix, ps = tr["lat_mix"], tr["p_sub"]
    after = np.flatnonzero(mix > 0.1)
    hold = (mix > 0.9) & (ps > 0)
    vowel = np.zeros(tr.n_frames, dtype=bool)
    vowel[after[-1] + 40:] = True
    vowel &= ps > 0

    def rms_db(m):
        i = np.flatnonzero(m)
        seg = y[i[0] * hop:min((i[-1] + 1) * hop, len(y))]
        return 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12)

    got = rms_db(hold) - rms_db(vowel)
    want = eng.profile.lateral["level_db"] if eng.profile else -2.0
    assert abs(got - want) < 2.5, (got, want)
    assert rms_db(hold) > rms_db(vowel) - 25.0          # 발성이 끊기지 않는다


def test_keyframe_track_defaults_before_first_key_and_zero_hold():
    tr = track_from_keyframes([dict(t=0.1, lat_z1=3000.0, f1=500.0),
                               dict(t=0.2, lat_z1=0.0, f1=800.0)], seconds=0.3)
    assert tr["lat_z1"][50] == 0.0 and tr["lat_z1"][150] == 3000.0 and tr["lat_z1"][250] == 0.0
    assert 500 < tr["f1"][150] < 800 and tr["f1"][50] == 0.0
    assert abs(tr["f1"][250] - 800.0) < 1e-6
