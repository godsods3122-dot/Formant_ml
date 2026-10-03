"""적합 단계 함수의 계약 (`engine/pipeline.py`·`engine/transients.py`, MEASUREMENTS §52.470).

알려진 위치의 사건을 담은 합성 신호로 각 단계가 약속한 일만 하는지 본다 — 옛 결함을 되살리면 실패하게.
"""
import numpy as np
import pytest
import soundfile as sf

from formant_ml.engine import pipeline as pl
from formant_ml.engine import transients as tr

FS = 48000


def _voice(dur=1.0, f0=200.0, seed=0):
    """성문 펄스열을 공명기 둘로 거른 모음 비슷한 신호 + 약한 잡음."""
    from scipy.signal import lfilter
    g = np.random.default_rng(seed)
    n = int(dur * FS)
    x = np.zeros(n)
    x[(np.arange(0, dur, 1 / f0) * FS).astype(int)] = 1.0
    for f, bw in ((700, 90), (1800, 120), (3000, 200)):
        r = np.exp(-np.pi * bw / FS)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r], x)
    x = x / np.abs(x).max() * 0.3
    return x + 1e-4 * g.standard_normal(n)


def _click(n=48, seed=1):
    g = np.random.default_rng(seed)
    from scipy.signal import butter, sosfiltfilt
    c = g.standard_normal(n) * np.exp(-np.arange(n) / 10.0)
    return sosfiltfilt(butter(2, 1500, "hp", fs=FS, output="sos"), c)


# ------------------------------------------------------------ 자동 클릭 규칙
def test_auto_rule_finds_an_isolated_click_and_not_the_glottal_bursts():
    x = _voice()
    t_click = 0.6137
    i = int(t_click * FS)
    x[i:i + 48] += 0.05 * _click()
    got = tr.auto_click_times(x, FS)
    assert any(abs(t - t_click) < 0.002 for t in got), got
    # 주기마다 되풀이되는 성문 솟음(5 ms 간격)은 안 잡는다
    assert all(abs(t - t_click) < 0.002 for t in got), got


def test_auto_rule_skips_a_train_of_transients():
    """±15 ms 안에 비슷한 세기의 다른 튐이 있으면(creak·크래클) 고립 클릭이 아니다."""
    x = _voice()
    for k, t in enumerate((0.40, 0.408, 0.416)):
        i = int(t * FS)
        x[i:i + 48] += 0.05 * _click(seed=k + 3)
    assert not any(0.39 < t < 0.43 for t in tr.auto_click_times(x, FS))


# ------------------------------------------------------------ 읽기·정리
def test_load_clean_keeps_positions_and_fills_only_the_click(tmp_path):
    x = _voice()
    t_click = 0.6137
    i = int(t_click * FS)
    y = x.copy()
    y[i:i + 48] += 0.05 * _click()
    p = tmp_path / "a.wav"
    sf.write(p, y, FS, subtype="DOUBLE")
    c = pl.load_clean(str(p), declick_at=f"{t_click}", no_denoise=True, log=lambda m: None)
    assert c.sr == FS and len(c.y) == len(y)
    assert len(c.clicks) == 1
    s0, s1, _ = c.clicks[0]
    assert abs(0.5 * (s0 + s1) / FS - t_click) < 0.003
    far = np.r_[0:max(0, s0 - 2000), min(len(y), s1 + 2000):len(y)]
    assert np.abs(c.y[far] - y[far]).max() < 1e-12            # 클릭 둘레 밖은 표본까지 그대로
    # 메우기는 클릭 밑의 **목소리를 되찾는다** — 클릭을 넣기 전 신호와의 오차가 클릭 자체보다 훨씬 작다
    w = slice(s0, s1)
    err_click = float(np.sum((y[w] - x[w]) ** 2))
    err_fill = float(np.sum((c.y[w] - x[w]) ** 2))
    assert err_fill < 0.1 * err_click, (err_fill, err_click)


def test_load_clean_without_anything_is_the_file(tmp_path):
    y = _voice()
    p = tmp_path / "b.wav"
    sf.write(p, y, FS, subtype="DOUBLE")
    c = pl.load_clean(str(p), no_denoise=True, log=lambda m: None)
    assert np.array_equal(c.y, y) and np.array_equal(c.y_raw, y) and c.clicks == []


# ------------------------------------------------------------ 모음 시작
class _Tr:
    def __init__(self, n, bursts, voiced):
        self.frame_ms = 1.0
        self.bursts = np.asarray(bursts, int)
        self.voiced = np.asarray(voiced, bool)
        self.n_frames = n


class _Seg:
    def __init__(self, audio):
        self.audio = audio


def _onset_case(loud_voiced_before=False):
    n = 300
    y = np.zeros(int(0.3 * FS))
    y[int(0.05 * FS):] = _voice(0.25)                       # 50 ms 에 단단한 시작
    if loud_voiced_before:
        y[int(0.01 * FS):int(0.04 * FS)] = _voice(0.03)     # 앞에 말 수준의 유성
    voiced = np.zeros(n, bool)
    voiced[52:] = True
    if loud_voiced_before:
        voiced[10:40] = True
    return _Tr(n, [50, 200], voiced), _Seg(y), y


def test_vowel_onset_removes_only_the_first_burst_of_a_vowel_initial_utterance():
    t, s, y = _onset_case()
    assert pl.vowel_onset_filter(t, s, y, FS, "어느게 시청자꺼야", 0.0, log=lambda m: None) == 50
    assert list(t.bursts) == [200]


def test_vowel_onset_keeps_a_consonant_initial_utterance():
    t, s, y = _onset_case()
    assert pl.vowel_onset_filter(t, s, y, FS, "커피 이만큼", 0.0, log=lambda m: None) is None
    assert list(t.bursts) == [50, 200]


def test_vowel_onset_keeps_the_burst_when_speech_came_before():
    t, s, y = _onset_case(loud_voiced_before=True)
    assert pl.vowel_onset_filter(t, s, y, FS, "어느게", 0.0, log=lambda m: None) is None
    assert list(t.bursts) == [50, 200]


def test_first_syllable_initial():
    assert pl.first_syllable_is_vowel("  어느게")[0] and not pl.first_syllable_is_vowel("토끼가")[0]
    assert pl.first_syllable_is_vowel(None) == (False, None)


# ------------------------------------------------------------ 되살리기
def test_restore_state_restores_the_phase_and_marks_it(tmp_path, monkeypatch):
    import torch
    from formant_ml.engine import fit as F
    from formant_ml.engine.control import ControlTrack, default_vector
    from formant_ml.engine.fit import CopySynthFitter
    from formant_ml.engine.profile import DEFAULT_PROFILE
    from formant_ml.engine.voice import EngineConfig, VoiceEngine
    eng = VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, residual=False), DEFAULT_PROFILE)
    v = np.tile(default_vector(), (200, 1))
    trk = ControlTrack(v, 1.0)
    trk["p_sub"] = 7.0; trk["adduction"] = 0.6; trk["f0_target"] = 200.0; trk["residual_mix"] = 0.0
    trk = trk.clamp()
    y = np.asarray(eng.render(trk), dtype=np.float64)
    monkeypatch.setattr(F, "PULSE_LOCK", True)
    monkeypatch.setattr(F, "HCORR_K", 0)
    trk.pulses = np.arange(0.0, len(y) / FS - 0.005, 0.005)
    f = CopySynthFitter(eng, y, FS, trk)
    ph = f._pulse_phase[0].detach().numpy() + 0.1
    stem = str(tmp_path / "prev")
    np.savez(stem + "_track.npz", pulse_phi0=np.float64(0.777), pulse_phase=ph)
    msgs = []
    pl.restore_state(f, stem, log=msgs.append)
    assert float(f.pulse_phi0.item()) == pytest.approx(0.777)
    assert np.allclose(f._pulse_phase[0].numpy(), ph)
    assert getattr(f, "_phi0_restored", False)
    assert msgs[0].strip().startswith("펄스 잠금 상태 복원") and any("적합기 전역 스칼라" in m for m in msgs)
