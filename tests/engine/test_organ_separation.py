"""기관별 분리 — 폐쇄 관측·앞공동 관측·소유 가지 보정 (MEASUREMENTS §52.353~360)."""
import numpy as np
import pytest
import torch

from formant_ml.engine import analyze as A
from formant_ml.engine import tract as T
from formant_ml.engine.control import ControlTrack, INDEX, default_vector
from formant_ml.engine.profile import DEFAULT_PROFILE
from formant_ml.engine.voice import EngineConfig, VoiceEngine

FS = 48000
HOP = 48                      # 1 ms


def _track(n=400):
    v = np.tile(default_vector(), (n, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = 200.0
    tr["f1"] = 700; tr["f2"] = 1200; tr["f3"] = 2600; tr["f4"] = 3600
    tr["a_c"] = 3.0
    tr["residual_mix"] = 0.0
    return tr.clamp()


def _audio_with_closure(n=400, close=(120, 223), burst=225):
    """말 – 무음(폐쇄) – 파열 – 말. 폐쇄는 파열 **직전까지** 이어진다 (실제 파열음의 꼴)."""
    rng = np.random.default_rng(0)
    y = 0.2 * rng.standard_normal(n * HOP)
    y[close[0] * HOP:close[1] * HOP] *= 1e-4          # 폐쇄 = 무음
    y[burst * HOP:burst * HOP + 60] += 3.0            # 파열
    return y


# ---------------------------------------------------------------- 폐쇄 관측

def test_close_stops_closes_a_c_over_the_silence_before_a_burst():
    tr = _track()
    tr.bursts = np.array([225])
    y = _audio_with_closure()
    n = A.close_stops(tr, y, FS, HOP)
    assert n > 0
    ac = tr.values[:, INDEX["a_c"]]
    # 폐쇄 한복판은 닫히고, 말하는 구간은 그대로다
    assert ac[170] < 0.1, ac[170]
    assert ac[50] > 2.0, ac[50]
    assert ac[350] > 2.0, ac[350]


def test_close_stops_also_cuts_the_oral_flow():
    """**협착만 조이면 폐쇄가 아니라 마찰이다** (MEASUREMENTS §52.361).

    `oral_open` 이 전 발화 1.000 이라 유량이 그대로면, `a_c` 를 조일수록 레이놀즈 수가 올라
    마찰이 **커진다**. 실제로 `N2` 에서 적합기가 파열 셋의 `a_c` 를 2.4~2.9 로 도로 열었다.
    """
    tr = _track()
    tr.bursts = np.array([225])
    A.close_stops(tr, _audio_with_closure(), FS, HOP)
    oo = tr.values[:, INDEX["oral_open"]]
    assert oo[170] < 0.1, oo[170]          # 폐쇄 한복판은 유량이 끊긴다
    assert oo[50] > 0.9, oo[50]            # 말하는 구간은 그대로


def test_close_stops_is_a_ramp_not_a_step():
    tr = _track()
    tr.bursts = np.array([225])
    A.close_stops(tr, _audio_with_closure(), FS, HOP)
    ac = tr.values[:, INDEX["a_c"]]
    # 인접 프레임 사이 도약이 구간 전체 낙차의 절반을 넘지 않는다
    d = np.abs(np.diff(ac))
    assert d.max() < 0.5 * (ac.max() - ac.min()), d.max()


def test_close_stops_does_nothing_without_bursts():
    tr = _track()
    before = tr.values.copy()
    assert A.close_stops(tr, _audio_with_closure(), FS, HOP) == 0
    assert np.array_equal(tr.values, before)


def test_close_stops_ignores_a_gap_that_is_too_short():
    tr = _track()
    tr.bursts = np.array([225])
    y = _audio_with_closure(close=(219, 223))         # 4 ms < STOP_CLOSE_MIN_MS
    assert A.close_stops(tr, y, FS, HOP) == 0


# ------------------------------------------------------------ 앞공동 관측

def test_observe_front_len_recovers_a_quarter_wave_peak():
    """8500 Hz 에 마루가 있는 잡음 -> front_len ≈ 34000/(4·8500) = 1.0 cm."""
    from scipy.signal import butter, sosfilt
    n = 300
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n * HOP)
    sos = butter(4, [8000 / (FS / 2), 9000 / (FS / 2)], btype="band", output="sos")
    y = sosfilt(sos, x)
    y = y / (np.abs(y).max() + 1e-9)
    tr = _track(n)
    k = A.observe_front_len(tr, y, FS, HOP)
    assert k > 0.5 * n, k
    fl = np.median(tr.values[:, INDEX["front_len"]])
    assert abs(fl - 1.0) < 0.25, fl


def test_observe_front_len_refuses_a_low_frequency_peak(monkeypatch):
    """**기식은 관측하지 않는다** (MEASUREMENTS §52.360).

    비주기적이지만 정점이 1.4 kHz 인 소리 — ㅎ 의 꼴이다. 그 정점은 앞공동이 아니라 모음
    포먼트이므로 관측하면 안 된다. `N1` 은 이것을 관측해 `front_len` 중앙을 5.43 cm 로 만들었고
    6~10 kHz 가 83.04 → 73.64 로 무너졌다.
    """
    from scipy.signal import butter, sosfilt
    n = 300
    rng = np.random.default_rng(2)
    sos = butter(4, [1300 / (FS / 2), 1600 / (FS / 2)], btype="band", output="sos")
    y = sosfilt(sos, rng.standard_normal(n * HOP))
    y = y / (np.abs(y).max() + 1e-9)
    tr = _track(n)
    tr["front_len"] = 0.92
    before = tr.values[:, INDEX["front_len"]].copy()
    assert A.observe_front_len(tr, y, FS, HOP) == 0
    assert np.allclose(tr.values[:, INDEX["front_len"]], before)


def test_observe_front_len_does_not_interpolate_across_a_long_gap():
    """관측이 드문드문하면 **긴 빈틈은 화자 기본값으로 되돌린다** (§52.360)."""
    from scipy.signal import butter, sosfilt
    n = 600
    rng = np.random.default_rng(3)
    sos = butter(4, [8000 / (FS / 2), 9000 / (FS / 2)], btype="band", output="sos")
    y = sosfilt(sos, rng.standard_normal(n * HOP))
    y = y / (np.abs(y).max() + 1e-9)
    # 앞 100 ms 만 마찰, 그 뒤는 **유성**(주기) — 관측이 끊기는 실제 꼴이다
    t = np.arange((n - 100) * HOP) / FS
    vo = sum(np.cos(2 * np.pi * 200.0 * k * t) / k for k in range(1, 40))
    y[100 * HOP:] = 0.5 * vo / (np.abs(vo).max() + 1e-9)
    tr = _track(n)
    tr["front_len"] = 0.92
    k = A.observe_front_len(tr, y, FS, HOP)
    assert k > 0
    fl = tr.values[:, INDEX["front_len"]]
    assert abs(fl[500] - 0.92) < 0.05, fl[500]         # 먼 곳은 기본값 그대로


def test_observe_front_len_leaves_the_track_alone_when_nothing_is_observed():
    tr = _track(50)
    before = tr.values.copy()
    assert A.observe_front_len(tr, np.zeros(10), FS, HOP) == 0
    assert np.array_equal(tr.values, before)


# ------------------------------------------------------- 소유 가지 보정

@pytest.fixture
def _engine():
    return VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, residual=False),
                       DEFAULT_PROFILE)


def test_hf_owned_changes_the_render_only_when_a_correction_is_live(_engine, monkeypatch):
    """보정이 꺼져 있으면 `HF_OWNED` 는 **항등**이어야 한다 (구조만 바꾸고 소리는 그대로)."""
    monkeypatch.setattr(T, "HF_POLES", False)
    monkeypatch.setattr(T, "HF_ZEROS", False)
    tr = _track(200)
    a = _engine.render(tr)
    monkeypatch.setattr(T, "HF_OWNED", True)
    b = _engine.render(tr)
    assert np.allclose(a, b, atol=1e-9), float(np.abs(a - b).max())




def test_hf_owned_default_is_off():
    assert T.HF_OWNED is False
    assert A.STOP_CLOSE is False and A.FRONT_LEN_OBS is False
