"""**사각지대 붙잡기** (`fit.HOLD_GAP`, MEASUREMENTS §52.339).

유성도 마찰도 아닌 구간은 세 자 어디에도 안 걸린다 — 조화 구조가 없어 포락 손실이 포먼트 자리를
거의 못 보고, `_hold` 의 마찰 마스크 밖이고, 움직임 예산은 유성만 본다. 사용자가 짚은 스윕 넷이
전부 그 자리(마찰 시작 9~44 ms 앞)였다. 여기서는 공명의 **모양**을 분석 궤적으로 갈아 끼우고
**수준**만 남긴다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.control import INDEX, PARAMS, ControlTrack


class _Stub:
    """`CopySynthFitter` 에서 사각지대 채비가 쓰는 부분만 흉내 낸다."""

    # static/class 메서드는 다시 감싼다 — 그냥 대입하면 인스턴스 메서드로 재바인딩된다.
    _runs = staticmethod(F.CopySynthFitter._runs)
    _ramp_mask = staticmethod(F.CopySynthFitter._ramp_mask)
    _edge_dist = staticmethod(F.CopySynthFitter._edge_dist)
    _seg_ops = staticmethod(F.CopySynthFitter._seg_ops)
    _gap_setup = F.CopySynthFitter._gap_setup
    _gap_smooth = F.CopySynthFitter._gap_smooth
    _gap = F.CopySynthFitter._gap

    def __init__(self, voiced, fricative, obs):
        n = len(voiced)
        vals = np.tile(np.array([p.default for p in PARAMS.values()], float), (n, 1))
        for name, series in obs.items():
            vals[:, INDEX[name]] = series
        self.track = ControlTrack(vals, 1.0)
        self.track.voiced = np.asarray(voiced, bool)
        self.track.fricative = np.asarray(fricative, bool)
        self.base = torch.as_tensor(vals, dtype=torch.float64)
        self.names = list(obs)
        self._gap_setup(np.asarray(fricative, bool), torch.float64, torch.device("cpu"))

    def apply(self, cols):
        return self._gap(torch.as_tensor(cols, dtype=torch.float64))


def _masks(n=300, fric=(200, 240)):
    """앞은 유성, 가운데가 사각지대, 그 뒤가 마찰인 흔한 모양."""
    voi = np.zeros(n, bool)
    voi[:150] = True
    fri = np.zeros(n, bool)
    fri[fric[0]:fric[1]] = True
    return voi, fri


def test_default_is_off():
    assert F.HOLD_GAP is False
    assert F.HOLD_GAP_RAMP_MS == pytest.approx(8.0)
    assert "a_c" not in F.HOLD_GAP_PARAMS          # 세기는 사각지대에서 변해야 한다


def test_short_run_still_reaches_the_plateau():
    """**짧은 구간도 고원에 닿는다** — 고정 램프는 7~17 ms 구간에서 0.47 까지밖에 안 올랐다."""
    mask = np.zeros(40, bool)
    mask[10:23] = True                              # 13 프레임: 예전 램프 합(5+15)보다 짧다
    d_in, d_out = F.CopySynthFitter._edge_dist(mask)
    m = F.CopySynthFitter._ramp_mask(mask, d_in, d_out, 5.0, 15.0)
    assert m.max() == pytest.approx(1.0)
    assert m[9] == 0.0 and m[23] == 0.0             # 구간 밖은 그대로 0


def test_ramp_is_unchanged_for_long_runs():
    mask = np.zeros(120, bool)
    mask[10:110] = True                             # 100 프레임: 램프를 줄일 까닭이 없다
    d_in, d_out = F.CopySynthFitter._edge_dist(mask)
    m = F.CopySynthFitter._ramp_mask(mask, d_in, d_out, 5.0, 15.0)
    assert m[14] == pytest.approx(1.0)              # 시작 5 프레임 뒤 고원
    assert m[12] == pytest.approx(3 / 5)            # 램프 안은 선형
    assert m[107] == pytest.approx(3 / 15)          # 끝 램프는 더 길다


def test_a_frozen_observation_freezes_us():
    """관측이 정지면 우리도 정지한다 — 2.717~2.784 s (관측 f1 범위 0 Hz) 의 경우다."""
    voi, fri = _masks()
    n = len(voi)
    obs = {"f1": np.full(n, 600.0)}
    st = _Stub(voi, fri, obs)
    cols = np.full((n, 1), 600.0)
    cols[160:195, 0] = np.linspace(600.0, 1300.0, 35)      # 사각지대에서 훑는 궤적
    out = st.apply(cols).numpy()[:, 0]
    inner = out[165:190]                                    # 램프 8 프레임 안쪽
    assert np.ptp(inner) < 1.0                              # 관측이 상수이므로 우리도 상수
    assert np.ptp(cols[165:190, 0]) > 400.0                 # 들어간 것은 크게 움직였다


def test_a_moving_observation_is_followed_in_shape():
    """관측이 실제로 전이하면 같은 모양으로 따라간다 — 2.055~2.129 s 는 관측 f2 범위가 1996 Hz 다."""
    voi, fri = _masks()
    n = len(voi)
    ramp = np.full(n, 1500.0)
    ramp[160:200] = np.linspace(1500.0, 2500.0, 40)
    cols = np.full((n, 1), 1500.0)                          # 우리는 아무것도 안 하고 있다
    k = slice(168, 192)
    out = _Stub(voi, fri, {"f2": ramp}).apply(cols).numpy()[:, 0]
    flat = _Stub(voi, fri, {"f2": np.full(n, 1500.0)}).apply(cols).numpy()[:, 0]
    assert np.all(np.diff(out[k]) > 0.0)                    # 같은 방향으로 전이한다
    # 기준 모양은 깎아 쓰므로 기울기가 눕는다 — 모양의 상관과 **정지 관측과의 차이**로 본다.
    a = np.log(out[k]) - np.log(out[k]).mean()
    b = np.log(ramp[k]) - np.log(ramp[k]).mean()
    assert np.corrcoef(a, b)[0, 1] > 0.99
    assert np.ptp(out[k]) > 50.0 * max(np.ptp(flat[k]), 1e-6)   # 정지 관측은 정지시킨다


def test_level_stays_free():
    """**수준은 적합기의 것이다** — 모양만 갈아 끼우고 구간 평균 오프셋은 남긴다."""
    voi, fri = _masks()
    n = len(voi)
    st = _Stub(voi, fri, {"f1": np.full(n, 600.0)})
    for lvl in (600.0, 900.0):
        cols = np.full((n, 1), lvl)
        out = st.apply(cols).numpy()[:, 0]
        assert out[175] == pytest.approx(lvl, rel=1e-9)     # 올린 수준이 그대로 산다


def test_voiced_and_fricative_frames_are_untouched():
    voi, fri = _masks()
    n = len(voi)
    st = _Stub(voi, fri, {"f1": np.full(n, 600.0)})
    cols = np.full((n, 1), 600.0)
    cols[:150, 0] = np.linspace(500.0, 700.0, 150)          # 유성 구간
    cols[200:240, 0] = np.linspace(800.0, 900.0, 40)        # 마찰 구간
    out = st.apply(cols).numpy()[:, 0]
    assert np.allclose(out[:150], cols[:150, 0])
    assert np.allclose(out[200:240], cols[200:240, 0])


def test_the_ramp_is_what_keeps_the_edge_from_stepping():
    """가장자리 계단을 막는 것이 램프다 — 램프를 없애면 같은 자리에 큰 점프가 선다."""
    voi, fri = _masks()
    n = len(voi)
    cols = np.full((n, 1), 600.0)
    cols[150:200, 0] = np.linspace(600.0, 1400.0, 50)       # 사각지대에서 800 Hz 를 훑는다
    jumps = {}
    for ramp in (8.0, 0.5):
        F.HOLD_GAP_RAMP_MS = ramp                           # conftest 가 시험 뒤 되돌린다
        st = _Stub(voi, fri, {"f1": np.full(n, 600.0)})
        out = st.apply(cols).numpy()[:, 0]
        jumps[ramp] = abs(out[150] - cols[149, 0])
    assert jumps[8.0] < 0.1 * np.ptp(cols[150:200, 0])      # 훑던 폭의 10 % 아래
    assert jumps[0.5] > 3.0 * jumps[8.0]                    # 램프가 없으면 계단이 선다


def test_no_gap_frames_means_no_change():
    n = 200
    voi = np.ones(n, bool)                                  # 전부 유성 — 사각지대가 없다
    fri = np.zeros(n, bool)
    st = _Stub(voi, fri, {"f1": np.full(n, 600.0)})
    cols = np.random.default_rng(0).uniform(500.0, 800.0, (n, 1))
    out = st.apply(cols).numpy()
    assert np.allclose(out, cols)


def test_gradient_flows_to_the_level():
    """미분이 흐른다 — 재파라미터화이므로 수준을 통해 기울기가 들어와야 한다."""
    voi, fri = _masks()
    n = len(voi)
    st = _Stub(voi, fri, {"f1": np.full(n, 600.0)})
    x = torch.full((n, 1), 600.0, dtype=torch.float64, requires_grad=True)
    st._gap(x)[175, 0].backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


def test_the_reference_shape_is_smoothed_first():
    """**관측을 그대로 믿지 않는다** — 사각지대엔 조화 구조가 없어 추적기가 과도음에 공진을 맞춘다.

    실측 2.055~2.130 s 의 관측 f2 는 프레임당 26.3 Hz 로 흐르는데 목표 소리는 그 자리에서 조용하다.
    """
    voi, fri = _masks()
    n = len(voi)
    rng = np.random.default_rng(7)
    jitter = np.full(n, 1800.0)
    jitter[150:200] += rng.normal(0.0, 120.0, 50)           # 추적기의 요동
    st = _Stub(voi, fri, {"f2": jitter})
    cols = np.full((n, 1), 1800.0)
    out = st.apply(cols).numpy()[:, 0]
    rough = np.abs(np.diff(jitter[165:190])).mean()
    ours = np.abs(np.diff(out[165:190])).mean()
    assert ours < 0.25 * rough                              # 요동을 그대로 베끼지 않는다


def test_smoothing_can_be_turned_off():
    voi, fri = _masks()
    n = len(voi)
    rng = np.random.default_rng(7)
    shape = np.full(n, 1800.0)
    shape[150:200] += rng.normal(0.0, 120.0, 50)
    F.HOLD_GAP_SMOOTH_MS = 0.0                              # conftest 가 되돌린다
    F.HOLD_GAP_SMOOTH_FRAC = 0.0                            # 몫 쪽도 함께 꺼야 꺼진다
    st = _Stub(voi, fri, {"f2": shape})
    out = st.apply(np.full((n, 1), 1800.0)).numpy()[:, 0]
    k = slice(168, 192)
    a = np.log(out[k]) - np.log(out[k]).mean()
    b = np.log(shape[k]) - np.log(shape[k]).mean()
    assert np.allclose(a, b, atol=1e-9)                     # 끄면 관측 모양 그대로


def test_smoothing_stays_inside_the_run():
    """구간 밖 값이 새어 들어오면 수준이 끌려간다 — 가장자리는 자기 값으로 채운다."""
    voi, fri = _masks()
    n = len(voi)
    shape = np.full(n, 1800.0)
    shape[:150] = 400.0                                     # 유성 쪽은 아주 낮다
    st = _Stub(voi, fri, {"f2": shape})
    out = st.apply(np.full((n, 1), 1800.0)).numpy()[:, 0]
    assert out[152] == pytest.approx(1800.0, rel=1e-6)      # 400 이 새어 들어오지 않았다


def test_a_longer_gap_is_smoothed_harder():
    """**구간이 길수록 세게 깎는다** — 손실이 포먼트 자리를 못 보는 시간이 길수록 추적기를 덜 믿는다."""
    rng = np.random.default_rng(3)
    rough = {}
    for length in (30, 90):
        n = 400
        voi = np.zeros(n, bool)
        voi[:150] = True
        voi[150 + length:] = True                           # 사각지대 길이만 바꾼다
        fri = np.zeros(n, bool)
        shape = np.full(n, 1800.0)
        shape[150:150 + length] += rng.normal(0.0, 100.0, length)
        st = _Stub(voi, fri, {"f2": shape})
        out = st.apply(np.full((n, 1), 1800.0)).numpy()[:, 0]
        k = slice(160, 150 + length - 10)
        rough[length] = np.abs(np.diff(out[k])).mean() / np.abs(np.diff(shape[k])).mean()
    assert rough[90] < rough[30]                            # 긴 구간이 더 매끈해진다


def test_frac_and_floor_can_both_be_turned_off():
    voi, fri = _masks()
    n = len(voi)
    rng = np.random.default_rng(7)
    shape = np.full(n, 1800.0)
    shape[150:200] += rng.normal(0.0, 120.0, 50)
    F.HOLD_GAP_SMOOTH_MS = 0.0
    F.HOLD_GAP_SMOOTH_FRAC = 0.0                            # conftest 가 되돌린다
    st = _Stub(voi, fri, {"f2": shape})
    out = st.apply(np.full((n, 1), 1800.0)).numpy()[:, 0]
    k = slice(168, 192)
    a = np.log(out[k]) - np.log(out[k]).mean()
    b = np.log(shape[k]) - np.log(shape[k]).mean()
    assert np.allclose(a, b, atol=1e-9)
