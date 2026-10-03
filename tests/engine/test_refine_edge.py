"""잠금 위상 보정의 지연 추정이 **탐색 끝에 걸린 창**을 표시한다 (MEASUREMENTS §52.256).

`measure_lag_ms` 는 ±`REFINE_MAX_MS` 안에서 상관 최댓값을 찾는다. 참 지연이 그 밖이면 최댓값이 끝에 붙고,
부표본 보정 없이 ±0.6 ms 가 유효한 지연처럼 들어갔다. 끝 걸림을 따로 세야 그것이 보정 흔들림의 원인인지 가를 수 있다.
"""
import types

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from formant_ml.engine import fit as F
from formant_ml.engine.fit import CopySynthFitter

FS = 48000


def _pair(delay_ms: float, n: int = FS // 2, lp_hz: float = 1000.0):
    rng = np.random.default_rng(0)
    x = sosfiltfilt(butter(4, lp_hz, "lp", fs=FS, output="sos"), rng.standard_normal(n + 4000))
    d = int(round(delay_ms * FS / 1000.0))
    t = x[2000:2000 + n]
    y = x[2000 - d:2000 - d + n]            # 합성이 d 표본 늦다 -> 양의 지연
    return t, y


def _measure(delay_ms, lp_hz=1000.0):
    t, y = _pair(delay_ms, lp_hz=lp_hz)
    obj = types.SimpleNamespace(target=torch.as_tensor(np.ascontiguousarray(t)[None, :]), fs=FS)
    lag, rr = CopySynthFitter.measure_lag_ms(obj, y)
    return lag, rr, obj._lag_edge


def test_in_range_delay_is_measured_without_edge_hits():
    lag, rr, edge = _measure(0.3)
    assert edge.size == lag.size and edge.size > 20
    assert not edge.any()
    assert abs(float(np.median(lag)) - 0.3) < 0.03, float(np.median(lag))


def test_out_of_range_delay_is_flagged_at_the_edge():
    """상관이 단조로 줄어드는 신호(150 Hz 저역)에서 범위 밖 지연은 끝에 붙는다."""
    lag, rr, edge = _measure(1.5, lp_hz=150.0)
    assert edge.mean() > 0.9, edge.mean()
    edge_ms = int(F.REFINE_MAX_MS * FS / 1000.0) / FS * 1000.0      # 탐색 끝 = 정수 표본 (0.6 ms -> 28 표본 = 0.583 ms)
    assert abs(float(np.median(lag)) - edge_ms) < 1e-9, float(np.median(lag))   # 끝 값이 그대로 들어간다 — 그래서 표시가 필요하다


def test_out_of_range_delay_can_also_land_on_a_false_interior_lobe():
    """상관이 **진동**하면(1 kHz 저역 — 음성의 F1 대역과 같은 성질) 범위 밖 지연이 끝이 아니라 **안쪽의 가짜 봉우리**에
    선다 — 주기를 건너뛴 추정인데 끝 걸림 표시로는 안 잡힌다. 그 양식이 있음을 못 박아 둔다 (§52.256)."""
    lag, rr, edge = _measure(1.5, lp_hz=1000.0)
    inner = ~edge
    assert inner.mean() > 0.3, inner.mean()
    assert np.all(np.abs(lag[inner] - 1.5) > 0.5)          # 안쪽 값은 참 지연(1.5 ms)과 멀다


def _fake(lag, rr, edge, n=FS):
    obj = types.SimpleNamespace(fs=FS, _pulse_phase=torch.as_tensor(np.linspace(0.0, 200.0, n)[None, :]))

    def measure():
        obj._lag_edge = edge.copy()
        return lag.copy(), rr.copy()

    obj.measure_lag_ms = measure
    return obj


def _edge_block():
    m = 190
    lag = np.zeros(m); rr = np.full(m, 0.95); edge = np.zeros(m, dtype=bool)
    edge_ms = int(F.REFINE_MAX_MS * FS / 1000.0) / FS * 1000.0
    lag[80:103] = edge_ms; rr[80:103] = 0.65; edge[80:103] = True      # `ihdmp0` 450~560 ms 와 같은 모양
    return lag, rr, edge


def test_edge_windows_are_dropped_so_they_do_not_move_the_pulses(monkeypatch):
    """끝 걸림 창만으로 된 +0.58 ms 덩어리는 **보정을 움직이지 않아야** 한다 (§52.256)."""
    monkeypatch.setattr(F, "REFINE_DUMP", "")
    monkeypatch.setattr(F, "REFINE_DROP_EDGE", True)
    lag, rr, edge = _edge_block()
    moved = CopySynthFitter.refine_pulse_phase(_fake(lag, rr, edge), verbose=False)
    assert moved < 0.01, moved


def test_legacy_switch_keeps_the_edge_windows(monkeypatch):
    monkeypatch.setattr(F, "REFINE_DUMP", "")
    monkeypatch.setattr(F, "REFINE_DROP_EDGE", False)
    lag, rr, edge = _edge_block()
    moved = CopySynthFitter.refine_pulse_phase(_fake(lag, rr, edge), verbose=False)
    assert moved > 0.1, moved


def test_interior_delays_still_move_the_pulses(monkeypatch):
    """끝이 아닌 창의 지연은 그대로 보정한다 — 고침이 보정 자체를 끄면 안 된다."""
    monkeypatch.setattr(F, "REFINE_DUMP", "")
    monkeypatch.setattr(F, "REFINE_DROP_EDGE", True)
    lag, rr, edge = _edge_block()
    lag[80:103] = 0.3; edge[:] = False; rr[80:103] = 0.95
    moved = CopySynthFitter.refine_pulse_phase(_fake(lag, rr, edge), verbose=False)
    assert moved > 0.05, moved
