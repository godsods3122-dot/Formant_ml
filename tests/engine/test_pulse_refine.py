"""잠금 위상 되먹임 보정 (`fit.PULSE_REFINE`, docs/MEASUREMENTS §52.80).

성도 군지연 때문에 목표 폐쇄 표시와 복사 파형의 펄스 자리는 어긋난다. 그 어긋남을 창마다 재서
잠금 위상을 앞당기는 고리를 검사한다 — 재는 쪽(부호·크기)과 고치는 쪽(위상 단조성) 둘 다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.control import ControlTrack, default_vector
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import DEFAULT_PROFILE
from formant_ml.engine.voice import EngineConfig, VoiceEngine


@pytest.fixture(scope="module")
def _engine():
    return VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False),
                       DEFAULT_PROFILE)


def _track(n=400):
    v = np.tile(default_vector(), (n, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = 200.0
    tr["f1"] = 700; tr["f2"] = 1200; tr["f3"] = 2600; tr["f4"] = 3600
    tr["bw1"] = 80; tr["bw2"] = 100; tr["bw3"] = 150; tr["bw4"] = 200
    tr["residual_mix"] = 0.0
    return tr.clamp()


@pytest.mark.parametrize("shift", [-24, -7, 0, 11, 23])
def test_measure_lag_recovers_a_known_shift(_engine, shift):
    """합성을 n 표본 늦추면 잰 지연이 **양수 n 표본**이어야 한다 (부호 약속)."""
    tr = _track()
    y = np.asarray(_engine.render(tr), dtype=np.float64)
    f = CopySynthFitter(_engine, y, 48000, tr)
    moved = np.roll(y, shift)                      # shift > 0 이면 합성이 늦다
    lag, rr = f.measure_lag_ms(moved)
    good = rr > 0.8
    assert good.sum() > 10, "상관이 높은 창이 있어야 한다"
    got = np.median(lag[good])
    assert got == pytest.approx(shift / 48000 * 1000.0, abs=0.03), (shift, got)


def test_refine_moves_phase_and_keeps_it_monotone(_engine, monkeypatch):
    """보정은 위상을 실제로 움직이되 단조성을 깨지 않는다 — 엔진이 단조가 아닌 위상을 거부한다."""
    tr = _track()
    y = np.asarray(_engine.render(tr), dtype=np.float64)
    monkeypatch.setattr(F, "PULSE_LOCK", True)
    # 분석기가 붙이는 폐쇄 시각을 손으로 단다 — F0 200 Hz 이므로 5 ms 마다 하나다.
    tr.pulses = np.arange(0.0, len(y) / 48000.0 - 0.005, 0.005)
    f = CopySynthFitter(_engine, y, 48000, tr)
    assert f._pulse_phase is not None, "폐쇄 표시를 줬으면 잠금 위상이 만들어져야 한다"
    before = f._pulse_phase.clone()
    moved = f.refine_pulse_phase(verbose=False)
    after = f._pulse_phase
    assert after.shape == before.shape
    assert torch.isfinite(after).all()
    d = torch.diff(after[0])
    assert float(d.min()) >= -1e-12, "위상은 단조여야 한다"
    assert moved >= 0.0


def test_refine_is_a_no_op_without_lock(_engine):
    """잠금이 꺼져 있으면 보정은 아무것도 하지 않는다 (자유 위상을 건드리지 않는다)."""
    tr = _track()
    y = np.asarray(_engine.render(tr), dtype=np.float64)
    f = CopySynthFitter(_engine, y, 48000, tr)
    assert f._pulse_phase is None
    assert f.refine_pulse_phase(verbose=False) == 0.0
