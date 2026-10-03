"""되살린 상태를 적합 시작 전에 덮지 않는다 (MEASUREMENTS §52.466·§52.469).

다듬기(`--init`)가 앞 판의 `pulse_phi0` 를 되살린 뒤에도 `fit_staged` 의 12 등분 오프셋 훑기가 격자점으로 덮었다 — 주기별 보정이 그 위상에 묶여
모든 다듬기가 처음 50 회에 포락 3~5 점을 잃었고, 038 은 0.3 주기 옮겨져 끝까지 69 % 였다.
"""
import math

import numpy as np
import pytest

from formant_ml.engine import fit as F
from formant_ml.engine.control import ControlTrack, default_vector
from formant_ml.engine.fit import CopySynthFitter, restore_pulse_state
from formant_ml.engine.profile import DEFAULT_PROFILE
from formant_ml.engine.voice import EngineConfig, VoiceEngine


class _Stop(Exception):
    pass


def _fitter(monkeypatch):
    eng = VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False), DEFAULT_PROFILE)
    v = np.tile(default_vector(), (300, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = 200.0
    tr["residual_mix"] = 0.0
    tr = tr.clamp()
    y = np.asarray(eng.render(tr), dtype=np.float64)
    monkeypatch.setattr(F, "PULSE_LOCK", True)
    monkeypatch.setattr(F, "HCORR_K", 0)
    tr.pulses = np.arange(0.0, len(y) / 48000.0 - 0.005, 0.005)
    return CopySynthFitter(eng, y, 48000, tr)


def _phi_at_first_stage(f, monkeypatch):
    seen = {}

    def stop(self, *a, **k):
        seen["phi0"] = float(self.pulse_phi0.item())
        raise _Stop

    monkeypatch.setattr(CopySynthFitter, "fit", stop)
    with pytest.raises(_Stop):
        f.fit_staged(global_iters=5, stage_iters=5, phase_iters=0, verbose=False, lr_global=0.01)
    return seen["phi0"]


def test_restored_phase_offset_survives_the_sweep(monkeypatch):
    f = _fitter(monkeypatch)
    want = 1.2345                                   # 격자점(−π + 2πk/12)이 아니다
    restore_pulse_state(f, {"pulse_phi0": np.float64(want), "pulse_phase": f._pulse_phase[0].detach().numpy()})
    assert _phi_at_first_stage(f, monkeypatch) == pytest.approx(want, abs=1e-12)


def test_without_restore_the_sweep_picks_a_grid_point(monkeypatch):
    """대조군 — 되살린 것이 없으면 예전처럼 12 등분 훑기로 시작점을 고른다."""
    f = _fitter(monkeypatch)
    got = _phi_at_first_stage(f, monkeypatch)
    grid = [-math.pi + 2.0 * math.pi * k / 12.0 for k in range(12)]
    assert min(abs(got - g) for g in grid) < 1e-9
