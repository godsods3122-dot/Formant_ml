"""주기 단위 펄스 시각 보정 (fit.REFINE_CYCLE, MEASUREMENTS §52.267).

기존 보정은 20 ms 창·5 ms 간격·40 ms 평활이라 **주기(약 3 ms)보다 성긴 자**다 — 주기별 어긋남을 원리적으로 못 본다.
주기 경로는 성문 주기마다 배음 위상차의 기울기로 δt 를 직접 얻는다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.control import ControlTrack, default_vector
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import DEFAULT_PROFILE
from formant_ml.engine.voice import EngineConfig, VoiceEngine

FS = 48000
F0 = 200.0
T = int(FS / F0)


def _fitter(monkeypatch):
    eng = VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, residual=False), DEFAULT_PROFILE)
    v = np.tile(default_vector(), (200, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = F0
    tr["residual_mix"] = 0.0
    tr = tr.clamp()
    y = np.asarray(eng.render(tr), dtype=np.float64)
    n = len(y)
    tr.pulses = np.arange(0, n - T, T) / FS          # 일정 간격의 성문 표시
    f = CopySynthFitter(eng, y, FS, tr)
    return f, y


def _pulse_train(n, shift_per_cycle=0.0):
    """주기마다 `shift_per_cycle` 표본씩 밀린 펄스열 — 시각 어긋남의 참값을 안다."""
    x = np.zeros(n)
    c = 0
    while True:
        i = int(round(c * T + shift_per_cycle))
        if i >= n - 1:
            break
        x[i] = 1.0
        c += 1
    return x


def test_cycle_lag_recovers_a_known_constant_shift(monkeypatch):
    f, _ = _fitter(monkeypatch)
    n = f.target.shape[-1]
    shift = 3.0                                       # 표본 = 0.0625 ms
    tgt = _pulse_train(n)
    syn = _pulse_train(n, shift)
    f.target = torch.as_tensor(tgt[None, :], dtype=torch.float64)
    dt, ctr = f.cycle_lag_ms(syn)
    assert dt.size > 20, dt.size
    got = float(np.median(dt))
    assert abs(got - shift / FS * 1000.0) < 0.01, (got, shift / FS * 1000.0)
    assert ctr.size == dt.size


def test_cycle_lag_is_zero_for_an_identical_signal(monkeypatch):
    f, y = _fitter(monkeypatch)
    dt, _ = f.cycle_lag_ms(y)
    assert dt.size > 20
    assert abs(float(np.median(dt))) < 1e-6, float(np.median(dt))


def test_cycle_lag_follows_a_per_cycle_alternating_shift(monkeypatch):
    """**주기마다 다른** 어긋남을 따라가야 한다 — 기존 40 ms 평활 경로가 못 하는 바로 그것이다."""
    f, _ = _fitter(monkeypatch)
    n = f.target.shape[-1]
    tgt = _pulse_train(n)
    # **주기마다 펄스가 정확히 하나** 들어가야 한다 — 어긋남이 음수면 앞 주기로 넘어가 두 개/영 개가 된다.
    syn = np.zeros(n)
    truth = []
    c = 0
    while True:
        sh = 2.0 if c % 2 == 0 else 6.0
        i = int(round(c * T + sh))
        if i >= n - 1:
            break
        syn[i] = 1.0
        truth.append(sh)
        c += 1
    f.target = torch.as_tensor(tgt[None, :], dtype=torch.float64)
    monkeypatch.setattr(F, "REFINE_CYCLE_MED", 1)      # 중앙값 거르기가 교대 성분을 지우지 않게
    dt, _ = f.cycle_lag_ms(syn)
    k = min(len(dt), len(truth)) - 2
    a = dt[1:k]
    b = np.asarray(truth[1:k]) / FS * 1000.0
    assert np.corrcoef(a, b)[0, 1] > 0.9, float(np.corrcoef(a, b)[0, 1])


def test_refine_uses_the_cycle_path_only_when_switched_on(monkeypatch):
    f, y = _fitter(monkeypatch)
    calls = {"cycle": 0, "window": 0}
    monkeypatch.setattr(f, "_refine_by_cycle", lambda verbose=True: calls.__setitem__("cycle", calls["cycle"] + 1) or 0.0)
    monkeypatch.setattr(f, "measure_lag_ms", lambda *a, **k: (calls.__setitem__("window", calls["window"] + 1) or (np.zeros(0), np.zeros(0))))
    f._pulse_phase = torch.zeros((1, f.target.shape[-1]), dtype=torch.float64)
    monkeypatch.setattr(F, "REFINE_CYCLE", False)
    f.refine_pulse_phase(verbose=False)
    assert (calls["cycle"], calls["window"]) == (0, 1), calls
    monkeypatch.setattr(F, "REFINE_CYCLE", True)
    f.refine_pulse_phase(verbose=False)
    assert (calls["cycle"], calls["window"]) == (1, 1), calls


def test_default_is_off():
    assert F.REFINE_CYCLE is False
