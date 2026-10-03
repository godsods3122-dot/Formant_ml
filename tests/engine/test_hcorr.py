"""주기별 배음 보정 (fit.HCORR_K, MEASUREMENTS §52.268).

H1~HK 의 복소 진폭을 **성문 주기마다** 적합한다. 성문 유량 미분에 더하므로 성도 응답을 그대로 탄다.
"""
import math

import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.control import ControlTrack, default_vector
from formant_ml.engine.fit import CopySynthFitter
from formant_ml.engine.profile import DEFAULT_PROFILE
from formant_ml.engine.voice import EngineConfig, VoiceEngine

FS = 48000


@pytest.fixture(scope="module")
def _base():
    eng = VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, residual=False), DEFAULT_PROFILE)
    v = np.tile(default_vector(), (150, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = 200.0
    tr["residual_mix"] = 0.0
    tr = tr.clamp()
    y = np.asarray(eng.render(tr), dtype=np.float64)
    tr.pulses = np.arange(0, len(y) - int(FS / 200.0), int(FS / 200.0)) / FS   # 성문 폐쇄 표시
    return eng, tr, y


def _fitter(monkeypatch, base, k=5):
    eng, tr, y = base
    monkeypatch.setattr(F, "PULSE_LOCK", True)
    monkeypatch.setattr(F, "HCORR_K", k)
    f = CopySynthFitter(eng, y, FS, tr)
    if f._pulse_phase is None:
        pytest.skip("이 트랙에서는 펄스 잠금 위상을 못 만든다")
    f._hcorr_prepare()
    return f


def test_off_by_default(_base, monkeypatch):
    eng, tr, y = _base
    monkeypatch.setattr(F, "PULSE_LOCK", True)
    assert F.HCORR_K == 0
    f = CopySynthFitter(eng, y, FS, tr)
    f._hcorr_prepare()
    assert f.hcorr is None
    assert f._hcorr_signal() is None


def test_prepare_allocates_one_pair_per_harmonic_per_cycle(_base, monkeypatch):
    f = _fitter(monkeypatch, _base, k=4)
    assert f.hcorr is not None and f.hcorr.requires_grad
    assert f.hcorr.shape[0] == 2 and f.hcorr.shape[1] == 4
    # 주기 수는 펄스 위상의 범위와 맞아야 한다
    u = (f._pulse_phase + f.pulse_phi0)[0] / (2 * math.pi)
    span = int(torch.floor(u.max()).item()) - int(torch.floor(u.min()).item()) + 2
    assert f.hcorr.shape[2] == span, (f.hcorr.shape, span)
    assert torch.count_nonzero(f.hcorr) == 0          # 0 에서 출발한다


def test_zero_correction_changes_nothing(_base, monkeypatch):
    f = _fitter(monkeypatch, _base)
    sig = f._hcorr_signal()
    assert sig is not None and sig.shape[-1] == f._pulse_phase.shape[-1]
    assert float(sig.detach().abs().max()) == 0.0


def test_one_harmonic_produces_that_harmonic(_base, monkeypatch):
    """한 배음의 계수만 세우면 그 배음 주파수에 에너지가 선다 — 신호가 실제로 그 모양이다."""
    f = _fitter(monkeypatch, _base, k=5)
    with torch.no_grad():
        f.hcorr[0, 2].fill_(1.0)                      # H3 의 cos 성분
    sig = f._hcorr_signal()[0].detach().numpy()
    n = len(sig)
    S = np.abs(np.fft.rfft(sig * np.hanning(n)))
    fr = np.fft.rfftfreq(n, 1.0 / FS)
    peak = fr[int(np.argmax(S))]
    assert abs(peak - 3 * 200.0) < 15.0, peak


def test_correction_reaches_the_engine_and_moves_the_audio(_base, monkeypatch):
    f = _fitter(monkeypatch, _base)
    with torch.no_grad():
        y0 = f.synth()[0].detach().clone()
        f.hcorr[0, 0].fill_(0.5)
        y1 = f.synth()[0].detach().clone()
    assert float((y1 - y0).abs().max()) > 1e-6, float((y1 - y0).abs().max())


def test_it_is_optimised_and_penalised(_base, monkeypatch):
    f = _fitter(monkeypatch, _base)
    assert any(q is f.hcorr for q in f.opt_params())
    monkeypatch.setattr(F, "HCORR_L2", 1.0)
    with torch.no_grad():
        f.hcorr.fill_(2.0)
    assert float(f.penalty()) >= 4.0 - 1e-9          # mean(2²) = 4


def test_cross_cycle_jumps_are_penalised(_base, monkeypatch):
    """**이전 주기와의 일관성** (사용자 지시, §52.274): 주기 간 도약을 문다.

    보정은 주기마다 독립이라 마음대로 도약할 수 있고, 그 불연속이 광대역 클릭(지지직)이 된다.
    """
    f = _fitter(monkeypatch, _base, k=3)
    monkeypatch.setattr(F, "HCORR_L2", 0.0)
    monkeypatch.setattr(F, "HCORR_TV", 1.0)
    with torch.no_grad():
        f.hcorr.zero_()
        f.hcorr[0, 0, ::2] = 1.0            # 주기마다 0/1 로 튄다
    jumpy = float(f.penalty())
    with torch.no_grad():
        f.hcorr.zero_()
        f.hcorr[0, 0].fill_(1.0)            # 같은 크기인데 **매끈**하다
    smooth = float(f.penalty())
    assert jumpy > 10.0 * max(smooth, 1e-12), (jumpy, smooth)


def test_tv_is_off_by_default_and_l2_still_works(_base, monkeypatch):
    assert F.HCORR_TV == 0.0
    f = _fitter(monkeypatch, _base, k=3)
    monkeypatch.setattr(F, "HCORR_L2", 1.0)
    monkeypatch.setattr(F, "HCORR_TV", 0.0)
    with torch.no_grad():
        f.hcorr.fill_(2.0)
    assert float(f.penalty()) >= 4.0 - 1e-9


def test_neighbouring_harmonic_jumps_are_penalised(_base, monkeypatch):
    """**이웃 배음과의 일관성** (§52.280): 배음 방향 도약을 문다 — 그 불규칙이 빗살이다."""
    f = _fitter(monkeypatch, _base, k=6)
    monkeypatch.setattr(F, "HCORR_L2", 0.0)
    monkeypatch.setattr(F, "HCORR_TV", 0.0)
    monkeypatch.setattr(F, "HCORR_KTV", 1.0)
    with torch.no_grad():
        f.hcorr.zero_()
        f.hcorr[0, ::2] = 1.0              # 배음마다 0/1 로 튄다
    jumpy = float(f.penalty())
    with torch.no_grad():
        f.hcorr.zero_()
        f.hcorr[0].fill_(1.0)              # 같은 크기인데 배음 방향으로 **매끈**하다
    smooth = float(f.penalty())
    assert jumpy > 10.0 * max(smooth, 1e-12), (jumpy, smooth)


def test_ktv_is_independent_of_tv(_base, monkeypatch):
    """두 축은 서로 다른 것을 문다 — 주기 방향으로만 튀면 KTV 는 0 이어야 한다."""
    f = _fitter(monkeypatch, _base, k=6)
    monkeypatch.setattr(F, "HCORR_L2", 0.0)
    monkeypatch.setattr(F, "HCORR_TV", 0.0)
    monkeypatch.setattr(F, "HCORR_KTV", 1.0)
    with torch.no_grad():
        f.hcorr.zero_()
        f.hcorr[:, :, ::2] = 1.0           # **주기** 방향으로만 튄다 (모든 배음·두 면이 같은 값)
    only_t = float(f.penalty())
    with torch.no_grad():
        f.hcorr.zero_()
        f.hcorr[:, ::2] = 1.0              # **배음** 방향으로 튄다
    only_k = float(f.penalty())
    # `penalty()` 에는 포먼트 순서 등 다른 항도 들어 있다 — 그 바닥(보정을 0 으로 둔 값)을 빼고 본다.
    with torch.no_grad():
        f.hcorr.zero_()
    base = float(f.penalty())
    assert only_t - base < 1e-9, (only_t, base)    # 주기 방향 도약은 KTV 가 안 문다
    assert only_k - base > 0.1, (only_k, base)
