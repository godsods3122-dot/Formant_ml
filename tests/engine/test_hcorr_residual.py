"""물리를 붙잡고 보정만 — `fit.FREEZE_CONTROLS`, `fit.HCORR_FMIN_HZ` (MEASUREMENTS §52.473)."""
import math

import numpy as np
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.fit import CopySynthFitter

FS = 48000


class _Stand:
    """`_hcorr_fmin_mask` 가 쓰는 것만."""

    def __init__(self, periods_s, K):
        n = int(sum(periods_s) * FS)
        edges = np.cumsum([0.0] + list(periods_s)) * FS
        u = np.interp(np.arange(n), edges, np.arange(len(edges)))      # 주기 수 (표본마다)
        self._pulse_phase = torch.as_tensor(2 * math.pi * u, dtype=torch.float64).unsqueeze(0)
        self.pulse_phi0 = torch.zeros((), dtype=torch.float64)
        self._hc_c0 = 0
        self.hcorr = torch.zeros(2, K, len(periods_s), dtype=torch.float64)
        self.fs = FS
        self.device = "cpu"


def test_fmin_mask_cuts_harmonics_below_the_frequency_per_cycle(monkeypatch):
    monkeypatch.setattr(F, "HCORR_FMIN_HZ", 3000.0)
    s = _Stand([1 / 200.0, 1 / 400.0, 1 / 250.0], K=30)          # f0 200·400·250 Hz
    m = CopySynthFitter._hcorr_fmin_mask(s).numpy()
    for c, f0 in enumerate((200.0, 400.0, 250.0)):
        k = np.arange(1, 31)
        got = m[:, c]                                             # 주기 길이를 표본으로 반올림하므로 경계 ±100 Hz 는 안 본다
        assert np.array_equal(got[k * f0 < 2900], np.zeros(int((k * f0 < 2900).sum())))
        assert np.all(got[k * f0 > 3100] == 1.0)


def test_fmin_zero_means_no_mask(monkeypatch):
    monkeypatch.setattr(F, "HCORR_FMIN_HZ", 0.0)
    assert CopySynthFitter._hcorr_fmin_mask(_Stand([1 / 200.0], K=10)) is None


def test_frozen_controls_do_not_move(monkeypatch):
    """보정이 없는 적합기에 FREEZE_CONTROLS 로 fit 을 돌리면 아무 손잡이도 안 움직인다."""
    from formant_ml.engine.control import ControlTrack, default_vector
    from formant_ml.engine.profile import DEFAULT_PROFILE
    from formant_ml.engine.voice import EngineConfig, VoiceEngine
    eng = VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, residual=False), DEFAULT_PROFILE)
    trk = ControlTrack(np.tile(default_vector(), (120, 1)), 1.0)
    trk["p_sub"] = 7.0; trk["adduction"] = 0.6; trk["f0_target"] = 200.0; trk["residual_mix"] = 0.0
    trk = trk.clamp()
    y = np.asarray(eng.render(trk), dtype=np.float64) * 1.3
    f = CopySynthFitter(eng, y, FS, trk)
    before = [q.detach().clone() for q in f.opt_params()]
    monkeypatch.setattr(F, "FREEZE_CONTROLS", True)
    f.fit(3, lr=0.05, verbose=False)
    for q, b in zip(f.opt_params(), before):
        assert torch.equal(q.detach(), b)


def test_fmax_mask_cuts_harmonics_above_the_frequency(monkeypatch):
    monkeypatch.setattr(F, "HCORR_FMIN_HZ", 0.0)
    monkeypatch.setattr(F, "HCORR_FMAX_HZ", 1500.0)
    s = _Stand([1 / 200.0, 1 / 300.0], K=30)
    m = CopySynthFitter._hcorr_fmin_mask(s).numpy()
    for c, f0 in enumerate((200.0, 300.0)):
        k = np.arange(1, 31)
        assert np.all(m[k * f0 < 1400, c] == 1.0) and np.all(m[k * f0 > 1600, c] == 0.0)
