"""주기별 보정은 성대가 떨 때만 소리를 낸다 (`voice.HCORR_GATED`, MEASUREMENTS §52.437)."""
import numpy as np
import torch

from formant_ml.engine import voice as V
from formant_ml.engine.control import ControlTrack, default_vector, INDEX
from formant_ml.engine.profile import SpeakerProfile


def _render(adduction, gated, monkeypatch):
    monkeypatch.setattr(V, "HCORR_GATED", gated)
    prof = SpeakerProfile.load("profiles/yang_female.json")
    eng = V.VoiceEngine(V.EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False, speaker="female"), prof)
    vals = np.tile(default_vector(), (300, 1))
    vals[:, INDEX["adduction"]] = adduction
    vals[:, INDEX["p_sub"]] = 8.0
    vals[:, INDEX["f0_target"]] = 250.0
    vals[:, INDEX["aspiration"]] = 0.0
    tr = ControlTrack(vals, 1.0)
    n = 300 * 48
    t = torch.arange(n) / 48000.0
    eng.source_add = (0.05 * torch.sin(2 * np.pi * 250.0 * t)).unsqueeze(0)
    with torch.no_grad():
        y = eng.render(tr)
    eng.source_add = None
    with torch.no_grad():
        y0 = eng.render(tr)
    return float(np.sqrt(np.mean((y - y0)[4800:] ** 2)))       # 보정이 더한 몫


def test_correction_is_silent_when_the_folds_do_not_vibrate(monkeypatch):
    loud = _render(0.0, False, monkeypatch)                 # 내전 0 — 성대가 안 떪
    gated = _render(0.0, True, monkeypatch)
    assert gated < 1e-3 * loud


def test_correction_still_acts_when_the_folds_vibrate(monkeypatch):
    assert _render(0.6, True, monkeypatch) > 0.1 * _render(0.6, False, monkeypatch)
