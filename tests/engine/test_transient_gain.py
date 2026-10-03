"""구강 과도음 이벤트의 **이벤트별 이득** (voice.transient_gain / fit.TRANSIENT_FIT, §52.307)."""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.control import ControlTrack, PARAMS
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine

FS = 48000


def _eng():
    prof = SpeakerProfile.load("profiles/yang_female.json")
    return VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, residual=False,
                                    speaker="female"), prof)


def _track(events, n=60):
    vals = np.tile(np.array([p.default for p in PARAMS.values()], float), (n, 1))
    return ControlTrack(vals, 1.0, events)


def _ctrl(n=60):
    """엔진은 (B, T, P) 텐서를 받는다 — ControlTrack 이 아니다."""
    vals = np.tile(np.array([p.default for p in PARAMS.values()], float), (n, 1))
    return torch.as_tensor(vals, dtype=torch.float64).unsqueeze(0)


def test_default_is_off():
    assert F.TRANSIENT_FIT is False
    assert _eng().transient_gain is None


def test_gain_scales_the_transient():
    eng = _eng()
    ev = [{"t": 0.010, "name": "tongue_contact", "amp": 0.05}]
    a = eng(_ctrl(), ev, 0.0)["transient"]
    eng.reset()
    eng.transient_gain = torch.tensor([3.0], dtype=torch.float64)
    b = eng(_ctrl(), ev, 0.0)["transient"]
    assert float(b.abs().max()) == pytest.approx(3.0 * float(a.abs().max()), rel=1e-4)


def test_each_event_gets_its_own_gain():
    eng = _eng()
    ev = [{"t": 0.005, "name": "tongue_contact", "amp": 0.05},
          {"t": 0.040, "name": "tongue_contact", "amp": 0.05}]
    eng.transient_gain = torch.tensor([0.0, 2.0], dtype=torch.float64)
    y = eng(_ctrl(), ev, 0.0)["transient"][0]
    i = int(0.005 * FS)
    j = int(0.040 * FS)
    assert float(y[i:i + 200].abs().max()) < 1e-12          # 첫 이벤트는 꺼진다
    assert float(y[j:j + 200].abs().max()) > 1e-6           # 둘째는 산다


def test_gain_is_differentiable():
    eng = _eng()
    ev = [{"t": 0.010, "name": "tongue_contact", "amp": 0.05}]
    g = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    eng.transient_gain = torch.exp(g)
    eng(_ctrl(), ev, 0.0)["transient"].pow(2).sum().backward()
    assert g.grad is not None and float(g.grad.abs().sum()) > 0.0


def test_no_events_means_no_gain_tensor(monkeypatch):
    monkeypatch.setattr(F, "TRANSIENT_FIT", True)
    eng = _eng()
    tgt = np.zeros(int(0.06 * FS), dtype=np.float32)
    fit = F.CopySynthFitter(eng, tgt, FS, _track([]))
    assert fit.tr_gain is None


def test_one_gain_per_event(monkeypatch):
    monkeypatch.setattr(F, "TRANSIENT_FIT", True)
    eng = _eng()
    ev = [{"t": 0.005, "name": "tongue_contact", "amp": 0.05},
          {"t": 0.040, "name": "lip_smack", "amp": 0.05}]
    tgt = np.zeros(int(0.06 * FS), dtype=np.float32)
    fit = F.CopySynthFitter(eng, tgt, FS, _track(ev))
    assert fit.tr_gain is not None and tuple(fit.tr_gain.shape) == (2,)
    assert any(p is fit.tr_gain for p in fit.opt_params())


def test_prior_default_is_off():
    assert F.TRANSIENT_PRIOR == 0.0


def test_prior_pulls_the_gain_back_to_the_measured_value(monkeypatch):
    """풀어 두면 적합기가 클릭을 꺼 버린다 — 사전이 잰 값(로그 이득 0)에 묶는다."""
    monkeypatch.setattr(F, "TRANSIENT_PRIOR", 2.0)

    class _Stub:
        pass

    st = _Stub()
    st.tr_gain = torch.tensor([-1.5, 0.0], dtype=torch.float64, requires_grad=True)
    pen = F.TRANSIENT_PRIOR * (st.tr_gain ** 2).mean()
    assert float(pen) == pytest.approx(2.0 * (1.5 ** 2) / 2)
    pen.backward()
    # 0 보다 작은 쪽은 **키우는** 방향으로 기울기가 선다
    assert float(st.tr_gain.grad[0]) < 0.0
    assert float(st.tr_gain.grad[1]) == pytest.approx(0.0)
