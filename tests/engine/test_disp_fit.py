"""성문 분산 세기를 적합 대상으로 푸는 손잡이 (`glottis.DISP_FIT`, §52.121)."""
import math

import torch

from formant_ml.engine import glottis as G
from formant_ml.engine.calibration import engine_parameters, parameter_scope
from formant_ml.engine.profile import SpeakerProfile
from formant_ml.engine.voice import EngineConfig, VoiceEngine


def _engine():
    return VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False),
                       SpeakerProfile.load("profiles/yang_female.json"))


def _mult(v):
    x = torch.tensor(float(v))
    return float(torch.exp(G.DISP_LIM * torch.tanh(x / G.DISP_LIM)))


def test_registered_as_an_utterance_constant():
    assert "glottis_disp_log" in engine_parameters(_engine())
    assert parameter_scope("glottis_disp_log") == "utterance"


def test_identity_at_zero_and_bounded_both_ways():
    assert abs(_mult(0.0) - 1.0) < 1e-9
    assert abs(_mult(50.0) - math.exp(G.DISP_LIM)) < 1e-6
    assert abs(_mult(-50.0) - math.exp(-G.DISP_LIM)) < 1e-6
    assert _mult(50.0) > _mult(1.0) > _mult(0.0) > _mult(-1.0) > _mult(-50.0)


def test_off_by_default_and_needs_a_base_value():
    """0 에 배율을 곱해도 0 이다 — `--dispersion` 없이는 아무 일도 안 한다."""
    assert G.DISP_FIT is False
    from formant_ml.engine import voice as V
    assert V.GLOTTAL_DISPERSION == 0.0


def test_parameter_carries_gradient():
    eng = _engine()
    p = eng.glottis.disp_log
    assert p.requires_grad
    y = 60.0 * torch.exp(G.DISP_LIM * torch.tanh(p / G.DISP_LIM))
    y.backward()
    assert p.grad is not None and abs(float(p.grad)) > 0.0


def test_listed_as_fitted_when_enabled():
    """`FIELDS` 에만 넣고 `enabled` 에 안 넣으면 **저장은 되는데 적합은 안 된다** (§52.137).

    실제로 그렇게 새서 `--disp-fit` 이 판 하나를 통째로 무동작으로 돌았다.
    """
    from pathlib import Path
    src = Path("src/formant_ml/engine/fit.py").read_text(encoding="utf-8")
    i = src.index("enabled = dict(")
    block = src[i:src.index("self.constant_parameters", i)]
    assert "glottis_disp_log=_glottis_mod.DISP_FIT" in block
