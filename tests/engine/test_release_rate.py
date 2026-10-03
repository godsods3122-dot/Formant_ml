"""파열 개방의 면적 증가 속도 (`voice.RELEASE_RATE`, `fit.RELEASE_PRIOR_W`, MEASUREMENTS §52.456)."""
import math
import types

import torch

from formant_ml.engine import fit as F
from formant_ml.engine import voice as V
from formant_ml.engine.voice import EngineConfig, VoiceEngine


def _eng():
    return VoiceEngine(EngineConfig(sample_rate=48000, frame_ms=1.0, residual=False))


def _step_area(n=200, r=50, top=3.0):
    a = torch.zeros(1, n, dtype=torch.float64)
    a[:, r:] = top
    return a


def test_limiter_caps_the_opening_at_the_rate():
    eng = _eng()
    eng.release_frames, eng.release_rate = [50], torch.tensor([25.0], dtype=torch.float64)
    a = _step_area()
    out = eng._limit_release(a, 0, a.shape[1], {})
    k = torch.arange(0, 60, dtype=torch.float64)
    assert torch.allclose(out[0, 50:110], 0.025 * (k + 1)), "개방이 25 cm²/s (틀마다 0.025 cm²) 로 커지지 않았다"
    assert torch.equal(out[0, :40], a[0, :40]), "창 앞을 건드렸다"
    post = int(V.RELEASE_POST_MS)
    assert float(out[0, 50 + post]) == 3.0, "창 밖을 묶었다"


def test_limiter_leaves_closing_alone():
    eng = _eng()
    eng.release_frames, eng.release_rate = [50], torch.tensor([25.0], dtype=torch.float64)
    a = torch.full((1, 200), 3.0, dtype=torch.float64)
    a[:, 45:52] = torch.linspace(3.0, 0.0, 7, dtype=torch.float64)
    out = eng._limit_release(a, 0, 200, {})
    assert torch.equal(out[0, :52], a[0, :52]), "닫히는 쪽을 묶었다"


def test_limiter_is_streaming_invariant():
    eng = _eng()
    eng.release_frames, eng.release_rate = [50, 120], torch.tensor([25.0, 100.0], dtype=torch.float64)
    a = _step_area()
    a[:, 110:118] = 0.0
    whole = eng._limit_release(a, 0, 200, {})
    st, parts = {}, []
    for c0, c1 in ((0, 57), (57, 123), (123, 200)):
        parts.append(eng._limit_release(a[:, c0:c1], c0, c1 - c0, st))
    assert torch.allclose(torch.cat(parts, 1), whole)


def test_rate_gradient_flows():
    eng = _eng()
    lr = torch.tensor([math.log(40.0)], dtype=torch.float64, requires_grad=True)
    eng.release_frames, eng.release_rate = [50], torch.exp(lr)
    out = eng._limit_release(_step_area(), 0, 200, {})
    out[0, 50:70].sum().backward()
    assert lr.grad is not None and float(lr.grad) > 0.0


def test_prior_prefers_the_literature_rates_and_keeps_heavy_tails():
    def nll(r):
        me = types.SimpleNamespace(rel_log_r=torch.tensor([math.log(r)], dtype=torch.float64))
        return float(F.CopySynthFitter.release_prior_loss(me))
    for r in (25.0, 50.0, 100.0):
        assert nll(r) < nll(r * 0.6) or r == 100.0
    assert nll(25.0) < nll(8.0) < nll(2.0)
    assert nll(100.0) < nll(400.0) < nll(3600.0)
    assert math.isfinite(nll(3600.0)) and nll(3600.0) - nll(50.0) < 40.0, "꼬리가 두꺼워야 한다 (막지 않고 비싸게)"
