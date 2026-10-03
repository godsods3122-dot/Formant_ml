"""주기별 배음 보정의 **저차원 묶음** (MEASUREMENTS §52.362).

사용자 A/B: 절제판 일곱 중 `nohcorr` 만 유성 구간이 안 갈라졌다. `hcorr` 의 배음 사이 상관이
0.000 이라 배음들이 각자 따로 논다 — 공통 운명 단서가 부서진다.
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


@pytest.fixture
def _engine():
    return VoiceEngine(EngineConfig(sample_rate=FS, frame_ms=1.0, residual=False),
                       DEFAULT_PROFILE)


def _track(n=300):
    v = np.tile(default_vector(), (n, 1))
    tr = ControlTrack(v, 1.0)
    tr["p_sub"] = 7.0; tr["adduction"] = 0.6; tr["f0_target"] = 200.0
    tr["f1"] = 700; tr["f2"] = 1200; tr["f3"] = 2600; tr["f4"] = 3600
    tr["residual_mix"] = 0.0
    return tr.clamp()


def _fitter(eng, monkeypatch, rank, k=12):
    monkeypatch.setattr(F, "HCORR_K", k)
    monkeypatch.setattr(F, "HCORR_RANK", rank)
    monkeypatch.setattr(F, "PULSE_LOCK", True)
    tr = _track()
    f = CopySynthFitter(eng, eng.render(tr), FS, tr)
    f._pulse_lock_prepare() if hasattr(f, "_pulse_lock_prepare") else None
    return f


def test_rank_zero_keeps_the_free_form(_engine, monkeypatch):
    monkeypatch.setattr(F, "HCORR_RANK", 0)
    f = CopySynthFitter(_engine, _engine.render(_track()), FS, _track())
    assert f._hc_u is None
    assert f.hcorr is None                      # HCORR_K = 0 이라 준비 자체를 안 한다


def test_factored_hcorr_starts_at_zero_and_has_the_right_shape(_engine, monkeypatch):
    """곱이 0 에서 출발해야 **예전과 같은 출발점**이다."""
    monkeypatch.setattr(F, "HCORR_RANK", 3)
    f = CopySynthFitter(_engine, _engine.render(_track()), FS, _track())
    f._hc_u = torch.randn((2, 8, 3), dtype=torch.float64, requires_grad=True)
    f._hc_v = torch.zeros((2, 3, 40), dtype=torch.float64, requires_grad=True)
    H = f.hcorr
    assert H.shape == (2, 8, 40)
    assert torch.allclose(H, torch.zeros_like(H))


def test_factored_hcorr_is_low_rank(_engine, monkeypatch):
    """**배음 축의 계수가 r 이어야 한다** — 이것이 배음을 함께 움직이게 하는 구조다."""
    monkeypatch.setattr(F, "HCORR_RANK", 2)
    f = CopySynthFitter(_engine, _engine.render(_track()), FS, _track())
    g = torch.Generator().manual_seed(0)
    f._hc_u = (torch.randn((2, 16, 2), generator=g, dtype=torch.float64)).requires_grad_(True)
    f._hc_v = (torch.randn((2, 2, 50), generator=g, dtype=torch.float64)).requires_grad_(True)
    H = f.hcorr.detach()
    for c in range(2):
        sv = torch.linalg.svdvals(H[c])
        assert float(sv[2]) < 1e-9 * float(sv[0]), (float(sv[0]), float(sv[2]))


def test_factored_hcorr_makes_harmonics_move_together(_engine, monkeypatch):
    """계수 1 이면 모든 배음이 **한 시간 모양**으로만 움직인다 — 상관 |1|."""
    monkeypatch.setattr(F, "HCORR_RANK", 1)
    f = CopySynthFitter(_engine, _engine.render(_track()), FS, _track())
    g = torch.Generator().manual_seed(1)
    f._hc_u = torch.randn((2, 20, 1), generator=g, dtype=torch.float64).requires_grad_(True)
    f._hc_v = torch.randn((2, 1, 80), generator=g, dtype=torch.float64).requires_grad_(True)
    A = f.hcorr.detach()[0].numpy()
    Z = A - A.mean(1, keepdims=True)
    C = np.corrcoef(Z)
    off = np.abs(C[~np.eye(len(C), dtype=bool)])
    assert off.min() > 0.999, off.min()


def test_gradient_reaches_both_factors(_engine, monkeypatch):
    monkeypatch.setattr(F, "HCORR_RANK", 2)
    f = CopySynthFitter(_engine, _engine.render(_track()), FS, _track())
    f._hc_u = torch.randn((2, 10, 2), dtype=torch.float64, requires_grad=True)
    f._hc_v = torch.zeros((2, 2, 30), dtype=torch.float64, requires_grad=True)
    (f.hcorr ** 2).sum().backward()
    assert f._hc_v.grad is not None and float(f._hc_v.grad.abs().sum()) == 0.0   # v=0 -> 기울기 0
    f._hc_v = (torch.randn((2, 2, 30), dtype=torch.float64) * 0.1).requires_grad_(True)
    f._hc_u.grad = None
    (f.hcorr ** 2).sum().backward()
    assert float(f._hc_u.grad.abs().sum()) > 0.0
    assert float(f._hc_v.grad.abs().sum()) > 0.0


def test_opt_params_registers_the_factors_not_the_product(_engine, monkeypatch):
    monkeypatch.setattr(F, "HCORR_RANK", 2)
    f = CopySynthFitter(_engine, _engine.render(_track()), FS, _track())
    f._hc_u = torch.zeros((2, 6, 2), dtype=torch.float64, requires_grad=True)
    f._hc_v = torch.zeros((2, 2, 20), dtype=torch.float64, requires_grad=True)
    ps = f.opt_params()
    assert any(p is f._hc_u for p in ps) and any(p is f._hc_v for p in ps)
    assert all(p.is_leaf for p in ps)


def test_default_rank_is_zero():
    assert F.HCORR_RANK == 0
