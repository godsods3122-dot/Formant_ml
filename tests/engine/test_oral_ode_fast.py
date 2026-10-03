"""구강압 방정식 빠른 경로가 torch 판과 같은 값·같은 기울기를 내는가 (MEASUREMENTS §52.468)."""
import numpy as np
import torch

from formant_ml.engine import glottis as gl
from formant_ml.engine import oral_ode_fast as of


def _controls(T=400, seed=0):
    g = np.random.default_rng(seed)
    t = np.arange(T)
    # 열림 → 폐쇄(파열) → 개방 → 치찰 → 모음, 성문도 여닫는다
    a_c = np.where(t < 80, 3.0, np.where(t < 160, 0.001, np.where(t < 240, np.linspace(0.05, 2.0, T)[t], 0.1)))
    a_c = np.where(t > 320, 5.0, a_c) * (1 + 0.05 * g.standard_normal(T))
    c = dict(p_sub=torch.tensor(7.0 + 2 * np.sin(t / 50.0) + 0.1 * g.standard_normal(T))[None],
             velum=torch.tensor(np.clip(0.3 * (t > 360), 0, 1) * 1.0)[None],
             a_c=torch.tensor(np.clip(a_c, 1e-3, None))[None])
    ag = torch.tensor(0.02 + 0.4 * (np.sin(t / 30.0) > 0.3) + 0.05 * g.random(T))[None]
    return c, ag


def _run(fast, c, ag):
    src = gl.GlottalSource(48000, 48)
    old = gl.ORAL_ODE_FAST
    gl.ORAL_ODE_FAST = fast
    try:
        c = {k: v.clone().requires_grad_(True) for k, v in c.items()}
        ag = ag.clone().requires_grad_(True)
        po = src._oral_ode(c, ag)
        uc = src._last_uc
        w1 = torch.linspace(-1, 1, po.shape[1], dtype=po.dtype)[None]
        w2 = torch.cos(torch.arange(po.shape[1], dtype=po.dtype) / 7.0)[None]
        L = (po * w1).sum() + 1e-3 * (uc * w2).sum()
        L.backward()
        return po.detach(), uc.detach(), {k: v.grad for k, v in c.items()}, ag.grad
    finally:
        gl.ORAL_ODE_FAST = old


def test_values_and_grads_match_torch():
    assert of.HAVE_FAST
    c, ag = _controls()
    p0, u0, g0, ga0 = _run(False, c, ag)
    p1, u1, g1, ga1 = _run(True, c, ag)
    assert torch.allclose(p1, p0, rtol=1e-6, atol=1e-7)
    assert torch.allclose(u1, u0, rtol=1e-6, atol=1e-5)
    for k in ("p_sub", "a_c", "velum"):
        a, b = g1[k], g0[k]
        scale = float(b.abs().max()) + 1e-12
        assert float((a - b).abs().max()) / scale < 1e-3, k
    scale = float(ga0.abs().max()) + 1e-12
    assert float((ga1 - ga0).abs().max()) / scale < 1e-3


def test_logistic_amp_matches_torch():
    """성대 떨림 진폭 재귀 (amp_fast) — 기동·꺼짐·재기동이 섞인 제어에서 torch 판과 같은 값·기울기."""
    from formant_ml.engine import amp_fast as af
    assert af.HAVE_FAST
    g = np.random.default_rng(1)
    T = 500
    t = np.arange(T)
    base = dict(p_sub=7.0 + 3.0 * (t > 50) - 5.0 * ((t > 250) & (t < 300)) + 0.1 * g.standard_normal(T),
                adduction=np.clip(0.45 - 0.4 * ((t > 200) & (t < 320)) + 0.02 * g.standard_normal(T), 0, 1),
                tension=0.5 + 0.1 * np.sin(t / 40.0), f0_target=220.0 + 30 * np.sin(t / 60.0),
                f0_scale=np.ones(T), rd_offset=np.zeros(T), a_c=np.full(T, 3.0), velum=np.zeros(T),
                aspiration=np.full(T, 0.2))
    src = gl.GlottalSource(48000, 48)
    outs = []
    for fast in (False, True):
        c = {k: torch.tensor(v, dtype=torch.float64)[None].requires_grad_(True) for k, v in base.items()}
        old = gl.AMP_FAST
        gl.AMP_FAST = fast
        try:
            st = src.physiology(c)
        finally:
            gl.AMP_FAST = old
        w = torch.cos(torch.arange(T, dtype=torch.float64) / 9.0)[None]
        (st["amp"] * w).sum().backward()
        outs.append((st["amp"].detach(), {k: v.grad.clone() for k, v in c.items() if v.grad is not None}))
    (a0, g0), (a1, g1) = outs
    assert torch.allclose(a1, a0, rtol=1e-9, atol=1e-10)
    for k in g0:
        scale = float(g0[k].abs().max()) + 1e-12
        assert float((g1[k] - g0[k]).abs().max()) / scale < 1e-6, k
