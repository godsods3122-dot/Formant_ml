"""성문 제트의 레이놀즈 기식 (`glottis.GLOTTAL_RE`, MEASUREMENTS §52.417)."""
import math

import numpy as np
import torch

from formant_ml.engine import glottis as G

N = 1600                      # 300 Hz 의 10 주기


def _env(add, ps, a_c, amp):
    g = G.GlottalSource(48000, 48, f0_range=(120.0, 950.0, 240.0))
    one = torch.ones(1, N)
    ph = torch.linspace(0, 20 * math.pi, N + 1)[:-1][None]
    rd = 0.3 + 2.4 * (1 - add) ** 1.5
    c = {"a_c": a_c * one, "p_sub": ps * one, "aspiration": one}
    st = {"ag_dc": (0.02 + 0.5 * (1 - add) ** 2.5) * one}
    return g.glottal_re_env(c, st, lambda v: v, ph, rd * one, amp * one, one)[0].numpy()


def test_pressed_closure_is_below_the_reynolds_threshold():
    """압착 발성은 닫힐 때 틈이 좁아 Re < Re_c — 그 순간 난류가 꺼진다."""
    e = _env(0.9, 8.0, 3.0, 0.5)[:160]
    assert e.min() < 1e-2 * e.max()          # 매끈한 문턱(softplus)이라 0 은 아니다 — 실측 −58 dB


def test_breathy_noise_is_steady_through_the_cycle():
    e = _env(0.2, 8.0, 3.0, 0.3)[:160]
    assert e.min() > 0.8 * e.max()


def test_oral_constriction_takes_the_pressure_drop():
    """/s/ 처럼 구강이 좁으면 압력이 협착에서 떨어져 성문 제트가 조용하다."""
    open_ = _env(0.06, 8.0, 3.0, 0.0).mean()
    closed = _env(0.06, 8.0, 0.05, 0.0).mean()
    assert closed < 1e-3 * open_


def test_louder_with_pressure_as_dipole_v_cubed():
    lo, hi = _env(0.6, 4.0, 3.0, 0.3).mean(), _env(0.6, 8.0, 3.0, 0.3).mean()
    assert hi > 2.0 * lo


def test_flow_table_starts_at_opening_and_peaks_at_one():
    g = G.GlottalSource(48000, 48, f0_range=(120.0, 950.0, 240.0))
    ph = torch.tensor([[0.0, 1e-4]])
    f = g._flow_pos(ph, torch.full((1, 2), 1.2))
    assert float(f.max()) < 0.01
    full = g._flow_pos(torch.linspace(0, 2 * math.pi, 2000)[None], torch.full((1, 2000), 1.2))
    assert abs(float(full.max()) - 1.0) < 0.02
