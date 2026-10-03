"""구강압 상미분방정식 (`glottis.ORAL_ODE`, MEASUREMENTS §52.436) — 문헌 실측을 재현하는가."""
import numpy as np
import pytest
import torch

from formant_ml.engine import glottis as G

T = 200


@pytest.fixture(autouse=True)
def _ode(monkeypatch):
    monkeypatch.setattr(G, "ORAL_ODE", True)


def _run(add, cw, ps=8.0, monkeypatch=None):
    G.ORAL_CW_ML = cw
    g = G.GlottalSource(48000, 48, f0_range=(120.0, 950.0, 240.0))
    a_c = torch.full((1, T), 3.0)
    a_c[0, 50:125] = 0.001                                   # 75 ms 폐쇄 (여성 /b/ 평균 75.5 ms)
    c = {"p_sub": torch.full((1, T), ps), "a_c": a_c, "velum": torch.zeros(1, T)}
    ag = torch.full((1, T), 0.02 + 0.5 * (1 - add) ** 2.5)
    po = g._oral_ode(c, ag)[0].numpy()
    return po, g._last_uc[0].numpy()


def test_voiced_bilabial_pressure_at_release_matches_women(monkeypatch):
    monkeypatch.setattr(G, "ORAL_CW_ML", 4.0)
    po, _ = _run(0.6, 4.0)
    assert 3.0 < po[124] < 5.5                              # 실측 4.3 cmH2O (PMC2651765)


def test_voiceless_open_glottis_fills_to_lung_pressure(monkeypatch):
    monkeypatch.setattr(G, "ORAL_CW_ML", 0.5)
    po, _ = _run(0.1, 0.5)
    assert po[60] > 7.5                                     # 10 ms 안에 거의 Ps


def test_open_vowel_has_no_oral_pressure_and_release_vents_fast(monkeypatch):
    monkeypatch.setattr(G, "ORAL_CW_ML", 2.0)
    po, uc = _run(0.6, 2.0)
    assert po[30] < 0.05 and po[180] < 0.05
    assert po[130] < 0.2 * po[124]                          # 개방 5 ms 안에 빠진다
    assert uc[125] > 5 * uc[180]                            # 개방 순간의 제트가 모음 유량보다 크다
