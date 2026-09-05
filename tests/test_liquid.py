"""유음 렌더링 검증 — 혀끝이 실제로 성도 면적함수가 되는가.

    PYTHONPATH=src python3 tests/test_liquid.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from formant_ml.config import Config, sections_for
from formant_ml.liquid import (lateral_antiformants, liquid_area,
                               side_branch_zero)

FS = 24000
OPEN = 0.30
TAP = [(0.0, OPEN), (0.10, OPEN), (0.115, -0.04), (0.145, -0.04),
       (0.16, OPEN), (0.25, OPEN)]


def _cfg():
    c = Config()
    c.filt.n_tract_sections = sections_for(FS, 14.6)     # 여성 성도
    return c


def test_side_branch_zero_lands_in_the_reported_band():
    """측지 2~4 cm -> 영점 2000~5000 Hz (Stevens 1998, Zhang & Espy-Wilson 2004)."""
    for L in (2.0, 3.0, 4.0):
        assert 2000.0 <= side_branch_zero(L) <= 5000.0, L


def test_lateral_antiformants_have_the_right_shape():
    fz, bw = lateral_antiformants(7)
    assert fz.shape == (1, 7, 2) and bw.shape == (1, 7, 2)
    assert bool((fz[..., 0] < fz[..., 1]).all())          # 설상공이 더 낮다


def test_tip_closure_reaches_the_area_function():
    """혀끝이 닿으면 면적함수의 전방 단이 실제로 닫혀야 한다.

    프레임률로 내릴 때 평균을 쓰면 접촉이 통째로 사라진다 — 최솟값을 써야 한다.
    """
    cfg = _cfg()
    area, contact, _ = liquid_area(0.25, TAP, cfg, po=2000.0, n_masses=5)
    assert area.shape[0] == 1 and area.shape[2] == cfg.filt.n_tract_sections
    assert bool((contact > 0).any()), "접촉이 없었다"
    closed = area[0, contact > 0, -7:-2].amin()
    open_ = area[0, contact == 0, -7:-2].amin()
    assert float(closed) < 0.05, float(closed)
    assert float(open_) > float(closed) * 5.0, (float(open_), float(closed))


def test_lateral_keeps_a_side_channel_open():
    """설측음은 중앙이 막혀도 면적이 0 으로 안 내려간다 — 옆이 열려 있다."""
    cfg = _cfg()
    hold = [(0.0, -0.05), (0.25, -0.05)]
    area, _, _ = liquid_area(0.25, hold, cfg, po=1200.0, n_masses=5,
                             lateral_area_cm2=0.22)
    assert float(area.amin()) >= 0.22 - 1e-6


def test_area_function_stays_positive_and_finite():
    """면적이 0 이하가 되면 도파관의 반사계수가 발산한다."""
    cfg = _cfg()
    for po in (500.0, 8000.0):
        area, _, _ = liquid_area(0.25, TAP, cfg, po=po, n_masses=5)
        assert float(area.amin()) > 0.0
        assert bool(torch.isfinite(area).all())


def test_waveguide_renders_the_liquid_without_blowing_up():
    """면적함수가 도파관을 통과해 유한한 파형이 나와야 한다 (폐쇄 구간 포함)."""
    from formant_ml.models.synth import Controls, PhysicalVoiceSynth
    cfg = _cfg()
    area, _, _ = liquid_area(0.25, TAP, cfg, po=2000.0, n_masses=5)
    t = area.shape[1]
    K, nb = cfg.filt.n_formants, cfg.noise.n_bands
    syn = PhysicalVoiceSynth(cfg, tract_mode="waveguide")
    c = Controls(
        f0=torch.full((1, t, 1), 210.0), harmonic_amp=torch.ones(1, t, 1),
        rd=torch.full((1, t, 1), 1.1),
        formant_freq=torch.linspace(500, 6000, K).reshape(1, 1, -1)
                          .expand(1, t, K).contiguous(),
        formant_bw=torch.full((1, t, K), 90.0),
        formant_gain=torch.ones(1, t, K),
        noise_bands=torch.full((1, t, nb), 2e-4),
        noise_entry=torch.zeros(1, t, 1),
        noise_am=torch.full((1, t, 1), 0.15), area=area)
    with torch.no_grad():
        y = syn(c)["audio"]
    assert bool(torch.isfinite(y).all())
    assert float(y.abs().max()) > 1e-4, "무음이 나왔다"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
