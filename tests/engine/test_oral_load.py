"""**경성문 압력차로 성문을 민다** (`glottis.ORAL_LOAD`, MEASUREMENTS §52.344).

사용자: *"진짜 물리적인 내압 때문에 순간적으로 발생하는 거야."* 성문 진동의 구동은 폐압이 아니라
`p_sub − p_oral` 이다. 마찰·파열 중에는 구강압이 차올라 압력차가 줄고, 그 소리가 끝나면 구강압이
τ 6 ms 로 빠지면서 압력차가 순간적으로 회복된다.

구강압 자체는 **이미 `noise.FricationNoise` 가 파열 버스트를 내려고 적분하고 있었다** — 성문이 그걸
몰랐을 뿐이다. 그래서 이 항은 새 파라미터를 늘리지 않는다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import glottis as G
from formant_ml.engine.control import INDEX, PARAMS


def _ctrl(n=300, fric=(100, 200), a_open=1.5, a_fric=0.06, ps=9.0):
    """앞뒤는 모음(협착이 열림), 가운데가 마찰(협착이 좁음).

    `a_fric` 기본값은 치찰음의 실제 협착 면적 규모다. **구강압이 차려면 협착이 성문
    (`ag_dc` ≈ 0.07 cm²)과 견줄 만큼 좁아야 한다** — 0.25 cm² 에서는 po/ps 가 7 % 뿐이다
    (po_ss = ps·Ag²/(Ag²+Ac²)). 우리 적합기가 마찰 구간에서 찾는 0.21~0.30 (§52.338) 은
    구강압 관점에서 여전히 넓다는 뜻이기도 하다.
    """
    vals = {k: np.full(n, p.default, float) for k, p in PARAMS.items()}
    vals["p_sub"][:] = ps
    vals["adduction"][:] = 0.6
    vals["tension"][:] = 0.5
    vals["f0_target"][:] = 0.0            # tension 으로 F0 를 정하는 경로
    vals["f0_scale"][:] = 1.0
    vals["a_c"][:] = a_open
    vals["a_c"][fric[0]:fric[1]] = a_fric
    vals["velum"][:] = 0.0
    return {k: torch.as_tensor(v, dtype=torch.float64).unsqueeze(0) for k, v in vals.items()}


def _src(hop=48, fs=48000):
    return G.GlottalSource(fs=fs, hop=hop)


def test_default_is_off():
    assert G.ORAL_LOAD == 0.0


def test_oral_pressure_rises_in_a_constriction_and_falls_after():
    """협착이 서면 구강압이 차고, 풀리면 빠진다 — 그 시상수가 사후의 '순간'을 만든다."""
    src = _src()
    c = _ctrl()
    ag = 0.02 + 0.5 * (1.0 - c["adduction"].clamp(0, 1)) ** 2.5
    po = src.oral_pressure(c, ag)[0].numpy()
    assert po[50] < 0.5                                  # 열린 모음 자세: 구강압이 거의 없다
    assert po[195] > 2.0, po[195]                        # 마찰 끝 무렵: 차 있다
    assert po[260] < 0.5 * po[195]                       # 풀리면 빠진다


def test_the_fall_takes_a_few_milliseconds():
    """열림 시상수는 6 ms 다 — 실측한 F0 빠른 성분이 30 ms 에 가라앉는 것과 자릿수가 같다."""
    src = _src()
    po = src.oral_pressure(_ctrl(), 0.02 + 0.5 * (1.0 - torch.full((1, 300), 0.6,
                                                                   dtype=torch.float64)) ** 2.5)[0]
    po = po.numpy()
    peak = po[199]
    after = po[200:240]
    # 1 프레임 = hop/fs = 1 ms. e^-1 까지 떨어지는 데 걸리는 프레임
    k = int(np.argmax(after < peak / np.e)) if (after < peak / np.e).any() else 99
    assert 2 <= k <= 15, k


def test_a_closed_tract_charges_towards_p_sub():
    """완전 폐쇄(파열 앞)에서는 구강압이 폐압까지 차오른다 — 그것이 버스트의 저장이다."""
    src = _src()
    c = _ctrl(a_fric=0.01, fric=(100, 250))              # a_c 0.01 = 닫힘
    ag = 0.02 + 0.5 * (1.0 - c["adduction"].clamp(0, 1)) ** 2.5
    po = src.oral_pressure(c, ag)[0].numpy()
    assert po[245] > 0.7 * 9.0                           # p_sub 의 70 % 넘게 찬다


def test_velum_open_vents_the_pressure():
    """연구개가 열리면 기류가 코로 빠져 구강압이 안 찬다 — 비음이 터지지 않는 까닭."""
    src = _src()
    c = _ctrl(a_fric=0.01)
    c["velum"][:] = 1.0
    ag = 0.02 + 0.5 * (1.0 - c["adduction"].clamp(0, 1)) ** 2.5
    assert float(src.oral_pressure(c, ag).max()) < 0.5


def test_drive_drops_inside_the_constriction_when_on():
    """켜면 마찰 안에서 성문 구동이 줄어든다 — 압력의 상당 부분이 협착에서 떨어지기 때문."""
    src = _src()
    c = _ctrl()
    off = src.physiology(c)
    G.ORAL_LOAD = 1.0                                    # conftest 가 시험 뒤 되돌린다
    on = src.physiology(c)
    a_off, a_on = off["amp"][0].numpy(), on["amp"][0].numpy()
    assert a_on[190] < 0.9 * a_off[190]                  # 마찰 안: 덜 떤다
    assert a_on[50] == pytest.approx(a_off[50], rel=0.02)  # 열린 모음: 그대로


def test_drive_recovers_after_the_constriction():
    """그리고 협착이 풀리면 되돌아온다 — 그 회복이 사후의 순간 변화다."""
    src = _src()
    c = _ctrl()
    G.ORAL_LOAD = 1.0
    on = src.physiology(c)["amp"][0].numpy()
    assert on[260] > 1.5 * on[195]                       # 마찰 끝 뒤에 살아난다


def test_it_is_a_single_integration_shared_with_frication():
    """구강압은 **한 곳에서만** 적분한다 — 성문이 낸 값을 마찰이 그대로 받는다."""
    src = _src()
    c = _ctrl()
    G.ORAL_LOAD = 1.0
    st = src.physiology(c)
    assert st["po"] is not None and st["po"].shape == c["p_sub"].shape
    ag = st["ag_dc"]
    assert torch.allclose(st["po"], src.oral_pressure(c, ag))


def test_off_means_byte_identical():
    """꺼 두면 예전 판과 한 치도 다르지 않아야 한다."""
    src = _src()
    c = _ctrl()
    a = src.physiology(c)
    assert a["po"] is None
    assert torch.isfinite(a["amp"]).all()
