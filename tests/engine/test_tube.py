"""면적 함수 -> 공진. 관의 절대 경계조건이 극의 위치와 대역폭을 정한다."""
import math

import pytest
import torch

from formant_ml.engine.tube import (C_SOUND, loss_bandwidths, uniform_area,
                                    webster_modes, webster_poles)

L = 14.6


def analytic(n):
    return (2 * n - 1) * C_SOUND / (4.0 * L)


def test_uniform_tube_reproduces_the_quarter_wave_series():
    """균일관의 해석해는 f_n = (2n−1)c/4L 이다 — 이산화가 그것을 재현해야 한다."""
    f = webster_poles(uniform_area(64, 3.0, dtype=torch.float64), L, 6)
    for k in range(6):
        assert abs(float(f[k]) - analytic(k + 1)) / analytic(k + 1) < 0.005


def test_discretisation_is_second_order():
    """격자를 반으로 줄이면 오차가 1/4 이 되어야 한다.

    셀 중심 격자 + 입술 쪽 유령셀이 그 조건이다. 마지막 미지수를 그냥 버리면
    실효 길이가 h/2 짧아져 1 차로 떨어진다 (N=64 에서 0.78 % -> 0.30 %).
    """
    err = []
    for n in (32, 64, 128):
        f = webster_poles(uniform_area(n, 3.0, dtype=torch.float64), L, 4)
        err.append(max(abs(float(f[k]) - analytic(k + 1)) / analytic(k + 1) for k in range(4)))
    for a, b in zip(err, err[1:]):
        assert 3.0 < a / b < 5.0, err


def test_pole_spacing_is_the_absolute_constraint():
    """면적 함수가 무엇이든 **고차** 극 간격은 c/(2L) 로 간다. 경계조건이 정한다.

    이것이 사용자가 물은 "성도의 절대적인 공진 경계조건" 이다. 조음은 저차 극을
    크게 움직이지만 (아래에서 F1↔F2 간격이 928~1300 Hz 로 흔들린다) 극 **밀도**는
    건드리지 못한다 — 슈투름-리우빌 문제의 점근 법칙이다. 자유 극 8 개짜리 종속에는
    이 제약이 아예 없어서, 피팅이 F_K 를 아무 데나 놓고 그 위에 구멍을 남겼다.
    """
    torch.manual_seed(0)
    lo = []
    for _ in range(8):
        x = (torch.arange(256, dtype=torch.float64) + 0.5) / 256
        la = torch.randn(3, dtype=torch.float64)
        area = torch.exp(0.9 * (la[0] + la[1] * torch.cos(math.pi * x) +
                                la[2] * torch.cos(2 * math.pi * x)))
        f = webster_poles(area, L, 16)
        lo.append(float(f[:3].diff().mean()))
        assert abs(float(f[4:].diff().mean()) - C_SOUND / (2.0 * L)) < 20.0
    assert max(lo) - min(lo) > 150.0, "저차 극은 조음에 따라 흔들려야 한다"


def test_area_function_moves_the_formants():
    """중성관보다 인두를 좁히면 F1 이 올라간다 (열린 모음). 기하가 극을 정한다."""
    x = (torch.arange(44, dtype=torch.float64) + 0.5) / 44
    neutral = torch.full_like(x, 3.0)
    pharyngeal = 3.0 * torch.exp(-1.4 * torch.exp(-0.5 * ((x - 0.25) / 0.09) ** 2))
    f0 = webster_poles(neutral, L, 3)
    f1 = webster_poles(pharyngeal, L, 3)
    assert float(f1[0]) > float(f0[0]) + 50.0


def test_loss_law_lands_in_the_literature_range():
    """대역폭은 자유 파라미터가 아니라 계산값이다 (Fant 1972 / Stevens 1998 범위)."""
    a = uniform_area(44, 3.0, dtype=torch.float64)
    f, p = webster_modes(a, L, 6)
    b = loss_bandwidths(a, L, f, p, wall_shape=1.3)
    for k, (lo, hi) in enumerate([(40, 70), (50, 90), (80, 140),
                                  (150, 250), (200, 300), (250, 400)]):
        assert lo <= float(b[k]) <= hi, (k + 1, float(b[k]))


def test_radiation_resistance_saturates():
    """배플 피스톤의 복사 저항은 ka>>1 에서 rho c/A 로 포화한다.

    (ka)^2/2 만 쓰면 포화가 없어 5·6 번 포먼트 대역폭이 문헌의 1.4~1.9 배가 된다.
    포화하면 B(f)/f 가 고역에서 평평해진다.
    """
    a = uniform_area(44, 3.0, dtype=torch.float64)
    f, p = webster_modes(a, L, 12)
    b = loss_bandwidths(a, L, f, p, wall_shape=1.3)
    rel = (b / f)[6:]
    assert float(rel.max() - rel.min()) < 0.01, rel


def test_poles_and_bandwidths_are_differentiable():
    la = torch.zeros(44, dtype=torch.float64, requires_grad=True)
    a = torch.exp(la) * 3.0
    f, p = webster_modes(a, L, 8)
    b = loss_bandwidths(a, L, f, p)
    (f.sum() + b.sum()).backward()
    assert torch.isfinite(la.grad).all()
    assert float(la.grad.abs().max()) > 0.0
