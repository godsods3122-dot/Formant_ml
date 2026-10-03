"""LF Rd 하한 (`glottis.LF_RD_MIN`, §52.149) — 더 날카로운 폐쇄를 허용한다."""
import numpy as np
import pytest

from formant_ml.engine import glottis as G


def _shape(rd_min):
    old = G.LF_RD_MIN
    try:
        G.LF_RD_MIN = rd_min
        e = G.lf_pulse(0.0)                 # 하한으로 잘린다
    finally:
        G.LF_RD_MIN = old
    a = np.abs(e)
    S = np.abs(np.fft.rfft(e))
    return (a > 0.5 * a.max()).sum() / len(e), float(S[100:].sum() / S[1:].sum())


def test_default_is_the_classic_floor():
    assert G.LF_RD_MIN == 0.3


def test_lower_floor_sharpens_and_adds_high_frequency():
    fw30, hf30 = _shape(0.30)
    fw25, hf25 = _shape(0.25)
    fw22, hf22 = _shape(0.22)
    assert fw30 > fw25 > fw22, (fw30, fw25, fw22)
    assert hf22 > hf25 > hf30
    # 0.30 -> 0.22 로 고역이 5 dB 넘게 는다 (실측 −10.53 -> −5.20)
    assert 10 * np.log10(hf22 / hf30) > 4.5


def test_parameterisation_guard():
    """ra = (−1 + 4.8·Rd)/100 이 0 이하가 되는 곳은 막아야 한다 (Rd ≤ 0.2083)."""
    fw, hf = _shape(0.05)                  # 막히지 않으면 발산하거나 NaN 이다
    assert np.isfinite(fw) and np.isfinite(hf)
    fw212, _ = _shape(0.212)
    assert abs(fw - fw212) < 1e-9, "하한 아래 값은 0.212 로 고정돼야 한다"


@pytest.mark.parametrize("rd_min", [0.30, 0.25, 0.22])
def test_pulse_stays_finite_and_zero_mean(rd_min):
    old = G.LF_RD_MIN
    try:
        G.LF_RD_MIN = rd_min
        e = G.lf_pulse(0.0)
    finally:
        G.LF_RD_MIN = old
    assert np.isfinite(e).all()
    assert abs(float(e.sum()) / len(e)) < 2e-3, "LF 는 한 주기 적분이 0 이다"


def test_table_follows_the_floor():
    old = G.LF_RD_MIN
    try:
        G.LF_RD_MIN = 0.22
        rds, _ = G.lf_table(n_rd=8)
    finally:
        G.LF_RD_MIN = old
    assert abs(float(rds[0]) - 0.22) < 1e-6
