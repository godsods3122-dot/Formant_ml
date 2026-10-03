"""봉우리형 보조 극 (`tract.AUX_PEAK`) — 고역 지직거림의 구조 결함 수정 (MEASUREMENTS §52.198~199).

예전 형태 `y = x + mix·(R(x) − x)` 에서 R 은 DC 정규화 공진기라 극보다 한참 위에서 0 이다. 그래서 고역 이득이 (1 − mix) 가
되어 5 kHz 아래 국소 공진을 켜기만 해도 12~17 kHz 전체가 내려간다. 봉우리 필터(먼 대역 이득 1)로 바꾸면 고역에 손대지 않는다.
"""
import math

import numpy as np
import pytest
import torch

from formant_ml.engine import tract as T
from formant_ml.engine.tviir import peak_coeffs, peaking_coeffs, resonator_coeffs

FS = 48000.0


def _gain_db(coeffs, mix, f):
    b0, b1, b2, a1, a2 = (float(v) for v in coeffs)
    z = np.exp(-1j * 2 * np.pi * f / FS)
    H = (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
    return 20 * np.log10(abs(1 + mix * (H - 1)))


def _c(fn, F, B, *extra):
    return fn(torch.tensor(F, dtype=torch.float64), torch.tensor(B, dtype=torch.float64), FS, *extra)


@pytest.mark.parametrize("F,B", [(1015.0, 145.0), (2409.0, 257.0), (4525.0, 426.0)])
def test_old_form_drags_the_whole_high_band_down(F, B):
    """결함의 재현: 예전 공진기는 mix 0.3 에서 15 kHz 를 약 −3 dB 끌어내린다 — 극 자리와 무관하게."""
    g = _gain_db(_c(resonator_coeffs, F, B), 0.3, 15000.0)
    assert -3.3 < g < -2.7, g


@pytest.mark.parametrize("F,B", [(1015.0, 145.0), (2409.0, 257.0), (4525.0, 426.0)])
@pytest.mark.parametrize("mix", [0.1, 0.3, 0.6, 1.0])
def test_peak_form_leaves_the_high_band_alone(F, B, mix):
    for f in (12000.0, 14500.0, 17000.0):
        g = _gain_db(_c(peaking_coeffs, F, B, T.AUX_PEAK_RATIO), mix, f)
        assert abs(g) < 0.35, (F, mix, f, g)


def test_peak_form_keeps_the_local_effect_size():
    """자리 이득은 1 + mix·(비 − 1) 쯤 — 예전 공진기가 mix 0.3 에서 내던 +9.4 dB 근처여야 한다."""
    F, B = 2409.0, 257.0
    g_new = _gain_db(_c(peaking_coeffs, F, B, T.AUX_PEAK_RATIO), 0.3, F)
    g_old = _gain_db(_c(resonator_coeffs, F, B), 0.3, F)
    assert abs(g_new - 20 * math.log10(1 + 0.3 * (T.AUX_PEAK_RATIO - 1))) < 1.0
    assert abs(g_new - g_old) < 2.0, (g_new, g_old)


def test_zero_mix_is_identity_for_both_forms():
    for fn, extra in ((resonator_coeffs, ()), (peaking_coeffs, (T.AUX_PEAK_RATIO,))):
        for f in (300.0, 2409.0, 15000.0):
            assert abs(_gain_db(_c(fn, 2409.0, 257.0, *extra), 0.0, f)) < 1e-9


def test_pole_zero_peak_is_not_flat_enough():
    """왜 `peak_coeffs` 가 아닌가: 영점을 넓힌 극-영점 쌍은 고역 끝 이득이 1 이 아니다 (§52.199)."""
    g = _gain_db(_c(peak_coeffs, 1015.0, 145.0, T.AUX_PEAK_RATIO), 0.6, 12000.0)
    assert g < -0.8, g


def test_peaking_is_exactly_unity_at_dc_and_nyquist():
    for F, B in ((1015.0, 145.0), (4525.0, 426.0)):
        for f in (0.0, FS / 2):
            assert abs(_gain_db(_c(peaking_coeffs, F, B, T.AUX_PEAK_RATIO), 1.0, f)) < 1e-6


def test_off_by_default():
    """기본값을 바꾸지 않는다 — 예전 판들의 제어열은 예전 형태의 뜻으로 적합됐다."""
    assert T.AUX_PEAK is False
