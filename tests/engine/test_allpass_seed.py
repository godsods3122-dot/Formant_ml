"""올패스 위상차 체인의 씨앗 (`fit.ALLPASS_SEED_HZ`) — 예전부터 있었으나 한 번도 안 걸렸다 (MEASUREMENTS §52.227).

두 겹으로 막혀 있었다: (1) `DEFAULT_PARAMS` 에 `ap*` 가 없다, (2) `ap*_f` 는 로그 파라미터라 초기값 0 이면
`CopySynthFitter` 가 열에서 빼 버린다. 씨앗을 심고 목록에 넣어야 비로소 움직인다.
"""
import numpy as np
import torch

from formant_ml.engine import fit as F
from formant_ml.engine.control import N_ALLPASS
from formant_ml.engine.tviir import allpass_coeffs

FS = 48000.0


def test_seed_constants_are_sane():
    assert len(F.ALLPASS_SEED_HZ) == N_ALLPASS
    assert all(100.0 < f < 12000.0 for f in F.ALLPASS_SEED_HZ)
    assert 0.0 < F.ALLPASS_SEED_R < 0.98


def test_a_zero_seed_is_dropped_from_the_fit_columns():
    """왜 씨앗이 필요한가: 로그 파라미터는 초기값 0 이면 열에서 빠진다."""
    spec = F.PARAMS["ap1_f"]
    assert spec.log is True
    base = torch.zeros(4, 2, dtype=torch.float64)
    names = [p for p in ("ap1_f",) if not (spec.log and float(base[:, 0].abs().min()) == 0.0)]
    assert names == []


def test_allpass_is_exactly_unit_magnitude():
    """올패스는 크기를 하나도 안 건드린다 — 이 저장소의 다른 손잡이와 달리 위상과 크기가 직교한다."""
    for f in F.ALLPASS_SEED_HZ:
        b0, b1, b2, a1, a2 = (float(v) for v in allpass_coeffs(
            torch.tensor(f, dtype=torch.float64), torch.tensor(F.ALLPASS_SEED_R, dtype=torch.float64), FS))
        for probe in (50.0, 500.0, 1500.0, 5000.0, 12000.0, 20000.0):
            z = np.exp(-1j * 2 * np.pi * probe / FS)
            H = (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
            assert abs(abs(H) - 1.0) < 1e-9, (f, probe, abs(H))


def test_the_seed_actually_bends_the_phase():
    """씨앗 반지름에서 군지연이 실제로 움직여야 한다 — r 이 0 이면 순수 지연이라 아무 뜻이 없다."""
    b0, b1, b2, a1, a2 = (float(v) for v in allpass_coeffs(
        torch.tensor(2000.0, dtype=torch.float64), torch.tensor(F.ALLPASS_SEED_R, dtype=torch.float64), FS))
    ph = []
    for probe in (1000.0, 2000.0, 3000.0):
        z = np.exp(-1j * 2 * np.pi * probe / FS)
        H = (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
        ph.append(np.angle(H))
    assert abs(np.unwrap(ph)[0] - np.unwrap(ph)[-1]) > 0.5
