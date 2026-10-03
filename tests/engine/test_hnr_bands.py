"""HNR 항의 **통과 대역을 여러 개** 받는다 (`fit.HNR_BANDS`).

대역 하나를 상수로 박아 두면 결함이 옮겨갔을 때 손실이 못 본다. 실측(§52.174): `HNR_BP_HZ`
(1.5~3 kHz) 의 결함은 고쳐졌는데(목표 대비 +0.000~+0.016) 3~4.5 kHz 가 +0.075~+0.138 로 남았다.
"""
import numpy as np
import pytest
import torch

from formant_ml.engine import fit as F


class _Stub:
    _periodicity = F.CopySynthFitter._periodicity

    def __init__(self, bands, fs=48000.0, n=4800):
        self.fs = fs
        self.target = torch.zeros(1, n, dtype=torch.float64)
        self._corr_win, self._corr_hop = 480, 240
        self._per_ok = torch.ones(8, dtype=torch.bool)
        self._per_lag = np.full(8, 138)
        k = 129
        t = np.arange(k) - k // 2
        lp = np.sinc(2 * (F.HNR_HP_HZ / fs) * t) * np.hanning(k)
        hp = -lp / lp.sum()
        hp[k // 2] += 1.0
        self._hp = torch.as_tensor(hp, dtype=torch.float64).view(1, 1, k)
        self._bps = []
        for lo, hi in bands:
            if not (lo and hi) or not hi > lo > 0.0 or hi >= 0.5 * fs:
                continue
            a = np.sinc(2 * (hi / fs) * t) * np.hanning(k)
            b = np.sinc(2 * (lo / fs) * t) * np.hanning(k)
            self._bps.append(torch.as_tensor(a / a.sum() - b / b.sum(),
                                             dtype=torch.float64).view(1, 1, k))
        self._bp = self._bps[0] if self._bps else None


def _sig(n=4800, fs=48000.0, seed=0):
    t = np.arange(n) / fs
    y = sum(np.cos(2 * np.pi * k * 347.0 * t) for k in range(1, 14))
    y = y + 0.3 * np.random.default_rng(seed).standard_normal(n)
    return torch.tensor(y, dtype=torch.float64).reshape(1, -1)


def test_the_default_keeps_exactly_three_bands():
    """기본값을 바꾸지 않는다 — 전대역 · >1 kHz · HNR_BP_HZ."""
    assert F.HNR_BANDS == ()
    assert _Stub([F.HNR_BP_HZ])._periodicity(_sig()).shape[0] == 3


def test_each_extra_band_adds_a_row():
    s = _Stub([F.HNR_BP_HZ, (3000.0, 4500.0), (4500.0, 6500.0)])
    assert s._periodicity(_sig()).shape[0] == 5


def test_the_extra_rows_are_not_copies():
    """대역이 정말 다르게 걸러져야 한다 — 같은 값이면 필터가 안 먹은 것이다."""
    r = _Stub([F.HNR_BP_HZ, (3000.0, 4500.0)])._periodicity(_sig())
    assert not torch.allclose(r[2], r[3], atol=1e-6)


@pytest.mark.parametrize("bad", [(0.0, 3000.0), (4500.0, 3000.0), (3000.0, 3000.0),
                                 (3000.0, 30000.0)])
def test_a_nonsense_band_is_dropped_not_crashed(bad):
    """나이퀴스트 위·뒤집힌·빈 대역은 조용히 버린다 — 적합을 중간에 죽이지 않는다."""
    assert _Stub([F.HNR_BP_HZ, bad])._periodicity(_sig()).shape[0] == 3


def test_the_old_attribute_still_points_at_the_first_band():
    """`_bp` 를 쓰는 진단이 있다 — 이름을 지우지 않는다."""
    s = _Stub([F.HNR_BP_HZ, (3000.0, 4500.0)])
    assert s._bp is s._bps[0]
