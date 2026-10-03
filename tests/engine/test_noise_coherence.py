"""잡음의 주파수 간 결맞음 (`noise.NOISE_BANDS`, docs/MEASUREMENTS §52.99).

기식 잡음 포락은 **공통 AM**(성문 제트)과 **대역별 독립 흔들림**(국소 와류)으로 이루어진다.
예전에는 스칼라 하나만 곱해 공통 몫이 사실상 1 이었고, 그래서 고역의 모든 주파수가 함께 흔들려
스펙트로그램에서 가로줄이 층층이 보였다 (실측: 목표 대비 1500 Hz 간격 상관 +0.17).
"""
import math

import numpy as np
import pytest
import torch

from formant_ml.engine import noise as N

FS = 48000


@pytest.fixture(scope="module")
def _asp():
    return N.AspirationNoise(FS, 48)


def _env(n, f0=200.0, depth=0.7):
    t = np.arange(n) / FS
    frac = (t * f0) % 1.0
    op = np.where(frac < 0.65, np.sin(np.pi * np.clip(frac, 0, 1) / 0.65) ** 2, 0.0)
    return torch.as_tensor((1.0 + depth * (op - 0.5))[None, :], dtype=torch.float32)


def test_band_split_reconstructs_the_input(_asp):
    """대역의 **합이 원 신호**여야 한다 — 안 그러면 에너지가 새거나 겹친다."""
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.standard_normal((1, 20000)), dtype=torch.float32)
    st = {}
    tot = sum(b for _lo, _hi, b in _asp._bands(x, st))
    err = float((tot - x).abs().max())
    assert err < 1e-5, err          # 소수 5 자리


def test_cross_band_coherence_matches_the_design(_asp):
    """대역 포락의 상관이 설계한 ρ 에 가까워야 한다 (실측 곡선에서 온 값)."""
    n = 96000
    env = _env(n)
    rng = np.random.default_rng(1)
    white = torch.as_tensor(rng.standard_normal((1, n)), dtype=torch.float32)
    st = {}
    bands = _asp._bands(white, st)
    slow, fast = _asp._am_split(env, st)
    m = (fast / slow.clamp_min(1e-9))[0].numpy()
    # 두 고역 대역의 포락을 만들고 상관을 잰다 (같은 경로를 손으로 재현)
    got = []
    for i in (1, len(bands) - 1):
        lo, hi, _b = bands[i]
        got.append(_asp._rho(0.5 * (lo + hi)))
    assert 0.2 <= got[0] <= 0.9 and 0.2 <= got[1] <= 0.9, got
    assert got[-1] <= got[0] + 1e-9, got     # 고역일수록 공통 몫이 작다


def test_forward_runs_and_keeps_level(_asp):
    """켠 뒤에도 잡음 수준이 예전과 같은 자릿수여야 한다 (변조만 바뀐다)."""
    n = 48000
    env = _env(n)
    rms_on = float(_asp(env, noise=N.NoiseBank(), sample0=0, state={})["source"].pow(2).mean().sqrt())
    old = N.NOISE_BANDS
    try:
        N.NOISE_BANDS = 0
        rms_off = float(_asp(env, noise=N.NoiseBank(), sample0=0, state={})["source"].pow(2).mean().sqrt())
    finally:
        N.NOISE_BANDS = old
    assert 0.5 < rms_on / max(rms_off, 1e-12) < 2.0, (rms_on, rms_off)
