"""합격 자(`scripts/acceptance.py`)의 **특이성** — 자가 틀리면 판정이 통째로 무의미하다.

세로 얼룩의 자는 "에너지가 모든 주파수에서 동시에 급변한다" 를 잡아야 하고,
그 외의 것(전체 이득, 매끈한 포락선 변화)에는 둔해야 한다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import acceptance as A  # noqa: E402

FS = 48000.0


def _speechlike(n=int(1.0 * 48000), seed=0):
    """공진기를 지난 펄스 열 — **주기적**이라 프레임 간 대역 에너지가 안정적이다.

    백색 잡음으로 시험하면 안 된다. 잡음은 그 자체로 대역 에너지가 프레임마다 크게
    흔들려(플럭스 p95 가 9 dB) 골을 뚫어도 묻힌다. 실제 목표 녹음의 플럭스 p95 는
    4.8 dB 로 훨씬 낮다 — 말소리가 주기적이기 때문이다.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[::int(FS / 200)] = 1.0
    ir = np.exp(-np.arange(400) / 60.0) * np.sin(2 * np.pi * 900 * np.arange(400) / FS)
    x = np.convolve(x, ir)[:n] + 0.01 * rng.standard_normal(n)
    t = np.arange(n) / FS
    return x * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t))


def test_flux_is_one_against_itself():
    x = _speechlike()
    live = np.ones(len(x) // 240, bool)
    f = A.flux(x, live)
    assert abs(np.percentile(f, 95) / np.percentile(f, 95) - 1.0) < 1e-9


def test_flux_rises_when_gaps_are_punched_in():
    """구간 안에 페이드 인/아웃이 되풀이되면 — 사용자가 말한 그 현상 — 값이 올라야 한다."""
    x = _speechlike()
    live = np.ones(len(x) // 240, bool)
    base = np.percentile(A.flux(x, live), 95)
    y = x.copy()
    w = int(0.004 * FS)
    for a in range(int(0.1 * FS), len(y) - w, int(0.03 * FS)):
        y[a:a + w] *= 0.05                      # 4 ms 짜리 골을 30 ms 마다
    got = np.percentile(A.flux(y, live), 95)
    assert got > 1.5 * base


def test_flux_ignores_a_constant_gain():
    """전체 이득에는 둔해야 한다 — dB 차분이라 원리적으로 그래야 하고, 그걸 건다."""
    x = _speechlike()
    live = np.ones(len(x) // 240, bool)
    a = np.percentile(A.flux(x, live), 95)
    b = np.percentile(A.flux(x * 0.1, live), 95)
    assert abs(a - b) < 0.05 * a


def test_centroid_trajectory_follows_a_sweep():
    """무게중심 궤적이 실제 스윕을 따라가야 한다 — 안 그러면 '추이 일치' 가 무의미하다."""
    n = int(1.0 * FS)
    t = np.arange(n) / FS
    f = 2000.0 + 6000.0 * t
    x = np.sin(2 * np.pi * np.cumsum(f) / FS)
    live = np.ones(n // 480, bool)
    cen, _ = A.centroid_tilt(x, live)
    assert cen[0] < cen[-1] - 3000.0
    assert np.corrcoef(cen, np.arange(len(cen)))[0, 1] > 0.98


def test_envelope_db_tracks_a_ramp():
    n = int(0.5 * FS)
    x = np.random.default_rng(1).standard_normal(n) * np.linspace(1.0, 0.01, n)
    e = A.envelope_db(x, 25.0)
    assert e[0] > e[-1] + 25.0
    assert np.all(np.diff(e) < 2.0)
