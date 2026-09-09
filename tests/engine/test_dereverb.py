"""잔향 제거의 **성질**을 건다 — 무엇을 하고 무엇을 못 하는지.

이 모듈은 늦은 잔향을 크기 스펙트럼에서 뺀다. 그래서 **꼬리는 지우지만 초기 반사가
만든 포락선의 매끄러움은 되돌리지 못한다** (빼기는 합성곱의 역이 아니다).
그 한계를 실측으로 확인해 두었으므로(docs/MEASUREMENTS.md §20) 여기서도 건다 —
누가 "이걸로 치찰음 지글거림이 잡힌다" 고 기대하지 않도록.
"""
import numpy as np

from formant_ml.engine.dereverb import dereverb, reverb_ratio


def _reverb(x, fs, rt60=0.30, seed=3):
    rng = np.random.default_rng(seed)
    n = int(1.5 * rt60 * fs)
    t = np.arange(n) / fs
    ir = rng.standard_normal(n) * np.exp(-6.907755 * t / rt60)
    ir[0] += 6.0 * np.sqrt((ir ** 2).sum())          # 강한 직접음
    ir /= np.sqrt((ir ** 2).sum())
    return np.convolve(x, ir)[:len(x)]


def _burst(fs=48000.0, dur=1.0, seed=1):
    """짧은 잡음 버스트 열 — 잔향의 꼬리가 눈에 띄는 신호."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    x = rng.standard_normal(n) * 0.1
    env = np.zeros(n)
    for k in range(4):
        a = int((0.1 + 0.22 * k) * fs)
        env[a:a + int(0.05 * fs)] = 1.0
    return x * env


def test_removes_the_tail_between_bursts():
    """버스트 사이의 **꼬리**가 실제로 줄어야 한다 — 이게 이 모듈의 일이다."""
    fs = 48000.0
    dry = _burst(fs)
    wet = _reverb(dry, fs)
    out = dereverb(wet, fs)
    gap = slice(int(0.17 * fs), int(0.21 * fs))       # 첫 버스트가 끝난 직후
    e_wet = float((wet[gap] ** 2).mean())
    e_out = float((out[gap] ** 2).mean())
    assert e_out < 0.5 * e_wet
    assert 0.0 < reverb_ratio(wet, out) < 100.0


def test_keeps_the_direct_sound():
    """직접음 구간은 거의 그대로 남아야 한다 — 빼기가 신호를 갉으면 안 된다."""
    fs = 48000.0
    dry = _burst(fs)
    wet = _reverb(dry, fs)
    out = dereverb(wet, fs)
    on = slice(int(0.11 * fs), int(0.14 * fs))
    r = float((out[on] ** 2).mean()) / float((wet[on] ** 2).mean())
    assert 0.4 < r < 1.2


def test_restores_a_gated_envelope_when_the_model_holds():
    """방이 메운 골을 되돌린다 — **모형이 맞을 때는.**

    합성 잔향(지수 감쇠, 아는 RT60)에서는 잘 된다: 게이트가 걸린 잡음의 포락선
    거칠기가 마름 2.25 → 잔향 1.98 로 떨어지고, 제거 뒤 2.16 으로 **격차의 67 %**
    가 돌아온다.

    그런데 **실제 목표에서는 안 됐다** — 마찰 구간의 변조 지표가 4.12 → 3.96 배로
    4 % 밖에 안 움직였고 초기창을 줄여 에너지의 91 % 를 빼도 오히려 나빠졌다.
    즉 그 녹음의 마찰음이 매끄러운 이유는 **늦은 잔향 모형의 모양이 아니다**
    (docs/MEASUREMENTS.md §20). 이 대비가 이 테스트의 요점이다.
    """
    from scipy.signal import hilbert
    fs = 48000.0
    dry = _burst(fs)                                  # 게이트가 걸린 신호
    wet = _reverb(dry, fs, rt60=0.30)

    def rough(x):
        e = np.abs(hilbert(x))
        return float(e.std() / (e.mean() + 1e-12))

    r_dry, r_wet, r_out = rough(dry), rough(wet), rough(dereverb(wet, fs))
    assert r_wet < r_dry                              # 방이 골을 메워 매끄러워진다
    frac = (r_out - r_wet) / (r_dry - r_wet)
    assert 0.4 < frac < 1.2                           # 격차의 상당 부분이 돌아온다


def test_stationary_noise_is_not_smoothed_by_a_room():
    """선형 필터는 정상 가우시안의 포락선 통계를 안 바꾼다 — 기제를 오독하지 않도록.

    "잔향이 난류를 매끄럽게 한다" 는 설명은 **성립하지 않는다.** 방이 지표를 낮추는
    것은 신호가 비정상일 때(위 테스트)이지 정상 난류 그 자체가 아니다.
    """
    from scipy.signal import hilbert
    fs = 48000.0
    x = np.random.default_rng(11).standard_normal(int(0.5 * fs)) * 0.1
    y = _reverb(x, fs, rt60=0.30)

    def rough(z):
        e = np.abs(hilbert(z))
        return float(e.std() / (e.mean() + 1e-12))

    assert abs(rough(y) - rough(x)) < 0.05 * rough(x)
